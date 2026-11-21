"""M6 -> M7 (PyTorch MLP) -> MLflow -> published package -> M8, end to end.

Everything is synthetic and offline: M5-shaped rows go through the real M6
runner, the real M7 ``TrainingRunner`` with ``model_type="mlp"``, a local
SQLite MLflow, ``publish_model`` into a local directory and an in-memory S3
stand-in, and the FastAPI M8 app. No OpenF1, no AWS. The numbers only prove
that the technical chain holds; they are not academic results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest

pytest.importorskip("torch", reason="install the optional dl extra")
mlflow = pytest.importorskip("mlflow", reason="install the optional training extra")
pytest.importorskip("fastapi", reason="install the optional serve extra")

from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling.dl import CHECKPOINT_FILENAME
from f1_race_intelligence.modeling.model_package import (
    DEFAULT_PACKAGE_ROOT,
    MANIFEST_FILE,
    PUBLISHED_FILE,
    publish_model,
    read_model_package,
)
from f1_race_intelligence.modeling.training import TrainingResult, TrainingRunner
from f1_race_intelligence.serving import app as app_module
from f1_race_intelligence.serving.app import create_app
from f1_race_intelligence.serving.contract import ArtifactError
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
from serving_support import M6Run, build_m6_run, to_records
from s3_fakes import FakeS3Client

REGISTERED_NAME = "synthetic-dl-m7-m8-mlp"
PACKAGE_FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MLP_SETTINGS = {
    "registered_model_name": REGISTERED_NAME,
    "hidden_dims": [16, 8],
    "dropout": 0.0,
    "learning_rate": 0.01,
    "batch_size": 8,
    "max_epochs": 30,
    "early_stopping_patience": 5,
}
REQUIRED_PACKAGE_FILES = {
    "data_provenance.json",
    "training_config.json",
    "training_report.json",
    "m6/feature_schema.json",
    "m6/preprocessor.joblib",
    "model/MLmodel",
}


@dataclass(frozen=True)
class MLPRuns:
    m6: M6Run
    m6_report: Dict[str, Any]
    settings: AppSettings
    tracking_uri: str
    first: TrainingResult
    second: TrainingResult


@pytest.fixture(scope="module")
def mlp_runs(tmp_path_factory) -> Iterator[MLPRuns]:
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    root = tmp_path_factory.mktemp("dl-m7-m8")
    m6 = build_m6_run(root / "m6")
    (report_path,) = (m6.output_dir / "reports").glob("modeling_report_*.json")
    settings = AppSettings(
        training={
            "m6_report_path": str(report_path),
            "tracking_db_path": str(root / "mlflow" / "tracking.db"),
            "artifact_path": str(root / "mlflow" / "artifacts"),
            "output_path": str(root / "training-reports"),
            "experiment_name": "synthetic-dl-m7-m8-integration",
            "model_type": "mlp",
            "mlp": MLP_SETTINGS,
        }
    )
    first = TrainingRunner(settings, m6_report_path=str(report_path)).run()
    second = TrainingRunner(settings, m6_report_path=str(report_path)).run()

    tracking_uri = f"sqlite:///{(root / 'mlflow' / 'tracking.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    m6_report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        yield MLPRuns(m6, m6_report, settings, tracking_uri, first, second)
    finally:
        # do not leak this module's SQLite tracking/registry URIs into other test modules
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


@pytest.fixture
def no_refit(monkeypatch) -> None:
    """Any attempt to fit the M6 ColumnTransformer during serving fails the test."""

    def forbidden_fit(*args, **kwargs):
        pytest.fail("M8 must only transform with the M6 preprocessor; it must never fit it")

    monkeypatch.setattr(ColumnTransformer, "fit", forbidden_fit)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden_fit)


def _download(run_id: str, artifact: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact}"))


def _m6_artifact_path(m6_report: Dict[str, Any], split: str, artifact: str) -> Path:
    (item,) = [
        entry
        for entry in m6_report["artifact_files"]
        if entry["split"] == split and entry["artifact"] == artifact
    ]
    return Path(item["path"])


def _test_rows_and_records(runs: MLPRuns, count: int = 3):
    rows = runs.m6.split_file("test", "dataset.parquet").head(count)
    records = to_records(rows, list(runs.m6.numeric) + list(runs.m6.categorical))
    return rows, records


def _expected_seconds(runs: MLPRuns, count: int) -> np.ndarray:
    """The registered pyfunc's answer on M6's own encoded test matrix."""
    import pandas as pd

    columns = runs.m6_report["preprocessing"]["output_feature_names"]
    matrix = pd.read_parquet(_m6_artifact_path(runs.m6_report, "test", "features.parquet"))
    model = mlflow.pyfunc.load_model(runs.first.model_uri)
    return np.asarray(model.predict(matrix.loc[:, columns].head(count).astype("float64")), dtype="float64")


def _assert_seconds_scale(values: List[float], actual_seconds: List[float], y_mean: float) -> None:
    """Predictions are next_lap_time in seconds (~90 s), never standardised units (~0)."""
    assert all(np.isfinite(values))
    assert all(80.0 <= value <= 110.0 for value in values)
    assert all(abs(value) > 50.0 for value in values)
    assert all(90.0 <= value <= 100.0 for value in actual_seconds)
    assert float(np.mean(np.abs(np.asarray(values) - np.asarray(actual_seconds)))) < 10.0
    assert abs(float(np.mean(values)) - y_mean) < 10.0


# --------------------------------------------------------------------------- #
# M7 registration
# --------------------------------------------------------------------------- #


def test_m7_mlp_registers_a_pyfunc_with_ordered_signature_and_m6_hashes(mlp_runs) -> None:
    runs = mlp_runs
    result = runs.first
    assert result.model_type == "mlp"
    assert result.model_name == REGISTERED_NAME
    assert result.model_uri == f"models:/{REGISTERED_NAME}/{result.model_version}"
    assert runs.second.model_uri != result.model_uri
    assert runs.second.metrics == result.metrics  # deterministic CPU training

    model = mlflow.pyfunc.load_model(result.model_uri)
    version = MlflowClient().get_model_version(REGISTERED_NAME, result.model_version)
    assert version.run_id == result.run_id
    assert version.status == "READY"
    assert model.metadata.run_id == result.run_id
    assert "python_function" in model.metadata.flavors
    assert "sklearn" not in model.metadata.flavors

    schema = json.loads(_download(result.run_id, "m6/feature_schema.json").read_text(encoding="utf-8"))
    encoded = schema["encoded_feature_names"]
    assert encoded == runs.m6_report["preprocessing"]["output_feature_names"]
    assert list(model.metadata.get_input_schema().input_names()) == encoded
    assert "next_lap_time" not in encoded
    assert not any(name.startswith("key_") for name in encoded)

    provenance = json.loads(_download(result.run_id, "data_provenance.json").read_text(encoding="utf-8"))
    assert provenance["m6_run_id"] == result.m6_run_id == runs.m6.run_id
    assert provenance["input_fingerprint"] == result.input_fingerprint
    assert provenance["m6_report_sha256"] == hashlib.sha256(
        Path(runs.settings.training.m6_report_path).read_bytes()
    ).hexdigest()
    for artifact in ("preprocessor.joblib", "feature_schema.json"):
        (entry,) = [
            item
            for item in provenance["verified_artifact_hashes"]
            if item["artifact"] == artifact and item["split"] is None
        ]
        downloaded = _download(result.run_id, f"m6/{artifact}")
        assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == entry["sha256"]
        assert hashlib.sha256(_m6_artifact_path(runs.m6_report, None, artifact).read_bytes()).hexdigest() == entry["sha256"]

    report = json.loads(_download(result.run_id, "training_report.json").read_text(encoding="utf-8"))
    assert report["run_id"] == result.run_id
    assert report["model_type"] == "mlp"
    assert report["m6_run_id"] == result.m6_run_id
    assert report["input_fingerprint"] == result.input_fingerprint
    assert report["registered_model"]["uri"] == result.model_uri
    assert report["registered_model"]["recovery_check"] == "passed"
    assert report["config"] == json.loads(_download(result.run_id, "training_config.json").read_text(encoding="utf-8"))
    assert set(report["metrics"]["test"]) == {"mlp", "linear_regression", "reference_mean"}


def test_m7_mlp_model_directory_has_no_symlinks_and_package_safe_names(mlp_runs) -> None:
    model_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=mlp_runs.first.model_uri))
    files = [path for path in model_dir.rglob("*")]

    assert (model_dir / "MLmodel").is_file()
    assert any(path.name == CHECKPOINT_FILENAME for path in files)
    assert not any(path.is_symlink() for path in files)
    for path in files:
        for part in path.relative_to(model_dir).parts:
            assert PACKAGE_FILE_PATTERN.fullmatch(part), part


# --------------------------------------------------------------------------- #
# Immutable publication
# --------------------------------------------------------------------------- #


def test_mlp_package_is_identical_in_local_and_simulated_s3_and_idempotent(mlp_runs, tmp_path) -> None:
    runs = mlp_runs
    local = LocalStorageBackend(tmp_path / "local")
    s3_client = FakeS3Client()
    s3 = S3StorageBackend("dl-model-package-test", client=s3_client)

    local_package = publish_model(runs.first.model_uri, local)
    s3_package = publish_model(runs.first.model_uri, s3)

    assert local_package.prefix == s3_package.prefix == f"{DEFAULT_PACKAGE_ROOT}/{REGISTERED_NAME}/versions/{runs.first.model_version}"
    assert local_package.manifest == s3_package.manifest
    assert local_package.artifacts == s3_package.artifacts
    assert local_package.manifest_sha256 == s3_package.manifest_sha256
    assert local.read_bytes(f"{local_package.prefix}/{PUBLISHED_FILE}") == s3.read_bytes(
        f"{s3_package.prefix}/{PUBLISHED_FILE}"
    )
    assert local.read_bytes(f"{local_package.prefix}/{PUBLISHED_FILE}").decode("ascii").strip() == (
        f"sha256:{local_package.manifest_sha256}"
    )

    manifest = local_package.manifest
    files = set(manifest["files"])
    assert files == set(local_package.artifacts)
    assert REQUIRED_PACKAGE_FILES <= files
    assert any(name.startswith("model/") and name.endswith(f"/{CHECKPOINT_FILENAME}") for name in files)
    for name in files:
        assert all(PACKAGE_FILE_PATTERN.fullmatch(part) for part in name.split("/")), name
    for name, content in local_package.artifacts.items():
        assert manifest["files"][name]["sha256"] == hashlib.sha256(content).hexdigest()
        assert manifest["files"][name]["size_bytes"] == len(content)

    assert manifest["model"]["name"] == REGISTERED_NAME
    assert manifest["model"]["uri"] == runs.first.model_uri
    assert manifest["model"]["m7_run_id"] == runs.first.run_id
    assert manifest["model"]["target"] == "next_lap_time"
    assert manifest["model"]["input_signature"] == runs.m6_report["preprocessing"]["output_feature_names"]
    assert manifest["provenance"]["m6_run_id"] == runs.first.m6_run_id
    assert manifest["provenance"]["input_fingerprint"] == runs.first.input_fingerprint
    assert manifest["provenance"]["data_version"] == f"m6:{runs.first.m6_run_id}:{runs.first.input_fingerprint}"

    # the published files on disk are regular files, never symlinks
    on_disk = list((tmp_path / "local").rglob("*"))
    assert on_disk and not any(path.is_symlink() for path in on_disk)

    # republishing the same registered version is a verified no-op
    local_keys = sorted(local.list(local_package.prefix))
    s3_keys = sorted(s3.list(s3_package.prefix))
    again_local = publish_model(runs.first.model_uri, local)
    again_s3 = publish_model(runs.first.model_uri, s3)
    assert again_local.manifest_sha256 == again_s3.manifest_sha256 == local_package.manifest_sha256
    assert again_local.artifacts == local_package.artifacts
    assert sorted(local.list(local_package.prefix)) == local_keys
    assert sorted(s3.list(s3_package.prefix)) == s3_keys
    assert f"{local_package.prefix}/{MANIFEST_FILE}" in local_keys
    assert read_model_package(local, local_package.prefix).manifest_sha256 == local_package.manifest_sha256
    assert read_model_package(s3, s3_package.prefix).manifest_sha256 == s3_package.manifest_sha256


# --------------------------------------------------------------------------- #
# M8 serving
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend_kind", ["local", "s3"])
def test_m8_serves_the_published_mlp_package_without_refitting(
    mlp_runs, tmp_path, no_refit, backend_kind
) -> None:
    runs = mlp_runs
    if backend_kind == "local":
        backend = LocalStorageBackend(tmp_path / "store")
    else:
        backend = S3StorageBackend("dl-m8-package-test", client=FakeS3Client())
    package = publish_model(runs.first.model_uri, backend)
    service = PredictionService.load_published_package(
        backend=backend, prefix=package.prefix, loader=MlflowPyfuncLoader()
    )
    rows, records = _test_rows_and_records(runs)

    with TestClient(create_app(lambda: service)) as client:
        live = client.get("/live")
        health = client.get("/health")
        info = client.get("/model-info")
        response = client.post("/predict", json={"records": records})

    assert live.status_code == health.status_code == info.status_code == response.status_code == 200
    assert live.json() == health.json() == {"status": "ok"}
    info_json = info.json()
    assert info_json["model"]["uri"] == runs.first.model_uri
    assert info_json["model"]["run_id"] == runs.first.run_id
    assert info_json["model"]["input_signature_verified"] is True
    assert "python_function" in info_json["model"]["flavors"]
    assert info_json["preprocessing"]["uri"] == package.package_uri
    assert info_json["provenance"] == {
        "m7_run_id": runs.first.run_id,
        "m6_run_id": runs.first.m6_run_id,
        "input_fingerprint": runs.first.input_fingerprint,
    }

    body = response.json()
    assert body["target"] == "next_lap_time"
    assert body["model"] == {"uri": runs.first.model_uri, "run_id": runs.first.run_id}
    values = [item["value"] for item in body["predictions"]]
    assert [item["index"] for item in body["predictions"]] == list(range(len(records)))
    report = json.loads(package.artifacts["training_report.json"].decode("utf-8"))
    _assert_seconds_scale(values, rows["next_lap_time"].astype(float).tolist(), report["mlp_training"]["y_mean"])
    # M8 (raw records -> M6 transform -> pyfunc) agrees with the pyfunc on M6's encoded matrix
    np.testing.assert_allclose(values, _expected_seconds(runs, len(records)), rtol=1e-5, atol=1e-5)


def test_m8_serves_the_mlp_from_the_matched_m7_run_without_refitting(mlp_runs, monkeypatch, no_refit) -> None:
    runs = mlp_runs
    result = runs.first
    preprocessing_uri = f"runs:/{result.run_id}/m6"

    service = PredictionService.load(
        preprocessing_uri=preprocessing_uri, model_uri=result.model_uri, loader=MlflowPyfuncLoader()
    )
    assert service.model.run_id == result.run_id
    assert service.m6_run_id == result.m6_run_id
    assert service.input_fingerprint == result.input_fingerprint

    monkeypatch.setenv("MLFLOW_TRACKING_URI", runs.tracking_uri)
    monkeypatch.setenv(app_module.PREPROCESSING_URI_ENV, preprocessing_uri)
    monkeypatch.setenv(app_module.MODEL_URI_ENV, result.model_uri)
    rows, records = _test_rows_and_records(runs)
    with TestClient(create_app()) as client:
        live = client.get("/live")
        health = client.get("/health")
        info = client.get("/model-info").json()
        response = client.post("/predict", json={"records": records})

    assert live.json() == health.json() == {"status": "ok"}
    assert info["model"]["uri"] == result.model_uri
    assert info["model"]["run_id"] == result.run_id
    assert info["model"]["input_signature_verified"] is True
    assert info["preprocessing"]["uri"] == preprocessing_uri
    assert info["provenance"] == {
        "m7_run_id": result.run_id,
        "m6_run_id": result.m6_run_id,
        "input_fingerprint": result.input_fingerprint,
    }
    assert response.status_code == 200
    values = [item["value"] for item in response.json()["predictions"]]
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    _assert_seconds_scale(values, rows["next_lap_time"].astype(float).tolist(), report["mlp_training"]["y_mean"])
    np.testing.assert_allclose(values, service.predict(records), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(values, _expected_seconds(runs, len(records)), rtol=1e-5, atol=1e-5)


def test_m8_rejects_an_mlp_model_paired_with_another_runs_preprocessor(mlp_runs) -> None:
    runs = mlp_runs
    with pytest.raises(ArtifactError, match="belong to M7 run"):
        PredictionService.load(
            preprocessing_uri=f"runs:/{runs.first.run_id}/m6",
            model_uri=runs.second.model_uri,
            loader=MlflowPyfuncLoader(),
        )


def test_m8_rejects_invalid_records_for_the_mlp_with_422(mlp_runs, tmp_path, no_refit) -> None:
    runs = mlp_runs
    backend = LocalStorageBackend(tmp_path / "store")
    package = publish_model(runs.first.model_uri, backend)
    service = PredictionService.load_published_package(
        backend=backend, prefix=package.prefix, loader=MlflowPyfuncLoader()
    )
    _, records = _test_rows_and_records(runs, count=1)
    broken = {key: value for key, value in records[0].items() if key != runs.m6.numeric[0]}

    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/predict", json={"records": [broken]})
        empty = client.post("/predict", json={"records": []})

    assert response.status_code == 422
    assert empty.status_code == 422
