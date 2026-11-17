"""M7-to-M8 integration using synthetic M5 rows and real local pipeline code."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow", reason="install the optional training extra")
pytest.importorskip("fastapi", reason="install the optional serve extra")

from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling.training import TrainingResult, TrainingRunner
from f1_race_intelligence.serving import app as app_module
from f1_race_intelligence.serving.app import create_app
from f1_race_intelligence.serving.contract import ArtifactError
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService
from serving_support import M6Run, build_m6_run, to_records


@dataclass(frozen=True)
class SyntheticM7Runs:
    m6: M6Run
    settings: AppSettings
    first: TrainingResult
    second: TrainingResult


@pytest.fixture(scope="module")
def synthetic_m7_runs(tmp_path_factory) -> SyntheticM7Runs:
    root = tmp_path_factory.mktemp("m7-m8")
    m6 = build_m6_run(root / "m6")
    (report_path,) = (m6.output_dir / "reports").glob("modeling_report_*.json")
    settings = AppSettings(
        training={
            "m6_report_path": str(report_path),
            "tracking_db_path": str(root / "mlflow" / "tracking.db"),
            "artifact_path": str(root / "mlflow" / "artifacts"),
            "output_path": str(root / "training-reports"),
            "experiment_name": "synthetic-m7-m8-integration",
            "registered_model_name": "synthetic-m7-m8-integration",
        }
    )
    first = TrainingRunner(settings, m6_report_path=str(report_path)).run()
    second = TrainingRunner(settings, m6_report_path=str(report_path)).run()

    tracking_uri = f"sqlite:///{(root / 'mlflow' / 'tracking.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return SyntheticM7Runs(m6, settings, first, second)


def _download(run_id: str, artifact: str) -> Path:
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact}"))


def _assert_m7_hash(provenance: dict, artifact: str, path: Path) -> None:
    entries = [
        item
        for item in provenance["verified_artifact_hashes"]
        if item["artifact"] == artifact and item["split"] is None
    ]
    assert len(entries) == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[0]["sha256"]


def test_real_m7_registration_is_pyfunc_loadable_and_bound_to_its_artifacts(synthetic_m7_runs) -> None:
    runs = synthetic_m7_runs
    result = runs.first
    assert result.model_uri == f"models:/{result.model_name}/{result.model_version}"
    assert re.fullmatch(r"models:/[^/]+/[0-9]+", result.model_uri)
    assert runs.second.model_uri != result.model_uri

    model = mlflow.pyfunc.load_model(result.model_uri)
    version = MlflowClient().get_model_version(result.model_name, result.model_version)
    assert version.run_id == result.run_id
    assert model.metadata.run_id == result.run_id
    input_schema = model.metadata.get_input_schema()
    assert input_schema is not None
    assert list(input_schema.input_names()) == json.loads(
        _download(result.run_id, "m6/feature_schema.json").read_text(encoding="utf-8")
    )["encoded_feature_names"]

    provenance = json.loads(_download(result.run_id, "data_provenance.json").read_text(encoding="utf-8"))
    assert provenance["m6_run_id"] == result.m6_run_id
    assert provenance["input_fingerprint"] == result.input_fingerprint
    _assert_m7_hash(provenance, "preprocessor.joblib", _download(result.run_id, "m6/preprocessor.joblib"))
    _assert_m7_hash(provenance, "feature_schema.json", _download(result.run_id, "m6/feature_schema.json"))

    report = json.loads(_download(result.run_id, "training_report.json").read_text(encoding="utf-8"))
    assert report["run_id"] == result.run_id
    assert report["registered_model"]["uri"] == result.model_uri
    assert report["m6_run_id"] == result.m6_run_id


def test_fastapi_serves_the_matched_m7_run_without_refitting(synthetic_m7_runs, monkeypatch) -> None:
    runs = synthetic_m7_runs
    result = runs.first
    preprocessing_uri = f"runs:/{result.run_id}/m6"
    tracking_uri = f"sqlite:///{Path(runs.settings.training.tracking_db_path).resolve().as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv(app_module.PREPROCESSING_URI_ENV, preprocessing_uri)
    monkeypatch.setenv(app_module.MODEL_URI_ENV, result.model_uri)
    rows = runs.m6.split_file("test", "dataset.parquet").head(2)
    records = to_records(rows, runs.m6.numeric + runs.m6.categorical)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("serving must only transform; it must not fit M6 preprocessing")

    monkeypatch.setattr(ColumnTransformer, "fit", forbidden_fit)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden_fit)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/model-info").json()
        response = client.post("/predict", json={"records": records})

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
    body = response.json()
    assert body["target"] == "next_lap_time"
    assert len(body["predictions"]) == len(records)
    expected_seconds = rows["next_lap_time"].astype(float).tolist()
    assert all(90.0 <= value <= 100.0 for value in expected_seconds)
    assert all(80.0 <= item["value"] <= 110.0 for item in body["predictions"])


def test_model_and_preprocessor_from_different_m7_runs_are_rejected(synthetic_m7_runs) -> None:
    runs = synthetic_m7_runs
    with pytest.raises(ArtifactError, match="belong to M7 run"):
        PredictionService.load(
            preprocessing_uri=f"runs:/{runs.first.run_id}/m6",
            model_uri=runs.second.model_uri,
            loader=MlflowPyfuncLoader(),
        )

    with pytest.raises(ArtifactError, match="same M7 run"):
        PredictionService.load(
            preprocessing_uri=str(runs.m6.preprocessing_dir),
            model_uri=runs.first.model_uri,
            loader=MlflowPyfuncLoader(),
        )


def test_tampered_m7_preprocessor_fails_its_provenance_hash(synthetic_m7_runs, tmp_path, monkeypatch) -> None:
    runs = synthetic_m7_runs
    preprocessing_uri = f"runs:/{runs.first.run_id}/m6"
    original_download = mlflow.artifacts.download_artifacts

    def tampered_download(*, artifact_uri, **kwargs):
        downloaded = Path(original_download(artifact_uri=artifact_uri, **kwargs))
        if artifact_uri != preprocessing_uri:
            return str(downloaded)
        target = tmp_path / "m6"
        shutil.copytree(downloaded, target)
        preprocessor = target / "preprocessor.joblib"
        preprocessor.write_bytes(preprocessor.read_bytes() + b"tampered")
        return str(target)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", tampered_download)
    with pytest.raises(ArtifactError, match="preprocessor.joblib SHA-256"):
        PredictionService.load(
            preprocessing_uri=preprocessing_uri,
            model_uri=runs.first.model_uri,
            loader=MlflowPyfuncLoader(),
        )


@pytest.mark.parametrize(
    "model_uri",
    ["models:/synthetic-model", "models:/synthetic-model/latest", "models:/synthetic-model@champion"],
)
def test_m7_serving_rejects_non_versioned_model_references(synthetic_m7_runs, model_uri) -> None:
    with pytest.raises(ArtifactError, match="exact registered-model URI"):
        PredictionService.load(
            preprocessing_uri=f"runs:/{synthetic_m7_runs.first.run_id}/m6",
            model_uri=model_uri,
            loader=MlflowPyfuncLoader(),
        )
