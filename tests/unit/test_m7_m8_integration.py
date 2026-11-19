"""M7-to-M8 integration using synthetic M5 rows and real local pipeline code."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Event, Lock, local

import pytest

mlflow = pytest.importorskip("mlflow", reason="install the optional training extra")
pytest.importorskip("fastapi", reason="install the optional serve extra")

import f1_race_intelligence.modeling.model_package as model_package_module
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling.model_package import (
    InvalidPublishedPackage,
    MANIFEST_FILE,
    PUBLISHED_FILE,
    ModelPackageError,
    PackageAlreadyPublished,
    canonical_json_bytes,
    publish_model,
    read_model_package,
)
from f1_race_intelligence.modeling.training import TrainingResult, TrainingRunner
from f1_race_intelligence.serving import app as app_module
from f1_race_intelligence.serving.app import create_app
from f1_race_intelligence.serving.contract import ArtifactError
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService
from f1_race_intelligence.storage.backends.base import StorageError
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
from serving_support import M6Run, build_m6_run, to_records
from s3_fakes import FakeClientError, FakeS3Client


@dataclass(frozen=True)
class SyntheticM7Runs:
    m6: M6Run
    settings: AppSettings
    first: TrainingResult
    second: TrainingResult


def _synthetic_package_source(variant: str):
    fingerprint = f"sha256:{hashlib.sha256(variant.encode('utf-8')).hexdigest()}"
    m6_run_id = f"m6-{variant}"
    run_id = f"m7-{variant}"
    model_name = "race-package"
    version = "1"
    manifest = {
        "format": "f1-race-intelligence-model-package",
        "schema_version": 1,
        "package_id": f"{model_name}:{version}",
        "model": {
            "name": model_name,
            "version": version,
            "uri": f"models:/{model_name}/{version}",
            "m7_run_id": run_id,
            "input_signature": ["encoded"],
            "target": "next_lap_time",
        },
        "provenance": {
            "m6_run_id": m6_run_id,
            "input_fingerprint": fingerprint,
            "m6_source_commit": "b" * 40,
            "m6_pipeline_version": "2",
            "m7_source_commit": "a" * 40,
            "m7_source_dirty": False,
            "input_fingerprint_scheme": "sha256-v1",
            "data_version": f"m6:{m6_run_id}:{fingerprint}",
        },
    }
    artifacts = {
        "data_provenance.json": f"provenance-{variant}".encode(),
        "m6/feature_schema.json": f"schema-{variant}".encode(),
        "m6/modeling_report_synthetic.json": f"m6-report-{variant}".encode(),
        "m6/preprocessor.joblib": f"preprocessor-{variant}".encode(),
        "model/MLmodel": f"model-{variant}".encode(),
        "training_config.json": f"config-{variant}".encode(),
        "training_report.json": f"report-{variant}".encode(),
    }
    if variant in {"alpha", "beta"}:
        artifacts["data_provenance.json"] = b"shared-provenance"
        artifacts["m6/feature_schema.json"] = b"shared-feature-schema"
    return manifest, artifacts


class _WriteProbeMixin:
    def __init__(
        self,
        *args,
        write_barriers=None,
        marker_started=None,
        marker_release=None,
        **kwargs,
    ):
        self.conditional_write_successes = {}
        self.conditional_write_conflicts = {}
        self.unconditional_write_keys = []
        self._write_barriers = write_barriers or {}
        self._marker_started = marker_started
        self._marker_release = marker_release
        self._probe_lock = Lock()
        super().__init__(*args, **kwargs)

    def write_bytes_if_absent(self, key, data, *, content_type=None):
        self._wait_before_marker(key)
        created = super().write_bytes_if_absent(key, data, content_type=content_type)
        with self._probe_lock:
            counters = self.conditional_write_successes if created else self.conditional_write_conflicts
            counters[key] = counters.get(key, 0) + 1
        self._pause_after_key(key)
        return created

    def write_bytes(self, key, data, *, content_type=None):
        self._wait_before_marker(key)
        with self._probe_lock:
            self.unconditional_write_keys.append(key)
        result = super().write_bytes(key, data, content_type=content_type)
        self._pause_after_key(key)
        return result

    def _wait_before_marker(self, key):
        if key.endswith(f"/{PUBLISHED_FILE}") and self._marker_started is not None:
            self._marker_started.set()
            if not self._marker_release.wait(timeout=30):
                raise TimeoutError("test did not release the publication marker")

    def _pause_after_key(self, key):
        for suffix, barrier in self._write_barriers.items():
            if key.endswith(suffix):
                barrier.wait(timeout=30)
                return


class _WriteProbeLocalBackend(_WriteProbeMixin, LocalStorageBackend):
    pass


class _WriteProbeS3Backend(_WriteProbeMixin, S3StorageBackend):
    pass


def _assert_only_conditional_writes(backend) -> None:
    assert backend.unconditional_write_keys == []


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


def test_published_package_is_identical_in_local_and_simulated_s3(synthetic_m7_runs, tmp_path) -> None:
    runs = synthetic_m7_runs
    local = LocalStorageBackend(tmp_path / "local")
    s3 = S3StorageBackend("model-package-test", client=FakeS3Client())

    local_package = publish_model(runs.first.model_uri, local)
    s3_package = publish_model(runs.first.model_uri, s3)

    assert local_package.manifest == s3_package.manifest
    assert local_package.artifacts == s3_package.artifacts
    assert local_package.manifest_sha256 == s3_package.manifest_sha256
    assert local_package.manifest["model"]["m7_run_id"] == runs.first.run_id
    assert local_package.manifest["provenance"]["m6_run_id"] == runs.first.m6_run_id
    assert local_package.manifest["provenance"]["input_fingerprint"] == runs.first.input_fingerprint
    assert set(local_package.artifacts) == set(local_package.manifest["files"])
    assert local.read_bytes(f"{local_package.prefix}/{PUBLISHED_FILE}") == s3.read_bytes(
        f"{s3_package.prefix}/{PUBLISHED_FILE}"
    )


def test_identical_republication_returns_the_verified_existing_version(synthetic_m7_runs, tmp_path) -> None:
    backend = LocalStorageBackend(tmp_path)
    package = publish_model(synthetic_m7_runs.first.model_uri, backend)

    repeated = publish_model(synthetic_m7_runs.first.model_uri, backend)

    assert repeated.manifest == package.manifest
    assert repeated.artifacts == package.artifacts
    assert read_model_package(backend, package.prefix).manifest == package.manifest


def test_invalid_existing_marker_is_distinguished_from_an_unpublished_version(monkeypatch, tmp_path) -> None:
    backend = LocalStorageBackend(tmp_path)
    manifest, artifacts = _synthetic_package_source("invalid-marker")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    marker = b"not-a-manifest-digest\n"
    backend.write_bytes(f"{prefix}/{PUBLISHED_FILE}", marker)
    backend.write_bytes(f"{prefix}/{MANIFEST_FILE}", b"{}")

    with pytest.raises(InvalidPublishedPackage, match="PUBLISHED marker but is invalid"):
        publish_model("models:/race-package/1", backend)

    assert backend.read_bytes(f"{prefix}/{PUBLISHED_FILE}") == marker
    with pytest.raises(ModelPackageError, match="marker is malformed"):
        read_model_package(backend, prefix)

    dangling_prefix = "published-models/race-package/versions/2"
    backend.write_bytes(f"{dangling_prefix}/{PUBLISHED_FILE}", b"sha256:" + b"0" * 64 + b"\n")
    with pytest.raises(InvalidPublishedPackage, match="PUBLISHED marker but no manifest"):
        read_model_package(backend, dangling_prefix)


def test_different_republication_cannot_replace_the_existing_version(
    synthetic_m7_runs, tmp_path, monkeypatch
) -> None:
    backend = LocalStorageBackend(tmp_path)
    result = synthetic_m7_runs.first
    package = publish_model(result.model_uri, backend)
    different_manifest, different_artifacts = _synthetic_package_source("different-version-content")
    monkeypatch.setattr(
        model_package_module,
        "_collect_m7_artifacts",
        lambda *args: (different_manifest, different_artifacts),
    )

    with pytest.raises(PackageAlreadyPublished, match="different content"):
        publish_model(result.model_uri, backend)

    assert read_model_package(backend, package.prefix).manifest == package.manifest
    assert read_model_package(backend, package.prefix).artifacts == package.artifacts


@pytest.mark.parametrize("damage", ["altered", "missing", "extra"])
def test_published_package_rejects_incomplete_or_changed_object_sets(
    synthetic_m7_runs, tmp_path, damage
) -> None:
    backend = LocalStorageBackend(tmp_path)
    package = publish_model(synthetic_m7_runs.first.model_uri, backend)
    artifact = next(name for name in package.artifacts if name.startswith("model/"))
    artifact_path = tmp_path / package.prefix / Path(*artifact.split("/"))

    if damage == "altered":
        backend.write_bytes(f"{package.prefix}/{artifact}", artifact_path.read_bytes() + b"tampered")
        message = "integrity check failed"
    elif damage == "missing":
        artifact_path.unlink()
        message = "object set differs"
    else:
        backend.write_bytes(f"{package.prefix}/unexpected.bin", b"unexpected")
        message = "object set differs"

    with pytest.raises(ValueError, match=message):
        read_model_package(backend, package.prefix)


def test_published_package_rejects_noncanonical_manifest(synthetic_m7_runs, tmp_path) -> None:
    backend = LocalStorageBackend(tmp_path)
    package = publish_model(synthetic_m7_runs.first.model_uri, backend)
    manifest_key = f"{package.prefix}/{MANIFEST_FILE}"
    invalid_manifest = b'{ "format":"wrong","schema_version":1}'
    backend.write_bytes(manifest_key, invalid_manifest)
    backend.write_bytes(
        f"{package.prefix}/{PUBLISHED_FILE}",
        f"sha256:{hashlib.sha256(invalid_manifest).hexdigest()}\n".encode("ascii"),
    )

    with pytest.raises(ValueError, match="canonical JSON"):
        read_model_package(backend, package.prefix)


def test_published_package_is_unavailable_until_publication_marker_exists(synthetic_m7_runs, tmp_path) -> None:
    class MarkerFailureBackend(LocalStorageBackend):
        def write_bytes_if_absent(self, key, data, *, content_type=None):
            if key.endswith(f"/{PUBLISHED_FILE}"):
                raise OSError("simulated failure before publication")
            return super().write_bytes_if_absent(key, data, content_type=content_type)

    backend = MarkerFailureBackend(tmp_path)
    with pytest.raises(OSError, match="before publication"):
        publish_model(synthetic_m7_runs.first.model_uri, backend)

    prefix = f"published-models/{synthetic_m7_runs.first.model_name}/versions/{synthetic_m7_runs.first.model_version}"
    assert backend.has_prefix(prefix)
    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ValueError, match="not published"):
        read_model_package(backend, prefix)


@pytest.mark.parametrize("backend_kind", ["local", "s3"])
def test_concurrent_different_publishers_cannot_publish_a_mixed_package(
    monkeypatch, tmp_path, backend_kind
) -> None:
    client = FakeS3Client() if backend_kind == "s3" else None
    barriers = {
        "data_provenance.json": Barrier(2),
        "m6/feature_schema.json": Barrier(2),
        "m6/modeling_report_synthetic.json": Barrier(2),
    }
    if backend_kind == "local":
        backend = _WriteProbeLocalBackend(tmp_path, write_barriers=barriers)
    else:
        backend = _WriteProbeS3Backend(
            "concurrent-package-test",
            client=client,
            write_barriers=barriers,
        )
    thread_state = local()

    def collect(model_uri, registered_name, version):
        return _synthetic_package_source(thread_state.variant)

    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", collect)

    def attempt(variant):
        thread_state.variant = variant
        try:
            return variant, publish_model("models:/race-package/1", backend), None
        except Exception as exc:
            return variant, None, exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ("alpha", "beta")))

    winners = [(variant, package) for variant, package, error in outcomes if package is not None]
    losers = [(variant, error) for variant, package, error in outcomes if package is None]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0][1], ModelPackageError)

    winner, published = winners[0]
    verified = read_model_package(backend, published.prefix)
    expected_manifest, expected_artifacts = _synthetic_package_source(winner)
    assert verified.manifest["model"]["m7_run_id"] == expected_manifest["model"]["m7_run_id"]
    assert verified.artifacts == expected_artifacts
    package_keys = {
        f"{published.prefix}/{path}" for path in verified.manifest["files"]
    } | {f"{published.prefix}/{MANIFEST_FILE}", f"{published.prefix}/{PUBLISHED_FILE}"}
    assert set(backend.list(published.prefix)) == package_keys
    _assert_only_conditional_writes(backend)
    assert set(backend.conditional_write_successes) == package_keys
    assert all(count == 1 for count in backend.conditional_write_successes.values())

    conflict_path = f"{published.prefix}/m6/modeling_report_synthetic.json"
    assert backend.conditional_write_conflicts.get(conflict_path, 0) >= 1
    if backend_kind == "s3":
        assert client is not None
        physical_keys = {
            backend.uri(key).split("/", 3)[-1]: 1
            for key in package_keys
        }
        assert client.conditional_write_successes == physical_keys
        assert client.unconditional_write_keys == {}
        physical_conflict = backend.uri(conflict_path).split("/", 3)[-1]
        assert client.conditional_write_conflicts[physical_conflict] >= 1


def test_write_protection_assertion_detects_unconditional_mutation(monkeypatch, tmp_path) -> None:
    backend = _WriteProbeLocalBackend(tmp_path)
    manifest, artifacts = _synthetic_package_source("mutated-writer")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))

    def unguarded_write(backend, prefix, objects, *, content_type=None):
        for path, content in sorted(objects.items()):
            backend.write_bytes(f"{prefix}/{path}", content, content_type=content_type)

    monkeypatch.setattr(model_package_module, "_write_or_match", unguarded_write)
    publish_model("models:/race-package/1", backend)

    with pytest.raises(AssertionError):
        _assert_only_conditional_writes(backend)


@pytest.mark.parametrize("backend_kind", ["local", "s3"])
def test_reader_rejects_package_while_publisher_is_waiting_to_publish_marker(
    monkeypatch, tmp_path, backend_kind
) -> None:
    client = FakeS3Client() if backend_kind == "s3" else None
    marker_started, marker_release = Event(), Event()
    if backend_kind == "local":
        backend = _WriteProbeLocalBackend(
            tmp_path,
            marker_started=marker_started,
            marker_release=marker_release,
        )
    else:
        backend = _WriteProbeS3Backend(
            "concurrent-reader-test",
            client=client,
            marker_started=marker_started,
            marker_release=marker_release,
        )
    manifest, artifacts = _synthetic_package_source("reader")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"

    with ThreadPoolExecutor(max_workers=2) as executor:
        publish_future = executor.submit(publish_model, "models:/race-package/1", backend)
        try:
            assert marker_started.wait(timeout=30)
            assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
            read_future = executor.submit(read_model_package, backend, prefix)
            with pytest.raises(ModelPackageError, match="not published"):
                read_future.result(timeout=30)
        finally:
            marker_release.set()
        published = publish_future.result(timeout=30)

    assert read_model_package(backend, prefix).artifacts == artifacts
    assert published.manifest["model"]["m7_run_id"] == manifest["model"]["m7_run_id"]


def test_s3_conditional_409_without_visible_object_is_not_reported_as_published(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("conditional-409-test", client=client)
    manifest, artifacts = _synthetic_package_source("conflict")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    marker_key = backend.uri(f"{prefix}/{PUBLISHED_FILE}").split("/", 3)[-1]
    client.fail_put_for_key[marker_key] = FakeClientError("ConditionalRequestConflict", 409, "PutObject")

    with pytest.raises(ModelPackageError, match="conflicted but no marker"):
        publish_model("models:/race-package/1", backend)

    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)


def test_s3_409_on_artifact_retries_when_preexisting_bytes_match(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("artifact-409-retry-test", client=client)
    manifest, artifacts = _synthetic_package_source("identical-after-409")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    first_key = f"{prefix}/data_provenance.json"
    physical_key = backend.uri(first_key).split("/", 3)[-1]
    client.fail_put_for_key[physical_key] = FakeClientError("ConditionalRequestConflict", 409, "PutObject")

    with pytest.raises(ModelPackageError, match="disappeared during publication"):
        publish_model("models:/race-package/1", backend)
    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)

    del client.fail_put_for_key[physical_key]
    assert backend.write_bytes(first_key, artifacts["data_provenance.json"])
    client.fail_put_for_key[physical_key] = FakeClientError(
        "ConditionalRequestConflict", 409, "PutObject"
    )
    published = publish_model("models:/race-package/1", backend)
    assert published.artifacts == artifacts
    assert backend.read_bytes(first_key) == artifacts["data_provenance.json"]
    assert client.conditional_write_attempts[physical_key] == 2
    assert client.unconditional_write_keys[physical_key] == 1


def test_s3_conditional_412_preserves_the_existing_object(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("conditional-412-test", client=client)
    manifest, artifacts = _synthetic_package_source("precondition")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    object_key = f"{prefix}/data_provenance.json"
    backend.write_bytes(object_key, b"existing-object")
    physical_key = backend.uri(object_key).split("/", 3)[-1]
    client.fail_put_for_key[physical_key] = FakeClientError("PreconditionFailed", 412, "PutObject")

    with pytest.raises(ModelPackageError, match="refusing to overwrite"):
        publish_model("models:/race-package/1", backend)

    assert backend.read_bytes(object_key) == b"existing-object"
    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)


def test_s3_partial_network_failure_and_mixed_retry_never_publish(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("partial-package-test", client=client)
    manifest, artifacts = _synthetic_package_source("original")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    client.fail_put_on_call[2] = FakeClientError("ConnectionReset", 0, "PutObject")

    with pytest.raises(StorageError, match="S3 write failed"):
        publish_model("models:/race-package/1", backend)

    first_key = f"{prefix}/data_provenance.json"
    assert backend.read_bytes(first_key) == artifacts["data_provenance.json"]
    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)

    competing_key = f"{prefix}/m6/feature_schema.json"
    assert backend.write_bytes_if_absent(competing_key, b"other-publisher")
    client.fail_put_on_call.clear()
    with pytest.raises(ModelPackageError, match="refusing to overwrite"):
        publish_model("models:/race-package/1", backend)

    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)


def test_s3_partial_failure_can_resume_only_with_identical_objects(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("identical-partial-retry-test", client=client)
    manifest, artifacts = _synthetic_package_source("identical-partial")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    client.fail_put_on_call[3] = FakeClientError("ConnectionReset", 0, "PutObject")

    with pytest.raises(StorageError, match="S3 write failed"):
        publish_model("models:/race-package/1", backend)

    assert backend.read_bytes(f"{prefix}/data_provenance.json") == artifacts["data_provenance.json"]
    assert backend.read_bytes(f"{prefix}/m6/feature_schema.json") == artifacts["m6/feature_schema.json"]
    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)

    client.fail_put_on_call.clear()
    published = publish_model("models:/race-package/1", backend)
    assert published.artifacts == artifacts
    assert read_model_package(backend, prefix).artifacts == artifacts
    first_key = backend.uri(f"{prefix}/data_provenance.json").split("/", 3)[-1]
    assert client.conditional_write_successes[first_key] == 1


def test_s3_marker_failure_is_unpublished_and_identical_retry_can_finish(monkeypatch) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("marker-failure-test", client=client)
    manifest, artifacts = _synthetic_package_source("marker")
    monkeypatch.setattr(model_package_module, "_collect_m7_artifacts", lambda *args: (manifest, artifacts))
    prefix = "published-models/race-package/versions/1"
    marker_key = backend.uri(f"{prefix}/{PUBLISHED_FILE}").split("/", 3)[-1]
    client.fail_put_for_key[marker_key] = FakeClientError("ServiceUnavailable", 503, "PutObject")

    with pytest.raises(StorageError, match="S3 write failed"):
        publish_model("models:/race-package/1", backend)

    assert not backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    with pytest.raises(ModelPackageError, match="not published"):
        read_model_package(backend, prefix)

    del client.fail_put_for_key[marker_key]
    published = publish_model("models:/race-package/1", backend)
    assert read_model_package(backend, prefix).artifacts == artifacts
    assert published.manifest["model"]["m7_run_id"] == manifest["model"]["m7_run_id"]


@pytest.mark.parametrize("retry_kind", ["identical", "different"])
def test_ambiguous_marker_timeout_is_idempotent_only_for_identical_content(
    monkeypatch, retry_kind
) -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("ambiguous-marker-test", client=client)
    manifest, artifacts = _synthetic_package_source("committed-before-timeout")
    current_source = [manifest, artifacts]
    monkeypatch.setattr(
        model_package_module,
        "_collect_m7_artifacts",
        lambda *args: tuple(current_source),
    )
    prefix = "published-models/race-package/versions/1"
    marker_key = backend.uri(f"{prefix}/{PUBLISHED_FILE}").split("/", 3)[-1]
    client.fail_after_put_for_key[marker_key] = FakeClientError("ReadTimeout", 0, "PutObject")

    with pytest.raises(StorageError, match="S3 write failed"):
        publish_model("models:/race-package/1", backend)

    assert backend.exists(f"{prefix}/{PUBLISHED_FILE}")
    verified = read_model_package(backend, prefix)
    counts_before_retry = dict(client.conditional_write_successes)
    if retry_kind == "different":
        current_source[:] = _synthetic_package_source("different-after-timeout")
        with pytest.raises(PackageAlreadyPublished, match="different content"):
            publish_model("models:/race-package/1", backend)
        assert read_model_package(backend, prefix).artifacts == artifacts
    else:
        repeated = publish_model("models:/race-package/1", backend)
        assert repeated.manifest == verified.manifest
        assert repeated.artifacts == artifacts
    assert client.conditional_write_successes == counts_before_retry


@pytest.mark.parametrize("backend_kind", ["local", "s3"])
def test_published_package_loads_in_m8_and_predicts_without_refitting(
    synthetic_m7_runs, tmp_path, monkeypatch, backend_kind
) -> None:
    runs = synthetic_m7_runs
    if backend_kind == "local":
        backend = LocalStorageBackend(tmp_path)
    else:
        backend = S3StorageBackend("m8-package-test", client=FakeS3Client())
    package = publish_model(runs.first.model_uri, backend)
    service = PredictionService.load_published_package(
        backend=backend,
        prefix=package.prefix,
        loader=MlflowPyfuncLoader(),
    )
    rows = runs.m6.split_file("test", "dataset.parquet").head(2)
    records = to_records(rows, runs.m6.numeric + runs.m6.categorical)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("published-package inference must not fit M6 preprocessing")

    monkeypatch.setattr(ColumnTransformer, "fit", forbidden_fit)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden_fit)
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/model-info").json()
        response = client.post("/predict", json={"records": records})

    assert info["model"]["uri"] == runs.first.model_uri
    assert info["model"]["run_id"] == runs.first.run_id
    assert info["preprocessing"]["uri"] == package.package_uri
    assert info["provenance"]["m6_run_id"] == runs.first.m6_run_id
    assert response.status_code == 200
    assert response.json()["target"] == "next_lap_time"
    assert all(80.0 <= item["value"] <= 110.0 for item in response.json()["predictions"])


@pytest.mark.parametrize("artifact", ["preprocessor.joblib", "feature_schema.json"])
def test_m8_rechecks_m6_artifact_hashes_from_m7_provenance(
    synthetic_m7_runs, tmp_path, artifact
) -> None:
    backend = LocalStorageBackend(tmp_path)
    package = publish_model(synthetic_m7_runs.first.model_uri, backend)
    provenance = json.loads(package.artifacts["data_provenance.json"])
    hashes = [
        item
        for item in provenance["verified_artifact_hashes"]
        if item["artifact"] == artifact and item["split"] is None
    ]
    assert len(hashes) == 1
    hashes[0]["sha256"] = "0" * 64
    provenance_bytes = json.dumps(provenance, sort_keys=True).encode("utf-8")
    manifest = json.loads(canonical_json_bytes(package.manifest))
    manifest["files"]["data_provenance.json"] = {
        "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "size_bytes": len(provenance_bytes),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    backend.write_bytes(f"{package.prefix}/data_provenance.json", provenance_bytes)
    backend.write_bytes(f"{package.prefix}/{MANIFEST_FILE}", manifest_bytes)
    backend.write_bytes(
        f"{package.prefix}/{PUBLISHED_FILE}",
        f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}\n".encode("ascii"),
    )

    with pytest.raises(ArtifactError, match=f"{artifact} SHA-256 does not match M7 run provenance"):
        PredictionService.load_published_package(
            backend=backend,
            prefix=package.prefix,
            loader=MlflowPyfuncLoader(),
        )


def test_published_package_rejects_model_artifact_from_another_m7_run(
    synthetic_m7_runs, tmp_path
) -> None:
    runs = synthetic_m7_runs
    backend = LocalStorageBackend(tmp_path)
    first = publish_model(runs.first.model_uri, backend)
    second = publish_model(runs.second.model_uri, backend, package_root="other-packages")
    first_models = {name: content for name, content in first.artifacts.items() if name.startswith("model/")}
    second_models = {name: content for name, content in second.artifacts.items() if name.startswith("model/")}
    assert set(first_models) == set(second_models)

    for name in first_models:
        backend.write_bytes(f"{first.prefix}/{name}", second_models[name])
    manifest = dict(first.manifest)
    files = dict(manifest["files"])
    for name, content in second_models.items():
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    manifest["files"] = files
    manifest_bytes = canonical_json_bytes(manifest)
    backend.write_bytes(f"{first.prefix}/{MANIFEST_FILE}", manifest_bytes)
    backend.write_bytes(
        f"{first.prefix}/{PUBLISHED_FILE}",
        f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}\n".encode("ascii"),
    )

    with pytest.raises(ArtifactError, match="belongs to M7 run"):
        PredictionService.load_published_package(
            backend=backend,
            prefix=first.prefix,
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
