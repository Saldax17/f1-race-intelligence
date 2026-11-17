"""Immutable publication and verification of registered M7 model artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from f1_race_intelligence.features.selection import TARGET_COLUMN
from f1_race_intelligence.storage.backends.base import JSON_CONTENT_TYPE, StorageBackend, join_key

PACKAGE_FORMAT = "f1-race-intelligence-model-package"
PACKAGE_SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
PUBLISHED_FILE = "PUBLISHED"
DEFAULT_PACKAGE_ROOT = "published-models"
_MODEL_URI = re.compile(r"^models:/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/([1-9][0-9]*)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INPUT_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


class ModelPackageError(ValueError):
    """A model package is incomplete, inconsistent, or fails integrity checks."""


class PackageAlreadyPublished(ModelPackageError):
    """The immutable package version has already been published."""


class InvalidPublishedPackage(ModelPackageError):
    """A publication marker exists, but the package cannot be verified."""


@dataclass(frozen=True)
class VerifiedModelPackage:
    """All package bytes after manifest, membership, and hash verification."""

    prefix: str
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, bytes]
    manifest_sha256: str
    package_uri: str


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a manifest deterministically, independently of platform."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def publish_model(
    model_uri: str,
    backend: StorageBackend,
    *,
    package_root: str = DEFAULT_PACKAGE_ROOT,
) -> VerifiedModelPackage:
    """Publish an exact registered M7 version and its paired M6 artifacts.

    MLflow is imported lazily. The source run is read only; this function never
    fits, registers, or changes a model.
    """
    registered_name, version = _parse_model_uri(model_uri)
    root = _validate_logical_path(package_root, allow_empty=False)
    prefix = join_key(root, registered_name, "versions", version)

    manifest, artifact_bytes = _collect_m7_artifacts(model_uri, registered_name, version)
    file_records = {
        path: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        for path, content in sorted(artifact_bytes.items())
    }
    manifest = {**manifest, "files": file_records}
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    marker_key = join_key(prefix, PUBLISHED_FILE)
    if backend.exists(marker_key):
        return _resolve_existing_package(backend, prefix, manifest_bytes, registered_name, version)

    _write_or_match(backend, prefix, artifact_bytes)
    _write_or_match(backend, prefix, {MANIFEST_FILE: manifest_bytes}, content_type=JSON_CONTENT_TYPE)

    expected_keys = {join_key(prefix, name) for name in artifact_bytes}
    expected_keys.add(join_key(prefix, MANIFEST_FILE))
    actual_keys = set(backend.list(prefix))
    if actual_keys != expected_keys:
        extras = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        raise ModelPackageError(f"package contents differ before publication (missing={missing}, extra={extras})")

    for path, expected in file_records.items():
        content = backend.read_bytes(join_key(prefix, path))
        if len(content) != expected["size_bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ModelPackageError(f"artifact failed post-write verification: {path}")
    if backend.read_bytes(join_key(prefix, MANIFEST_FILE)) != manifest_bytes:
        raise ModelPackageError("manifest changed during publication")

    marker = f"sha256:{manifest_hash}\n".encode("ascii")
    if not backend.write_bytes_if_absent(marker_key, marker):
        if not backend.exists(marker_key):
            raise ModelPackageError(
                "publication marker conditional write conflicted but no marker is visible"
            )
        return _resolve_existing_package(backend, prefix, manifest_bytes, registered_name, version)

    return read_model_package(backend, prefix)


def _resolve_existing_package(
    backend: StorageBackend,
    prefix: str,
    requested_manifest: bytes,
    registered_name: str,
    version: str,
) -> VerifiedModelPackage:
    try:
        existing = read_model_package(backend, prefix)
    except ModelPackageError as exc:
        raise InvalidPublishedPackage(
            f"model package {registered_name!r} version {version} has a PUBLISHED marker but is invalid: {exc}"
        ) from exc
    if canonical_json_bytes(existing.manifest) != requested_manifest:
        raise PackageAlreadyPublished(
            f"model package {registered_name!r} version {version} is already published with different content"
        )
    return existing


def read_model_package(backend: StorageBackend, prefix: str) -> VerifiedModelPackage:
    """Read only a fully published package and verify its exact object set."""
    prefix = _validate_logical_path(prefix, allow_empty=False)
    marker_key = join_key(prefix, PUBLISHED_FILE)
    manifest_key = join_key(prefix, MANIFEST_FILE)
    try:
        marker = backend.read_bytes(marker_key)
    except FileNotFoundError as exc:
        raise ModelPackageError(f"model package {backend.uri(prefix)!r} is not published") from exc
    try:
        manifest_bytes = backend.read_bytes(manifest_key)
    except FileNotFoundError as exc:
        raise InvalidPublishedPackage(
            f"model package {backend.uri(prefix)!r} has a PUBLISHED marker but no manifest"
        ) from exc

    marker_match = re.fullmatch(rb"sha256:([0-9a-f]{64})\n", marker)
    if marker_match is None:
        raise ModelPackageError("package publication marker is malformed")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if marker_match.group(1).decode("ascii") != manifest_hash:
        raise ModelPackageError("package manifest SHA-256 does not match its publication marker")

    manifest = _parse_canonical_manifest(manifest_bytes)
    file_records = manifest.get("files")
    _validate_manifest(manifest)
    assert isinstance(file_records, dict)

    expected_keys = {join_key(prefix, path) for path in file_records}
    expected_keys.update({manifest_key, marker_key})
    actual_keys = set(backend.list(prefix))
    if actual_keys != expected_keys:
        extras = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        raise ModelPackageError(f"package object set differs from its manifest (missing={missing}, extra={extras})")

    artifacts: Dict[str, bytes] = {}
    for path, record in file_records.items():
        content = backend.read_bytes(join_key(prefix, path))
        actual_hash = hashlib.sha256(content).hexdigest()
        if len(content) != record["size_bytes"] or actual_hash != record["sha256"]:
            raise ModelPackageError(f"artifact integrity check failed: {path}")
        artifacts[path] = content

    return VerifiedModelPackage(prefix, manifest, artifacts, manifest_hash, backend.uri(prefix))


def _collect_m7_artifacts(
    model_uri: str,
    registered_name: str,
    version: str,
) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "Publishing a registered M7 model requires the training extra: "
            '`python -m pip install -e ".[training]"`'
        ) from exc

    client = MlflowClient()
    model_version = client.get_model_version(registered_name, version)
    if str(model_version.version) != version or model_version.status != "READY":
        raise ModelPackageError(f"registered model {model_uri!r} is not the expected READY version")
    run_id = model_version.run_id
    if not isinstance(run_id, str) or not run_id:
        raise ModelPackageError(f"registered model {model_uri!r} has no source M7 run")

    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        if getattr(getattr(loaded_model, "metadata", None), "run_id", None) != run_id:
            raise ModelPackageError(f"registered model {model_uri!r} is not bound to M7 run {run_id!r}")
        signature = loaded_model.metadata.get_input_schema()
        input_names = list(signature.input_names()) if signature is not None else []
        if not input_names or not all(isinstance(name, str) and name for name in input_names):
            raise ModelPackageError(f"registered model {model_uri!r} has no named input signature")
        model_directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model"))
        m6_directory = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/m6"))
        provenance_path = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/data_provenance.json")
        )
        report_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/training_report.json"))
        config_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/training_config.json"))
    except ModelPackageError:
        raise
    except Exception as exc:
        raise ModelPackageError(f"could not read registered M7 model {model_uri!r}: {exc}") from exc

    provenance = _read_json(provenance_path, "data_provenance.json")
    report = _read_json(report_path, "training_report.json")
    config = _read_json(config_path, "training_config.json")
    m6_report_paths = sorted(m6_directory.glob("modeling_report_*.json"))
    if len(m6_report_paths) != 1:
        raise ModelPackageError("M7 run must contain exactly one M6 modeling report")
    m6_report_path = m6_report_paths[0]
    m6_report = _read_json(m6_report_path, m6_report_path.name)
    preprocessor = (m6_directory / "preprocessor.joblib").read_bytes()
    schema_bytes = (m6_directory / "feature_schema.json").read_bytes()
    schema = _decode_json(schema_bytes, "feature_schema.json")
    _validate_m7_traceability(
        model_uri=model_uri,
        registered_name=registered_name,
        version=version,
        run_id=run_id,
        input_names=input_names,
        provenance=provenance,
        report=report,
        config=config,
        m6_report=m6_report,
        m6_report_bytes=m6_report_path.read_bytes(),
        preprocessor=preprocessor,
        schema_bytes=schema_bytes,
        schema=schema,
    )

    artifacts: Dict[str, bytes] = {}
    for path, content in _read_directory(model_directory).items():
        artifacts[f"model/{path}"] = content
    artifacts.update(
        {
            "m6/preprocessor.joblib": preprocessor,
            "m6/feature_schema.json": schema_bytes,
            f"m6/{m6_report_path.name}": m6_report_path.read_bytes(),
            "data_provenance.json": provenance_path.read_bytes(),
            "training_report.json": report_path.read_bytes(),
            "training_config.json": config_path.read_bytes(),
        }
    )

    manifest = {
        "format": PACKAGE_FORMAT,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": f"{registered_name}:{version}",
        "model": {
            "name": registered_name,
            "version": version,
            "uri": model_uri,
            "m7_run_id": run_id,
            "input_signature": input_names,
            "target": schema["target"],
        },
        "provenance": {
            "m6_run_id": provenance["m6_run_id"],
            "input_fingerprint": provenance["input_fingerprint"],
            "m6_source_commit": provenance.get("m6_source_commit"),
            "m6_pipeline_version": provenance.get("m6_pipeline_version"),
            "m7_source_commit": provenance.get("m7_source_commit"),
            "m7_source_dirty": provenance.get("m7_source_dirty"),
            "input_fingerprint_scheme": provenance.get("input_fingerprint_scheme"),
            "data_version": provenance.get("data_version"),
        },
    }
    return manifest, artifacts


def _validate_m7_traceability(
    *,
    model_uri: str,
    registered_name: str,
    version: str,
    run_id: str,
    input_names: list,
    provenance: Dict[str, Any],
    report: Dict[str, Any],
    config: Dict[str, Any],
    m6_report: Dict[str, Any],
    m6_report_bytes: bytes,
    preprocessor: bytes,
    schema_bytes: bytes,
    schema: Dict[str, Any],
) -> None:
    if (
        report.get("run_id") != run_id
        or report.get("source_commit") != provenance.get("m7_source_commit")
        or report.get("source_dirty") != provenance.get("m7_source_dirty")
    ):
        raise ModelPackageError("M7 training report and provenance identify different runs or commits")
    registered = report.get("registered_model")
    if not isinstance(registered, dict) or (
        registered.get("name") != registered_name
        or str(registered.get("version")) != version
        or registered.get("uri") != model_uri
    ):
        raise ModelPackageError("M7 training report does not match the selected registered model version")
    m6_run_id = provenance.get("m6_run_id")
    fingerprint = provenance.get("input_fingerprint")
    if not isinstance(m6_run_id, str) or not m6_run_id or not isinstance(fingerprint, str):
        raise ModelPackageError("M7 provenance has no M6 run ID or input_fingerprint")
    if not _INPUT_FINGERPRINT.fullmatch(fingerprint):
        raise ModelPackageError("M7 provenance has an invalid input_fingerprint")
    if report.get("m6_run_id") != m6_run_id or report.get("input_fingerprint") != fingerprint:
        raise ModelPackageError("M7 report and provenance identify different M6 inputs")
    m6_source = m6_report.get("source")
    if (
        m6_report.get("run_id") != m6_run_id
        or not isinstance(m6_source, dict)
        or m6_source.get("input_fingerprint") != fingerprint
        or m6_report.get("source_commit") != provenance.get("m6_source_commit")
        or m6_report.get("pipeline_version") != provenance.get("m6_pipeline_version")
    ):
        raise ModelPackageError("M7 provenance and bundled M6 report identify different M6 inputs")
    expected_m6_report_hash = provenance.get("m6_report_sha256")
    if not isinstance(expected_m6_report_hash, str) or hashlib.sha256(m6_report_bytes).hexdigest() != expected_m6_report_hash:
        raise ModelPackageError("bundled M6 report SHA-256 does not match M7 provenance")
    if not isinstance(schema.get("encoded_feature_names"), list) or schema["encoded_feature_names"] != input_names:
        raise ModelPackageError("MLflow input signature does not match M6 encoded_feature_names order")
    if schema.get("target") != report.get("target") or schema.get("target") != TARGET_COLUMN:
        raise ModelPackageError("M7 target does not match the M6 feature schema")
    if not isinstance(report.get("features"), list) or report["features"] != input_names:
        raise ModelPackageError("M7 training report features do not match the model signature")
    if report.get("config") != config:
        raise ModelPackageError("M7 training report and configuration artifact do not match")
    _verify_m6_hash(provenance, "preprocessor.joblib", preprocessor)
    _verify_m6_hash(provenance, "feature_schema.json", schema_bytes)


def _verify_m6_hash(provenance: Mapping[str, Any], artifact: str, content: bytes) -> None:
    entries = provenance.get("verified_artifact_hashes")
    matching = [
        item
        for item in entries or []
        if isinstance(item, dict) and item.get("artifact") == artifact and item.get("split") is None
    ]
    if len(matching) != 1 or matching[0].get("sha256") != hashlib.sha256(content).hexdigest():
        raise ModelPackageError(f"M7 provenance does not verify its M6 {artifact} artifact")


def _write_or_match(
    backend: StorageBackend,
    prefix: str,
    artifacts: Mapping[str, bytes],
    *,
    content_type: Optional[str] = None,
) -> None:
    for path, content in sorted(artifacts.items()):
        _validate_logical_path(path, allow_empty=False)
        key = join_key(prefix, path)
        if backend.write_bytes_if_absent(key, content, content_type=content_type):
            continue
        try:
            existing = backend.read_bytes(key)
        except FileNotFoundError as exc:
            raise ModelPackageError(f"immutable package object disappeared during publication: {path}") from exc
        if existing != content:
            raise ModelPackageError(f"refusing to overwrite a different package object: {path}")


def _parse_canonical_manifest(data: bytes) -> Dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ModelPackageError(f"manifest repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"package manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != data:
        raise ModelPackageError("package manifest is not in canonical JSON form")
    return payload


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    if set(manifest) != {"format", "schema_version", "package_id", "model", "provenance", "files"}:
        raise ModelPackageError("manifest has unknown or missing top-level fields")
    if manifest.get("format") != PACKAGE_FORMAT or manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ModelPackageError("unsupported model package format or schema version")
    model = manifest.get("model")
    provenance = manifest.get("provenance")
    files = manifest.get("files")
    if not isinstance(model, dict) or not isinstance(provenance, dict) or not isinstance(files, dict):
        raise ModelPackageError("package manifest is missing model, provenance, or file metadata")
    if set(model) != {"name", "version", "uri", "m7_run_id", "input_signature", "target"}:
        raise ModelPackageError("manifest has unknown or missing model fields")
    if set(provenance) != {
        "m6_run_id",
        "input_fingerprint",
        "m6_source_commit",
        "m6_pipeline_version",
        "m7_source_commit",
        "m7_source_dirty",
        "input_fingerprint_scheme",
        "data_version",
    }:
        raise ModelPackageError("manifest has unknown or missing provenance fields")
    expected_uri = f"models:/{model.get('name')}/{model.get('version')}"
    if (
        not isinstance(model.get("name"), str)
        or not isinstance(model.get("version"), str)
        or not _MODEL_URI.fullmatch(expected_uri)
        or model.get("uri") != expected_uri
        or manifest.get("package_id") != f"{model.get('name')}:{model.get('version')}"
        or not isinstance(model.get("m7_run_id"), str)
        or not model["m7_run_id"]
    ):
        raise ModelPackageError("manifest has an invalid registered-model identity")
    if (
        not isinstance(provenance.get("m6_run_id"), str)
        or not provenance["m6_run_id"]
        or not isinstance(provenance.get("input_fingerprint"), str)
        or not _INPUT_FINGERPRINT.fullmatch(provenance["input_fingerprint"])
        or (
            provenance.get("m7_source_dirty") is not None
            and not isinstance(provenance.get("m7_source_dirty"), bool)
        )
        or provenance.get("data_version")
        != f"m6:{provenance.get('m6_run_id')}:{provenance.get('input_fingerprint')}"
        or model.get("target") != TARGET_COLUMN
    ):
        raise ModelPackageError("manifest has invalid M6 traceability")
    signature = model.get("input_signature")
    if (
        not isinstance(signature, list)
        or not signature
        or not all(isinstance(name, str) and name for name in signature)
        or len(signature) != len(set(signature))
        or not isinstance(model.get("target"), str)
        or not model["target"]
    ):
        raise ModelPackageError("manifest has an invalid model input contract")

    required = {
        "model/MLmodel",
        "m6/preprocessor.joblib",
        "m6/feature_schema.json",
        "data_provenance.json",
        "training_report.json",
        "training_config.json",
    }
    if not required.issubset(files):
        raise ModelPackageError("manifest omits required model or M6 artifacts")
    for path, record in files.items():
        _validate_logical_path(path, allow_empty=False)
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
            raise ModelPackageError(f"manifest has invalid file metadata for {path!r}")
        if (
            not isinstance(record["sha256"], str)
            or not _SHA256.fullmatch(record["sha256"])
            or not isinstance(record["size_bytes"], int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] < 0
        ):
            raise ModelPackageError(f"manifest has invalid size or SHA-256 for {path!r}")


def _parse_model_uri(model_uri: str) -> Tuple[str, str]:
    match = _MODEL_URI.fullmatch(model_uri)
    if match is None:
        raise ModelPackageError("publishing requires models:/<registered_name>/<exact_numeric_version>")
    return match.group(1), match.group(2)


def _validate_logical_path(path: str, *, allow_empty: bool) -> str:
    if not isinstance(path, str) or "\\" in path or path.startswith("/") or ":" in path:
        raise ModelPackageError(f"invalid package path: {path!r}")
    if not path and allow_empty:
        return path
    parts = path.split("/")
    if not path or any(part in {"", ".", ".."} for part in parts):
        raise ModelPackageError(f"invalid package path: {path!r}")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts):
        raise ModelPackageError(f"invalid package path: {path!r}")
    return path


def _read_directory(directory: Path) -> Dict[str, bytes]:
    if not directory.is_dir():
        raise ModelPackageError(f"MLflow model artifact is not a directory: {directory}")
    result: Dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ModelPackageError(f"MLflow model artifact contains a symlink: {path.name}")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            _validate_logical_path(relative, allow_empty=False)
            result[relative] = path.read_bytes()
    if "MLmodel" not in result:
        raise ModelPackageError("MLflow model artifact has no MLmodel file")
    return result


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        return _decode_json(path.read_bytes(), label)
    except OSError as exc:
        raise ModelPackageError(f"M7 artifact {label!r} is unavailable: {exc}") from exc


def _decode_json(content: bytes, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"M7 artifact {label!r} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelPackageError(f"M7 artifact {label!r} must contain a JSON object")
    return payload
