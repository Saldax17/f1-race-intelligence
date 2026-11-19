"""Read-only verification of report-declared M6 artifact hashes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from f1_race_intelligence.config.settings import StorageSettings
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend


def verify_report(location: str, *, storage: Optional[StorageSettings] = None) -> Dict[str, Any]:
    """Verify a v1 or v2 report and every artifact it declares, without rewriting it."""
    errors: List[str] = []
    try:
        payload = json.loads(_read_location(location, storage=storage).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "report": location, "errors": [f"cannot read report: {exc}"], "artifacts_checked": 0}

    if not isinstance(payload, dict):
        return {"ok": False, "report": location, "errors": ["report root must be a JSON object"], "artifacts_checked": 0}
    version = payload.get("schema_version", 1)
    if version not in (1, 2):
        errors.append(f"unsupported schema_version: {version!r}")
    if version == 2:
        if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
            errors.append("schema v2 report has no run_id")
        if not _is_sha256(payload.get("config_fingerprint")):
            errors.append("schema v2 report has an invalid config_fingerprint")
        if not _is_sha256(payload.get("dependency_lock_sha256")):
            errors.append("schema v2 report has no valid dependency_lock_sha256")
        verdicts = {"VALID", "VALID_WITH_WARNINGS", "INCOMPLETE", "INCONCLUSIVE", "INVALID", "ABORTED"}
        if payload.get("verdict") not in verdicts:
            errors.append("schema v2 report has an invalid global verdict")
        source = payload.get("source")
        fingerprint = source.get("input_fingerprint") if isinstance(source, dict) else None
        if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or not _is_sha256(fingerprint[7:]):
            errors.append("schema v2 report has no valid input_fingerprint")

    artifact_files = payload.get("artifact_files")
    if not isinstance(artifact_files, list):
        errors.append("report has no artifact_files list")
        artifact_files = []
    elif version == 2 and payload.get("verdict") in {"VALID", "VALID_WITH_WARNINGS"} and len(artifact_files) != 8:
        errors.append("a valid schema v2 M6 report must declare all eight artifacts")

    checked = 0
    for index, artifact in enumerate(artifact_files):
        if not isinstance(artifact, dict):
            errors.append(f"artifact_files[{index}] is not an object")
            continue
        path, expected, size = artifact.get("path"), artifact.get("sha256"), artifact.get("size_bytes")
        if not isinstance(path, str) or not _is_sha256(expected):
            errors.append(f"artifact_files[{index}] has invalid path or sha256")
            continue
        try:
            data = _read_location(path, storage=storage)
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: cannot read artifact: {exc}")
            continue
        checked += 1
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            errors.append(f"{path}: sha256 mismatch (expected {expected}, got {actual})")
        if not isinstance(size, int) or size != len(data):
            errors.append(f"{path}: size_bytes mismatch (expected {size!r}, got {len(data)})")

    return {
        "ok": not errors,
        "report": location,
        "schema_version": version,
        "run_id": payload.get("run_id"),
        "recorded_verdict": payload.get("verdict"),
        "artifacts_checked": checked,
        "errors": errors,
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_location(location: str, *, storage: Optional[StorageSettings]) -> bytes:
    if location.startswith("s3://"):
        bucket, separator, key = location[5:].partition("/")
        if not separator or not bucket or not key:
            raise ValueError(f"invalid S3 object URI: {location!r}")
        backend = S3StorageBackend(
            bucket,
            client=None,
            region_name=storage.region if storage else None,
            endpoint_url=storage.endpoint_url if storage else None,
        )
        return backend.read_bytes(key)
    return Path(location).read_bytes()
