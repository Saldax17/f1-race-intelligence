"""Shared M7 contracts for the traditional race model: provenance, manifests,
generalization diagnosis, structural interpretability, Markdown reporting,
serving schema validation, and optional lazy MLflow logging.

Every function here is pure Python / NumPy / stdlib and never imports MLflow at
module import time, keeping the M1-M6 and AWS Fargate runtime image isolated
from training-only extras (``requirements/training-lock.txt``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from f1_race_intelligence.modeling.traceability import (
    config_fingerprint,
    dependency_lock_sha256,
    source_revision,
)
from f1_race_intelligence.storage.backends.base import join_key

SCHEMA_VERSION = 2
PIPELINE_NAME = "race_model"
PIPELINE_VERSION = "1.0.0"
MODEL_TYPE = "hierarchical_state_space"
ESTIMATOR_NAME = "f1_race_intelligence.race_model.RaceModelRunner"
INPUT_FINGERPRINT_SCHEME = "m7-race-model-input-fingerprint/v2"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PUBLISHED_MARKER = "PUBLISHED"
MODEL_FILENAME = "model.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLISHED_MARKER_RE = re.compile(rb"^sha256:([0-9a-f]{64})\n$")


class ModelPackageIntegrityError(ValueError):
    """A saved race-model version package is incomplete, unpublished, or tampered."""

# Generalization diagnosis thresholds (mirroring modeling/dl.py discipline;
# evaluated strictly on train and validation, never on the sealed test split).
OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO = 1.25
OVERFITTING_VAL_TO_TRAIN_RPS_RATIO = 1.35
CALIBRATION_MAX_Z2 = 1.50
_MIN_DENOM = 1e-9

STRUCTURAL_INTERPRETABILITY_NOTE = (
    "Structural parameters (AR(1) persistence phi, clean-lap noise sigma_eps, tyre "
    "degradation slopes, Kalman variance components and event hazards) are fitted "
    "strictly on training races (2023-24). Initial team/driver prior variances "
    "(p0_gamma, p0_delta) are fixed priors rather than data-identified parameters; "
    "variance shares reflect the innovation and race-level decomposition across "
    "fitted Level-2 components."
)

MLFLOW_INSTALL_HINT = (
    "python -m pip install -r requirements/training-lock.txt && "
    "python -m pip install -e '.[training]' --no-deps"
)

ALLOWED_FORECAST_REQUEST_KEYS = frozenset(
    {"session_key", "lap", "source", "model_version", "total_laps", "year", "mode"}
)


def _normalize_for_json(value: Any, *, allow_non_finite_sentinel: bool = False) -> Any:
    """Recursively normalize Python/NumPy/Pandas values for deterministic JSON."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            if allow_non_finite_sentinel:
                return None
            raise ValueError("canonical JSON does not permit NaN float values")
        if math.isinf(f):
            if allow_non_finite_sentinel:
                return "+Infinity" if f > 0 else "-Infinity"
            raise ValueError("canonical JSON does not permit infinite float values")
        return 0.0 if f == 0.0 else f
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value
        return ts.isoformat()
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo is not None else value
        return dt.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_for_json(v, allow_non_finite_sentinel=allow_non_finite_sentinel)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, np.ndarray):
        return [_normalize_for_json(x, allow_non_finite_sentinel=allow_non_finite_sentinel) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(x, allow_non_finite_sentinel=allow_non_finite_sentinel) for x in value]
    if isinstance(value, (str,)):
        return value
    return str(value)


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic RFC 8259 UTF-8 JSON (sorted keys, compact separators, no NaN)."""
    normalized = _normalize_for_json(payload, allow_non_finite_sentinel=False)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


_canonical_json_bytes = canonical_json_bytes


def _canonical_hash_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON for hashing raw observations (normalizes NaN/NaT to null)."""
    normalized = _normalize_for_json(payload, allow_non_finite_sentinel=True)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


# -- 1. Provenance & Input Fingerprinting -------------------------------------


def _hash_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over all concrete key/value pairs of record dicts."""
    encoded = sorted(_canonical_hash_json_bytes(dict(r)) for r in records)
    hasher = hashlib.sha256()
    hasher.update(f"records:{len(encoded)}\n".encode("ascii"))
    for item in encoded:
        hasher.update(f"{len(item)}:".encode("ascii"))
        hasher.update(item)
        hasher.update(b"\n")
    return hasher.hexdigest()


def _hash_dataframe(frame: Optional[pd.DataFrame], sort_keys: Sequence[str]) -> str:
    """Deterministic SHA-256 over every concrete cell value of a DataFrame."""
    if frame is None or frame.empty:
        cols = sorted(str(c) for c in frame.columns) if frame is not None else []
        return sha256_bytes(_canonical_hash_json_bytes({"columns": cols, "rows": 0}))

    cols = sorted(frame.columns, key=str)
    ordered = frame[cols]
    key_cols = [c for c in sort_keys if c in ordered.columns]
    if key_cols:
        try:
            ordered = ordered.sort_values(by=key_cols, kind="mergesort", na_position="first")
        except TypeError:
            row_keys = [
                _canonical_hash_json_bytes({str(c): row[idx] for idx, c in enumerate(cols)})
                for row in ordered.itertuples(index=False, name=None)
            ]
            order_idx = sorted(range(len(row_keys)), key=row_keys.__getitem__)
            ordered = ordered.iloc[order_idx]

    hasher = hashlib.sha256()
    hasher.update(f"dataframe:{len(ordered)}\n".encode("ascii"))
    for col in cols:
        series = ordered[col]
        col_name = str(col).encode("utf-8")
        hasher.update(f"col:{len(col_name)}:".encode("ascii"))
        hasher.update(col_name)
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            ts = pd.to_datetime(series, utc=True)
            arr = ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy(dtype="<i8")
            hasher.update(b":dt64ns:")
            hasher.update(arr.tobytes())
        elif pd.api.types.is_bool_dtype(series.dtype):
            arr = series.to_numpy(dtype="?")
            hasher.update(b":bool:")
            hasher.update(arr.tobytes())
        elif pd.api.types.is_integer_dtype(series.dtype):
            arr = series.to_numpy(dtype="<i8")
            hasher.update(b":int64:")
            hasher.update(arr.tobytes())
        elif pd.api.types.is_float_dtype(series.dtype):
            arr = series.to_numpy(dtype="<f8", copy=True)
            nan_mask = np.isnan(arr)
            arr[nan_mask] = 0.0
            arr[arr == 0.0] = 0.0
            hasher.update(b":float64:")
            hasher.update(nan_mask.astype("?").tobytes())
            hasher.update(arr.tobytes())
        else:
            payload = _canonical_hash_json_bytes(series.tolist())
            hasher.update(f":json:{len(payload)}:".encode("ascii"))
            hasher.update(payload)
        hasher.update(b"\n")
    return hasher.hexdigest()


def race_input_fingerprint(train_races: Sequence[Any]) -> Dict[str, Any]:
    """Deterministic SHA-256 fingerprint over every concrete observation of training races.

    Hashes all race metadata (``ref``), ``drivers``, ``laps``, ``stints``, ``pit``,
    ``race_control``, ``intervals``, ``grid``, and ``result`` strictly for
    ``train_races``. Modifying any training observation changes the fingerprint,
    while replacing validation or test races leaves it completely invariant.
    """
    entries: List[Dict[str, Any]] = []
    for race in sorted(train_races, key=lambda r: (r.ref.year, r.ref.session_key)):
        driver_records = [
            {"_driver_key": int(driver_num), **dict(info)}
            for driver_num, info in sorted(race.drivers.items(), key=lambda kv: int(kv[0]))
        ]
        section_hashes = {
            "drivers": _hash_records(driver_records),
            "laps": _hash_dataframe(race.laps, ("driver_number", "lap_number", "date_start")),
            "stints": _hash_records(race.stints),
            "pit": _hash_records(race.pit),
            "race_control": _hash_records(race.race_control),
            "intervals": _hash_dataframe(race.intervals, ("driver_number", "date", "interval", "gap_to_leader")),
            "grid": _hash_records(race.grid),
            "result": _hash_records(race.result),
        }
        race_digest_payload = {
            "ref": _normalize_for_json(race.ref.to_dict(), allow_non_finite_sentinel=True),
            "counts": {
                "drivers": len(race.drivers),
                "laps": int(len(race.laps)) if race.laps is not None else 0,
                "stints": len(race.stints),
                "pit": len(race.pit),
                "race_control": len(race.race_control),
                "intervals": int(len(race.intervals)) if race.intervals is not None else 0,
                "grid": len(race.grid),
                "result": len(race.result),
            },
            "sections_sha256": section_hashes,
        }
        entries.append(
            {
                "year": int(race.ref.year),
                "session_key": int(race.ref.session_key),
                "tag": race.tag,
                "sha256": sha256_bytes(canonical_json_bytes(race_digest_payload)),
            }
        )
    doc = {"scheme": INPUT_FINGERPRINT_SCHEME, "inputs": entries}
    return {
        "input_fingerprint": f"sha256:{sha256_bytes(canonical_json_bytes(doc))}",
        "input_fingerprint_scheme": INPUT_FINGERPRINT_SCHEME,
        "train_races_count": len(entries),
        "train_inputs": entries,
    }


def build_provenance(
    settings_dump: Mapping[str, Any],
    train_races: Sequence[Any],
    *,
    validation_races_count: int,
    test_sealed: bool,
    random_seed: int,
) -> Dict[str, Any]:
    """Build M6/M7-aligned provenance metadata for a race-model version."""
    commit, dirty = source_revision()
    lock_sha = dependency_lock_sha256()
    cfg_safe = {
        k: v
        for k, v in settings_dump.items()
        if k not in {"raw_path", "output_path", "input_storage", "output_storage"}
    }
    fp = race_input_fingerprint(train_races)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "model_type": MODEL_TYPE,
        "estimator": ESTIMATOR_NAME,
        "source_commit": commit,
        "source_dirty": dirty,
        "dependency_lock_sha256": lock_sha,
        "config_fingerprint": config_fingerprint(cfg_safe),
        "input_fingerprint": fp["input_fingerprint"],
        "input_fingerprint_scheme": fp["input_fingerprint_scheme"],
        "train_races_count": fp["train_races_count"],
        "validation_races_count": int(validation_races_count),
        "test_sealed": bool(test_sealed),
        "random_seed": int(random_seed),
    }


# -- 2. Immutable Package Manifest & Verification -----------------------------


def _parse_json_no_duplicates(data: bytes, label: str) -> Dict[str, Any]:
    """Parse a UTF-8 JSON object while rejecting duplicate keys."""

    def _reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"{label} repeats JSON key {k!r}")
            out[k] = v
        return out

    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return parsed


def _validate_relative_filename(rel_path: Any) -> Optional[str]:
    if not isinstance(rel_path, str) or not rel_path or "\\" in rel_path or rel_path.startswith("/") or ":" in rel_path:
        return None
    parts = rel_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return rel_path


def _write_or_match_bytes(
    backend: Any,
    key: str,
    data: bytes,
    *,
    content_type: str,
    label: str,
) -> None:
    """Conditionally create ``key`` or verify existing bytes match identically."""
    if backend.write_bytes_if_absent(key, data, content_type=content_type):
        return
    try:
        existing = backend.read_bytes(key)
    except FileNotFoundError as exc:
        raise ModelPackageIntegrityError(f"immutable package object disappeared during publication: {label}") from exc
    if existing != data:
        raise FileExistsError(f"conflicting {label} already exists at {backend.uri(key)}")


def _build_version_manifest(
    *,
    version: str,
    created_at: str,
    model_format: str,
    model_bytes: bytes,
    provenance: Mapping[str, Any],
    parent_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], bytes, str, bytes, str]:
    model_sha = sha256_bytes(model_bytes)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": model_format,
        "version": version,
        "created_at": created_at,
        "model_type": MODEL_TYPE,
        "estimator": ESTIMATOR_NAME,
        "parent_version": parent_version,
        "provenance": dict(provenance),
        "files": [
            {
                "path": MODEL_FILENAME,
                "sha256": model_sha,
                "size_bytes": len(model_bytes),
            }
        ],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    marker_bytes = f"sha256:{manifest_sha}\n".encode("ascii")
    return manifest, manifest_bytes, manifest_sha, marker_bytes, model_sha


def _resolve_existing_published_version(
    backend: Any,
    version_prefix: str,
    *,
    version: str,
    model_format: str,
    model_bytes: bytes,
    manifest_bytes: bytes,
    manifest_sha: str,
    model_sha: str,
) -> Dict[str, Any]:
    """Verify an already-published version and allow only byte-identical idempotent retries."""
    verification = verify_version_package(
        backend,
        version_prefix,
        version,
        expected_format=model_format,
    )
    if not verification["ok"]:
        raise ModelPackageIntegrityError(
            f"model version {version} has a {PUBLISHED_MARKER} marker but failed integrity verification: "
            f"{'; '.join(verification['errors'])}"
        )
    manifest_key = join_key(version_prefix, MANIFEST_FILENAME)
    model_key = join_key(version_prefix, MODEL_FILENAME)
    published_key = join_key(version_prefix, PUBLISHED_MARKER)
    existing_manifest = backend.read_bytes(manifest_key)
    existing_model = backend.read_bytes(model_key)
    if existing_manifest != manifest_bytes or existing_model != model_bytes:
        raise FileExistsError(
            f"model version {version} is already published with different content at {backend.uri(version_prefix)}"
        )
    return {
        "model_uri": backend.uri(model_key),
        "manifest_uri": backend.uri(manifest_key),
        "published_uri": backend.uri(published_key),
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "model_size_bytes": len(model_bytes),
    }


def publish_version_package(
    backend: Any,
    version_prefix: str,
    *,
    version: str,
    created_at: str,
    model_format: str,
    model_bytes: bytes,
    provenance: Mapping[str, Any],
    parent_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically publish ``model.json``, ``manifest.json``, and ``PUBLISHED``.

    Mirrors ``modeling/model_package.py``:
    - Conditional writes only (``write_bytes_if_absent``) with write-or-match
      recovery when an earlier attempt stopped before writing ``PUBLISHED``.
    - Pre-commit verification of exact object membership and read-back hashes
      before creating ``PUBLISHED`` last.
    - Idempotent retry when an already-published package matches byte-for-byte;
      conflicting content raises ``FileExistsError``; corrupt published packages
      raise ``ModelPackageIntegrityError``.
    """
    _, manifest_bytes, manifest_sha, marker_bytes, model_sha = _build_version_manifest(
        version=version,
        created_at=created_at,
        model_format=model_format,
        model_bytes=model_bytes,
        provenance=provenance,
        parent_version=parent_version,
    )
    model_key = join_key(version_prefix, MODEL_FILENAME)
    manifest_key = join_key(version_prefix, MANIFEST_FILENAME)
    published_key = join_key(version_prefix, PUBLISHED_MARKER)

    if backend.exists(published_key):
        return _resolve_existing_published_version(
            backend,
            version_prefix,
            version=version,
            model_format=model_format,
            model_bytes=model_bytes,
            manifest_bytes=manifest_bytes,
            manifest_sha=manifest_sha,
            model_sha=model_sha,
        )

    _write_or_match_bytes(
        backend,
        model_key,
        model_bytes,
        content_type="application/json",
        label=MODEL_FILENAME,
    )
    _write_or_match_bytes(
        backend,
        manifest_key,
        manifest_bytes,
        content_type="application/json",
        label=MANIFEST_FILENAME,
    )

    expected_pre_commit = {model_key, manifest_key}
    actual_keys = set(backend.list(version_prefix))
    if published_key in actual_keys:
        return _resolve_existing_published_version(
            backend,
            version_prefix,
            version=version,
            model_format=model_format,
            model_bytes=model_bytes,
            manifest_bytes=manifest_bytes,
            manifest_sha=manifest_sha,
            model_sha=model_sha,
        )
    if actual_keys != expected_pre_commit:
        extras = sorted(actual_keys - expected_pre_commit)
        missing = sorted(expected_pre_commit - actual_keys)
        raise ModelPackageIntegrityError(
            f"package contents differ before publication (missing={missing}, extra={extras})"
        )

    read_model = backend.read_bytes(model_key)
    if len(read_model) != len(model_bytes) or sha256_bytes(read_model) != model_sha:
        raise ModelPackageIntegrityError(f"{MODEL_FILENAME} failed post-write verification before publication")
    if backend.read_bytes(manifest_key) != manifest_bytes:
        raise ModelPackageIntegrityError(f"{MANIFEST_FILENAME} changed during publication")

    if not backend.write_bytes_if_absent(published_key, marker_bytes, content_type="text/plain; charset=utf-8"):
        if not backend.exists(published_key):
            raise ModelPackageIntegrityError(
                f"{PUBLISHED_MARKER} conditional write conflicted but no marker is visible"
            )
        return _resolve_existing_published_version(
            backend,
            version_prefix,
            version=version,
            model_format=model_format,
            model_bytes=model_bytes,
            manifest_bytes=manifest_bytes,
            manifest_sha=manifest_sha,
            model_sha=model_sha,
        )

    verified = verify_version_package(
        backend,
        version_prefix,
        version,
        expected_format=model_format,
    )
    if not verified["ok"]:
        raise ModelPackageIntegrityError(
            f"published model version {version} failed post-commit verification: {'; '.join(verified['errors'])}"
        )

    return {
        "model_uri": backend.uri(model_key),
        "manifest_uri": backend.uri(manifest_key),
        "published_uri": backend.uri(published_key),
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "model_size_bytes": len(model_bytes),
    }


def write_version_manifest(
    backend: Any,
    version_prefix: str,
    *,
    version: str,
    created_at: str,
    model_format: str,
    model_bytes: bytes,
    provenance: Mapping[str, Any],
    parent_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Backwards-compatible wrapper that publishes ``model.json``, ``manifest.json``, and ``PUBLISHED``."""
    return publish_version_package(
        backend,
        version_prefix,
        version=version,
        created_at=created_at,
        model_format=model_format,
        model_bytes=model_bytes,
        provenance=provenance,
        parent_version=parent_version,
    )


def verify_version_package(
    backend: Any,
    version_prefix: str,
    version: str,
    *,
    expected_format: str = "f1-race-model/1",
) -> Dict[str, Any]:
    """Read-only SHA-256 verification of a saved race-model version package.

    Checks ``PUBLISHED`` marker -> canonical ``manifest.json`` -> exact object
    membership -> ``model.json`` bytes, hashes, and schema consistency,
    mirroring ``modeling/verification.py`` and ``modeling/model_package.py``.
    """
    errors: List[str] = []
    manifest_key = join_key(version_prefix, MANIFEST_FILENAME)
    published_key = join_key(version_prefix, PUBLISHED_MARKER)
    files_checked = 0
    manifest_sha: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None

    if not backend.exists(published_key):
        errors.append(f"missing {PUBLISHED_MARKER} marker at {backend.uri(published_key)}")
        declared_manifest_sha = None
    else:
        raw_marker = backend.read_bytes(published_key)
        marker_match = _PUBLISHED_MARKER_RE.fullmatch(raw_marker)
        if marker_match is None:
            errors.append(f"invalid {PUBLISHED_MARKER} marker format: {raw_marker!r}")
            declared_manifest_sha = None
        else:
            declared_manifest_sha = marker_match.group(1).decode("ascii")

    if not backend.exists(manifest_key):
        errors.append(f"missing {MANIFEST_FILENAME} at {backend.uri(manifest_key)}")
    else:
        manifest_bytes = backend.read_bytes(manifest_key)
        manifest_sha = sha256_bytes(manifest_bytes)
        if declared_manifest_sha is not None and manifest_sha != declared_manifest_sha:
            errors.append(
                f"{MANIFEST_FILENAME}: sha256 mismatch against {PUBLISHED_MARKER} "
                f"(expected {declared_manifest_sha}, got {manifest_sha})"
            )
        try:
            parsed = _parse_json_no_duplicates(manifest_bytes, MANIFEST_FILENAME)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"cannot parse {MANIFEST_FILENAME}: {exc}")
        else:
            try:
                is_canonical = canonical_json_bytes(parsed) == manifest_bytes
            except ValueError:
                is_canonical = False
            if not is_canonical:
                errors.append(f"{MANIFEST_FILENAME} is not in canonical JSON form")
            manifest = parsed
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                errors.append(
                    f"{MANIFEST_FILENAME} schema_version mismatch: expected {MANIFEST_SCHEMA_VERSION!r}, "
                    f"got {manifest.get('schema_version')!r}"
                )
            if manifest.get("version") != version:
                errors.append(
                    f"{MANIFEST_FILENAME} version mismatch: expected {version!r}, got {manifest.get('version')!r}"
                )
            if manifest.get("format") != expected_format:
                errors.append(
                    f"{MANIFEST_FILENAME} format mismatch: expected {expected_format!r}, got {manifest.get('format')!r}"
                )
            declared_files = manifest.get("files")
            if not isinstance(declared_files, list) or not declared_files:
                errors.append(f"{MANIFEST_FILENAME} has no declared files")
            else:
                seen_paths: List[str] = []
                expected_keys = {manifest_key, published_key}
                for idx, entry in enumerate(declared_files):
                    if not isinstance(entry, dict):
                        errors.append(f"{MANIFEST_FILENAME} files[{idx}] must be a JSON object")
                        continue
                    rel_path = _validate_relative_filename(entry.get("path"))
                    if rel_path is None:
                        errors.append(f"{MANIFEST_FILENAME} files[{idx}] has invalid path {entry.get('path')!r}")
                        continue
                    if rel_path in seen_paths:
                        errors.append(f"{MANIFEST_FILENAME} repeats declared file {rel_path!r}")
                        continue
                    seen_paths.append(rel_path)
                    expected_sha = entry.get("sha256")
                    expected_size = entry.get("size_bytes")
                    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
                        errors.append(f"{rel_path}: invalid declared sha256 {expected_sha!r}")
                    if not _is_int(expected_size) or int(expected_size) < 0:
                        errors.append(f"{rel_path}: invalid declared size_bytes {expected_size!r}")
                    file_key = join_key(version_prefix, rel_path)
                    expected_keys.add(file_key)
                    if not backend.exists(file_key):
                        errors.append(f"{rel_path}: missing artifact at {backend.uri(file_key)}")
                        continue
                    content = backend.read_bytes(file_key)
                    files_checked += 1
                    actual_sha = sha256_bytes(content)
                    if actual_sha != expected_sha:
                        errors.append(
                            f"{rel_path}: sha256 mismatch (expected {expected_sha}, got {actual_sha})"
                        )
                    if expected_size != len(content):
                        errors.append(
                            f"{rel_path}: size_bytes mismatch (expected {expected_size}, got {len(content)})"
                        )
                    if rel_path == MODEL_FILENAME:
                        try:
                            model_doc = _parse_json_no_duplicates(content, rel_path)
                        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                            errors.append(f"{rel_path}: invalid JSON ({exc})")
                        else:
                            if model_doc.get("format") != expected_format:
                                errors.append(f"{rel_path}: unexpected model format {model_doc.get('format')!r}")
                            if model_doc.get("version") != version:
                                errors.append(f"{rel_path}: version mismatch {model_doc.get('version')!r}")
                if MODEL_FILENAME not in seen_paths:
                    errors.append(f"{MANIFEST_FILENAME} omits required {MODEL_FILENAME}")
                actual_keys = set(backend.list(version_prefix))
                extra_keys = sorted(actual_keys - expected_keys)
                if extra_keys:
                    errors.append(f"package contains unexpected extra object(s): {extra_keys}")

    return {
        "ok": not errors,
        "version": version,
        "manifest_uri": backend.uri(manifest_key),
        "manifest_sha256": manifest_sha,
        "files_checked": files_checked,
        "provenance": manifest.get("provenance") if isinstance(manifest, dict) else None,
        "errors": errors,
    }


# -- 3. Generalization Diagnosis (strictly train & validation) ----------------


def diagnose_race_model_generalization(
    *,
    prior_check: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    noise_scale_rps: Optional[Mapping[str, Any]] = None,
    baseline_key: str = "persist",
) -> Dict[str, Any]:
    """Transparent over/underfitting verdict strictly on ``train`` and ``validation``.

    Mirrors ``modeling.dl.diagnose_generalization`` with the exact same 4-state
    vocabulary (``balanced``, ``overfitting_risk``, ``underfitting``,
    ``insufficient_metrics``) and strict ``evaluated_splits = ["train", "validation"]``
    anti-snooping guarantee (sealed ``test`` never participates).
    """
    train_pc = prior_check.get("train") or {}
    val_pc = prior_check.get("validation") or {}
    train_rmse = float(train_pc["rmse"]) if _is_finite_real(train_pc.get("rmse")) else None
    val_rmse = float(val_pc["rmse"]) if _is_finite_real(val_pc.get("rmse")) else None
    train_z2 = float(train_pc["z2"]) if _is_finite_real(train_pc.get("z2")) else None
    val_z2 = float(val_pc["z2"]) if _is_finite_real(val_pc.get("z2")) else None

    overall = (validation_summary.get("by_bucket") or {}).get("all") or {}
    val_rps = float(overall["model_rps"]) if _is_finite_real(overall.get("model_rps")) else None
    base_rps = float(overall.get(f"{baseline_key}_rps")) if _is_finite_real(overall.get(f"{baseline_key}_rps")) else None
    trans_rps = float(overall["transition_rps"]) if _is_finite_real(overall.get("transition_rps")) else None

    # Best training calibration RPS (from noise_scale selection on training races), if recorded.
    train_cal_rps: Optional[float] = None
    if isinstance(noise_scale_rps, Mapping):
        overall_cal = noise_scale_rps.get("overall")
        if isinstance(overall_cal, Mapping):
            finite_vals = [float(v) for v in overall_cal.values() if _is_finite_real(v)]
            if finite_vals:
                train_cal_rps = min(finite_vals)

    val_to_train_prior_rmse_ratio = (
        val_rmse / train_rmse
        if train_rmse is not None and val_rmse is not None and train_rmse > _MIN_DENOM
        else None
    )
    val_to_train_rps_ratio = (
        val_rps / train_cal_rps
        if train_cal_rps is not None and val_rps is not None and train_cal_rps > _MIN_DENOM
        else None
    )
    val_vs_baseline_rps_delta = (
        val_rps - base_rps if val_rps is not None and base_rps is not None else None
    )
    val_vs_transition_rps_delta = (
        val_rps - trans_rps if val_rps is not None and trans_rps is not None else None
    )

    if val_rps is None or (train_rmse is None and val_rmse is None):
        status = "insufficient_metrics"
        explanation = "Cannot diagnose generalization: validation RPS or prior-check RMSE is missing."
    elif base_rps is not None and val_rps >= base_rps:
        status = "underfitting"
        explanation = (
            f"Underfitting: validation RPS {val_rps:.4f} does not beat the {baseline_key!r} "
            f"baseline ({base_rps:.4f})."
        )
    elif (
        (val_to_train_prior_rmse_ratio is not None and val_to_train_prior_rmse_ratio >= OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO)
        or (val_z2 is not None and val_z2 >= CALIBRATION_MAX_Z2)
        or (val_to_train_rps_ratio is not None and val_to_train_rps_ratio >= OVERFITTING_VAL_TO_TRAIN_RPS_RATIO)
    ):
        status = "overfitting_risk"
        reasons = []
        if val_to_train_prior_rmse_ratio is not None and val_to_train_prior_rmse_ratio >= OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:
            reasons.append(
                f"validation prior RMSE ratio {val_to_train_prior_rmse_ratio:.2f}x >= "
                f"{OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:.2f}x"
            )
        if val_z2 is not None and val_z2 >= CALIBRATION_MAX_Z2:
            reasons.append(f"validation calibration z² {val_z2:.2f} >= {CALIBRATION_MAX_Z2:.2f}")
        if val_to_train_rps_ratio is not None and val_to_train_rps_ratio >= OVERFITTING_VAL_TO_TRAIN_RPS_RATIO:
            reasons.append(
                f"validation/train calibration RPS ratio {val_to_train_rps_ratio:.2f}x >= "
                f"{OVERFITTING_VAL_TO_TRAIN_RPS_RATIO:.2f}x"
            )
        explanation = "Overfitting risk: " + "; ".join(reasons) + "."
    else:
        status = "balanced"
        parts = [f"validation RPS {val_rps:.4f}"]
        if val_vs_baseline_rps_delta is not None:
            parts.append(f"beats {baseline_key!r} by {-val_vs_baseline_rps_delta:.4f}")
        if val_to_train_prior_rmse_ratio is not None:
            parts.append(f"prior RMSE ratio {val_to_train_prior_rmse_ratio:.2f}x (< {OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:.2f}x)")
        if val_z2 is not None:
            parts.append(f"calibration z² {val_z2:.2f} (< {CALIBRATION_MAX_Z2:.2f})")
        explanation = "Balanced: " + ", ".join(parts) + "."

    return {
        "status": status,
        "model_key": "race_model",
        "baseline_key": baseline_key,
        "train_prior_rmse": train_rmse,
        "validation_prior_rmse": val_rmse,
        "val_to_train_prior_rmse_ratio": val_to_train_prior_rmse_ratio,
        "train_prior_z2": train_z2,
        "validation_prior_z2": val_z2,
        "train_calibration_rps": train_cal_rps,
        "validation_rps": val_rps,
        "baseline_validation_rps": base_rps,
        "transition_validation_rps": trans_rps,
        "val_vs_baseline_rps_delta": val_vs_baseline_rps_delta,
        "val_vs_transition_rps_delta": val_vs_transition_rps_delta,
        "thresholds": {
            "overfitting_val_to_train_prior_rmse_ratio": OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO,
            "overfitting_val_to_train_rps_ratio": OVERFITTING_VAL_TO_TRAIN_RPS_RATIO,
            "calibration_max_z2": CALIBRATION_MAX_Z2,
        },
        "explanation": explanation,
        "evaluated_splits": ["train", "validation"],
    }


# -- 4. Structural Interpretability -------------------------------------------


def extract_interpretability(artifact: Any) -> Dict[str, Any]:
    """Structured interpretability summary of the fitted hierarchical state-space model.

    Mirrors ``training_report.json["interpretability"]`` in the DL pipeline:
    instead of post-hoc PFI on a black-box MLP, the traditional model exposes
    its training-fitted structural parameters (AR(1) persistence, compound
    degradation curves, Kalman variance decomposition across car/driver/circuit/
    race-day noise, and event hazards).
    """
    l1 = artifact.level1 or {}
    phi = float(l1.get("phi", 0.0))
    sigma_eps = float(l1.get("sigma_eps", 0.0))
    ar1_mult = math.sqrt((1.0 + phi) / max(1e-6, 1.0 - phi)) if abs(phi) < 1.0 else float("nan")

    tyres_raw = l1.get("tyres") or {}
    offsets = tyres_raw.get("offset") or {}
    slopes = tyres_raw.get("slope") or {}
    compounds = sorted(set(offsets) | set(slopes))
    tyre_table = [
        {
            "compound": str(c),
            "offset_pct": float(offsets.get(c, 0.0)),
            "degradation_slope_pct_per_lap": float(slopes.get(c, 0.0)),
        }
        for c in compounds
    ]
    tyre_table.sort(key=lambda row: -row["degradation_slope_pct_per_lap"])

    season_params = (artifact.season or {}).get("state", {}).get("params") or {}
    var_components = {
        "team_in_season_q_gamma": float(season_params.get("q_gamma", 0.0)) ** 2,
        "driver_in_season_q_delta": float(season_params.get("q_delta", 0.0)) ** 2,
        "team_circuit_fit_s_c": float(season_params.get("s_c", 0.0)) ** 2,
        "race_day_transient_s_w": float(season_params.get("s_w", 0.0)) ** 2,
    }
    total_var = sum(var_components.values())
    variance_decomposition = [
        {
            "component": name,
            "sd_pct": math.sqrt(val),
            "variance": val,
            "variance_share": (val / total_var) if total_var > 0 else 0.0,
        }
        for name, val in sorted(var_components.items(), key=lambda kv: -kv[1])
    ]

    events = artifact.events or {}
    hazards_raw = events.get("hazards") if isinstance(events.get("hazards"), Mapping) else {}
    global_hazards = {k: float(v) for k, v in hazards_raw.items() if _is_finite_real(v)}
    for src_key, dst_key in (
        ("sc_hazard", "SC"),
        ("vsc_hazard", "VSC"),
        ("red_hazard", "RED"),
        ("dnf_hazard", "DNF"),
    ):
        if dst_key not in global_hazards and _is_finite_real(events.get(src_key)):
            global_hazards[dst_key] = float(events[src_key])

    pit_green = events.get("pit_green") if _is_finite_real(events.get("pit_green")) else events.get("pit_loss")
    pit_sc = events.get("pit_sc")
    if not _is_finite_real(pit_sc) and _is_finite_real(pit_green) and _is_finite_real(events.get("sc_pit_factor")):
        pit_sc = float(pit_green) * float(events["sc_pit_factor"])
    pit_vsc = events.get("pit_vsc")
    if not _is_finite_real(pit_vsc) and _is_finite_real(pit_green) and _is_finite_real(events.get("vsc_pit_factor")):
        pit_vsc = float(pit_green) * float(events["vsc_pit_factor"])

    sc_by_loc = (events.get("sc_by_location") or {}) if isinstance(events.get("sc_by_location"), Mapping) else {}
    top_sc_circuits = sorted(
        [{"location": k, "sc_per_lap": float(v)} for k, v in sc_by_loc.items() if _is_finite_real(v)],
        key=lambda r: -r["sc_per_lap"],
    )[:5]

    fc = artifact.forecast or {}
    return {
        "status": "computed",
        "method": "structural_parameter_decomposition",
        "fit_split": "train",
        "level1_pace": {
            "phi_ar1": phi,
            "sigma_eps_pct": sigma_eps,
            "ar1_long_run_sd_multiplier": ar1_mult,
            "tyre_compounds": tyre_table,
        },
        "level2_season_variance": {
            "total_in_race_innovation_variance": total_var,
            "components": variance_decomposition,
            "winter_team_reset_sd_pct": float(season_params.get("q_gamma_season", 0.0)),
            "qualifying_weight_lambda": float(season_params.get("lam", 0.0)),
            "qualifying_noise_sd_pct": float(season_params.get("sigma_q", 0.0)),
        },
        "events": {
            "global_hazards": global_hazards,
            "pit_loss_green_s": float(pit_green) if _is_finite_real(pit_green) else None,
            "pit_loss_sc_s": float(pit_sc) if _is_finite_real(pit_sc) else None,
            "pit_loss_vsc_s": float(pit_vsc) if _is_finite_real(pit_vsc) else None,
            "top_sc_hazard_circuits": top_sc_circuits,
        },
        "phase_noise_multipliers": {
            "default_noise_scale": fc.get("noise_scale"),
            "by_fraction": fc.get("noise_scale_by_fraction"),
        },
        "overtaking_friction": {
            "pass_cost_seconds": fc.get("pass_cost_seconds", 0.0),
            "pooled_pass_rate": events.get("pass_rate"),
            "wet_noise_multiplier": events.get("wet_noise_multiplier", 1.0),
        },
        "phase_transition_shrinkage": fc.get("transition_shrinkage_by_fraction"),
        "note": STRUCTURAL_INTERPRETABILITY_NOTE,
    }


# -- 5. Academic Markdown Evaluation Report -----------------------------------


def render_evaluation_markdown(
    *,
    report: Mapping[str, Any],
    artifact: Any,
    promoted: bool,
    verification: Optional[Mapping[str, Any]] = None,
    mlflow_info: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render an academic Markdown evaluation report mirroring ``mlp_comparison_<id>.md``."""
    version = str(report.get("version", artifact.version))
    split = str(report.get("split", "validation"))
    prov = report.get("provenance") or getattr(artifact, "provenance", None) or {}
    by_bucket = report.get("by_bucket") or {}
    calib = report.get("calibration") or {}
    diag = report.get("generalization_diagnosis") or {}
    interp = report.get("interpretability") or {}

    def fmt(val: Any, digits: int = 4) -> str:
        return f"{float(val):.{digits}f}" if _is_finite_real(val) else "n/a"

    def fmt_ci(ci: Any) -> str:
        if not isinstance(ci, Mapping) or not _is_finite_real(ci.get("mean")):
            return "n/a"
        return f"{float(ci['mean']):+.4f} [{float(ci['lo']):+.4f}, {float(ci['hi']):+.4f}]"

    lines: List[str] = [
        f"# M7 Race Model Evaluation Report (`{version}` — `{split}`)",
        "",
        "## 1. Provenance & Experimental Protocol",
        "",
        f"- **Model Version:** `{version}`",
        f"- **Model Type:** `{prov.get('model_type', MODEL_TYPE)}` (`{prov.get('estimator', ESTIMATOR_NAME)}`)",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Evaluated Split:** `{split}` (`{report.get('races', 0)}` races, `{report.get('rows', 0)}` race-lap checkpoints)",
        f"- **Source Commit:** `{prov.get('source_commit') or 'unavailable'}` (dirty: `{prov.get('source_dirty')}`)",
        f"- **Train Input Fingerprint:** `{prov.get('input_fingerprint') or 'unavailable'}`",
        f"- **Config Fingerprint:** `{prov.get('config_fingerprint') or 'unavailable'}`",
        f"- **Dependency Lock SHA-256:** `{prov.get('dependency_lock_sha256') or 'unavailable'}`",
        f"- **Test Set Sealed:** `{prov.get('test_sealed')}`",
        f"- **Promotion Gate:** beat `{diag.get('baseline_key', 'persist')}` on validation mean RPS -> promoted: `{promoted}`",
        "",
        "> **Anti-Snooping & Leakage Policy:** Every pooled quantity (AR(1) `phi`, `sigma_eps`, tyres, season hyperparameters, event hazards, race distances, transition baseline and phase noise multipliers) is fitted strictly on training races (`2023-2024`). Promotion and generalization diagnosis use `train` and `validation` exclusively; the `test` split remains sealed unless explicitly unsealed once via `m7-evaluate --split test --unseal-test`.",
        "",
        "## 2. Comparative Metrics (`persist` vs `transition` vs `race_model`)",
        "",
        "| Bucket | Race Model RPS | Persist RPS | Transition RPS | Model Pairwise LL | Transition Pairwise LL | Δ vs Persist [95% CI] | Δ vs Transition [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    bucket_order = [b for b in by_bucket if b != "all"] + (["all"] if "all" in by_bucket else [])
    for bucket in bucket_order:
        row = by_bucket[bucket]
        label = "**all (overall)**" if bucket == "all" else bucket
        lines.append(
            f"| {label} | {fmt(row.get('model_rps'))} | {fmt(row.get('persist_rps'))} | "
            f"{fmt(row.get('transition_rps'))} | {fmt(row.get('model_pll'))} | "
            f"{fmt(row.get('transition_pll'))} | {fmt_ci(row.get('delta_rps_vs_persist'))} | "
            f"{fmt_ci(row.get('delta_rps_vs_transition'))} |"
        )

    lines.extend([
        "",
        "### Event Calibration (Brier Scores)",
        "",
        "| Event | Brier Score | Notes |",
        "|---|---:|---|",
    ])
    for event_name in ("win", "podium", "points", "dnf"):
        entry = calib.get(event_name) or {}
        extra = ""
        if event_name == "dnf" and _is_finite_real(entry.get("mean_forecast")) and _is_finite_real(entry.get("observed")):
            extra = f"mean forecast {float(entry['mean_forecast']):.3f} vs observed {float(entry['observed']):.3f}"
        lines.append(f"| `{event_name}` | {fmt(entry.get('brier'))} | {extra} |")

    ratio_val = diag.get("val_to_train_prior_rmse_ratio")
    ratio_str = f"{float(ratio_val):.2f}x" if _is_finite_real(ratio_val) else "n/a"
    lines.extend([
        "",
        "## 3. Overfitting / Underfitting Diagnosis (`train` vs `validation`)",
        "",
        "| Model | Diagnosis | Train Prior RMSE | Val Prior RMSE | Prior Ratio | Val z² | Val RPS | Persist RPS | Transition RPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `race_model` | `{diag.get('status', 'n/a')}` | {fmt(diag.get('train_prior_rmse'))} | "
        f"{fmt(diag.get('validation_prior_rmse'))} | {ratio_str} | "
        f"{fmt(diag.get('validation_prior_z2'), 2)} | {fmt(diag.get('validation_rps'))} | "
        f"{fmt(diag.get('baseline_validation_rps'))} | {fmt(diag.get('transition_validation_rps'))} |",
        "",
        f"- **{diag.get('status', 'n/a')}:** {diag.get('explanation', '')}",
        "",
        "## 4. Interpretability — Structural Parameter & Variance Decomposition (`train`)",
        "",
    ])

    l1_interp = interp.get("level1_pace") or {}
    lines.append(
        f"- **Level 1 Pace (clean laps):** `phi = {fmt(l1_interp.get('phi_ar1'))}`, "
        f"`sigma_eps = {fmt(l1_interp.get('sigma_eps_pct'))}%`, "
        f"AR(1) long-run SD multiplier `= {fmt(l1_interp.get('ar1_long_run_sd_multiplier'), 3)}x`"
    )
    tyres = l1_interp.get("tyre_compounds") or []
    if tyres:
        lines.extend([
            "",
            "| Compound | Offset (% lap) | Degradation Slope (% / lap) |",
            "|---|---:|---:|",
        ])
        for t in tyres:
            lines.append(
                f"| `{t['compound']}` | {fmt(t.get('offset_pct'))} | {fmt(t.get('degradation_slope_pct_per_lap'))} |"
            )

    l2_interp = interp.get("level2_season_variance") or {}
    comps = l2_interp.get("components") or []
    if comps:
        lines.extend([
            "",
            "| Level-2 Variance Component | SD (% lap) | Variance | Share |",
            "|---|---:|---:|---:|",
        ])
        for c in comps:
            share_val = c.get("variance_share")
            share_str = f"{float(share_val) * 100:.1f}%" if _is_finite_real(share_val) else "n/a"
            lines.append(
                f"| `{c['component']}` | {fmt(c.get('sd_pct'))} | {fmt(c.get('variance'))} | {share_str} |"
            )
        lines.append("")
        lines.append(
            f"- **Winter reset SD (`q_gamma_season`):** `{fmt(l2_interp.get('winter_team_reset_sd_pct'))}%` · "
            f"**Qualifying weight (`lambda`):** `{fmt(l2_interp.get('qualifying_weight_lambda'))}` · "
            f"**Qualifying noise (`sigma_q`):** `{fmt(l2_interp.get('qualifying_noise_sd_pct'))}%`"
        )

    lines.extend([
        "",
        f"> **Interpretability Note:** {interp.get('note', STRUCTURAL_INTERPRETABILITY_NOTE)}",
        "",
        "## 5. Model Registry & Artifact Integrity Status",
        "",
        f"- **Version:** `{version}`",
        f"- **Promoted to `latest.json`:** `{promoted}`",
    ])
    if verification:
        lines.append(
            f"- **Immutable Package Verification (`PUBLISHED` + `manifest.json`):** "
            f"`ok={verification.get('ok')}` (`files_checked={verification.get('files_checked')}`, "
            f"`manifest_sha256={verification.get('manifest_sha256')}`)"
        )
    if mlflow_info:
        lines.append(
            f"- **MLflow Registry:** `{mlflow_info.get('model_uri')}` (run `{mlflow_info.get('run_id')}`)"
        )
    else:
        lines.append("- **MLflow Registry:** not logged in this run (pass `--log-mlflow` to log & register locally)")
    lines.append("")
    return "\n".join(lines)


# -- 6. Serving Input / Output Contract Validation ----------------------------

_VALID_REGIMES = frozenset({"GREEN", "SC", "VSC", "RED"})


def _validate_model_version_field(value: Any, field_name: str = "model_version") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name!r} must be a non-empty string")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"{field_name!r} contains invalid path characters: {value!r}")
    return value


def validate_forecast_request(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an on-demand or live forecast request against the serving contract.

    Rejects unknown keys, invalid types, conflicting live/on-demand parameters,
    out-of-range ``lap``/``total_laps``, and missing required routing fields
    (mirroring ``serving/contract.py`` for M8).
    """
    if not isinstance(event, Mapping):
        raise ValueError("forecast event must be a JSON object")
    unknown = sorted(set(event) - ALLOWED_FORECAST_REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unknown forecast event field(s): {unknown}")

    if "mode" in event:
        mode = event["mode"]
        if mode != "live":
            raise ValueError(f"unsupported mode {mode!r}; expected 'live'")
        conflicting = [
            key
            for key in ("session_key", "lap", "total_laps", "year")
            if key in event and event[key] is not None
        ]
        if conflicting:
            raise ValueError(
                f"live mode cannot be combined with on-demand field(s): {conflicting}"
            )
        if "source" in event and event["source"] != "live":
            raise ValueError(
                f"live mode only permits 'source': 'live' when provided, got {event['source']!r}"
            )
        if "model_version" in event and event["model_version"] is not None:
            _validate_model_version_field(event["model_version"])
        return dict(event)

    if "session_key" not in event or not _is_int(event["session_key"]) or int(event["session_key"]) <= 0:
        raise ValueError("event needs a positive integer 'session_key' or 'mode': 'live'")

    lap_val: Optional[int] = None
    if "lap" in event and event["lap"] is not None:
        if not _is_int(event["lap"]) or int(event["lap"]) < 1:
            raise ValueError("'lap' must be an integer >= 1 when provided")
        lap_val = int(event["lap"])

    total_laps_val: Optional[int] = None
    if "total_laps" in event and event["total_laps"] is not None:
        if not _is_int(event["total_laps"]) or int(event["total_laps"]) < 1:
            raise ValueError("'total_laps' must be an integer >= 1 when provided")
        total_laps_val = int(event["total_laps"])

    if lap_val is not None and total_laps_val is not None and lap_val > total_laps_val:
        raise ValueError(f"'lap' ({lap_val}) cannot exceed 'total_laps' ({total_laps_val})")

    if "year" in event and event["year"] is not None:
        if not _is_int(event["year"]) or int(event["year"]) < 2023:
            raise ValueError("'year' must be an integer >= 2023 when provided")

    if "source" in event:
        source = event["source"]
        if not isinstance(source, str) or source not in ("raw", "live"):
            raise ValueError(f"'source' must be 'raw' or 'live', got {source!r}")

    if "model_version" in event and event["model_version"] is not None:
        _validate_model_version_field(event["model_version"])

    return dict(event)


def _validate_square_probability_matrix(raw: Any, n: int, name: str) -> np.ndarray:
    """Strictly validate a 2D N x N matrix of finite probabilities in [0, 1]."""
    if isinstance(raw, np.ndarray):
        if raw.ndim != 2 or raw.shape != (n, n):
            raise ValueError(f"forecast {name} shape {raw.shape} does not match ({n}, {n})")
        if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"forecast {name} must contain finite real numbers")
        rows_iter: Any = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        if len(raw) != n:
            raise ValueError(f"forecast {name} row count {len(raw)} does not match {n} drivers")
        rows_iter = raw
    else:
        raise ValueError(f"forecast {name} must be a 2D {n}x{n} list or array")

    parsed = np.empty((n, n), dtype=float)
    for i, row in enumerate(rows_iter):
        if isinstance(row, np.ndarray):
            if row.ndim != 1 or len(row) != n or row.dtype == np.bool_:
                raise ValueError(f"forecast {name}[{i}] must have length {n} with numeric values")
            row_seq: Any = row.tolist()
        elif isinstance(row, (list, tuple)):
            if len(row) != n:
                raise ValueError(f"forecast {name}[{i}] length {len(row)} does not match {n} drivers")
            row_seq = row
        else:
            raise ValueError(f"forecast {name}[{i}] must be a sequence of length {n}")
        for j, cell in enumerate(row_seq):
            if not _is_finite_real(cell):
                raise ValueError(f"forecast {name}[{i}][{j}] must be a finite real number, got {cell!r}")
            val = float(cell)
            if val < 0.0 or val > 1.0:
                raise ValueError(f"forecast {name}[{i}][{j}]={val!r} must be within [0, 1]")
            parsed[i, j] = val
    return parsed


def validate_forecast_response(payload: Mapping[str, Any], *, atol: float = 1e-4) -> None:
    """Validate the forecast output contract (doubly stochastic N x N matrix + Pr(DNF)).

    Enforces strict types, ranges (``lap``, ``total_laps``, ``[0, 1]`` probabilities
    without negative/over-1 tolerance slack), driver uniqueness, 2D matrix structure,
    and driver-to-matrix correspondence.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("forecast response must be a JSON object")

    for required in ("model_version", "lap", "total_laps", "drivers", "matrix"):
        if required not in payload:
            raise ValueError(f"forecast response missing required field {required!r}")

    _validate_model_version_field(payload["model_version"], "model_version")

    lap = payload["lap"]
    total_laps = payload["total_laps"]
    if not _is_int(lap) or int(lap) < 1:
        raise ValueError(f"forecast response 'lap' must be an integer >= 1, got {lap!r}")
    if not _is_int(total_laps) or int(total_laps) < 1:
        raise ValueError(f"forecast response 'total_laps' must be an integer >= 1, got {total_laps!r}")
    if int(lap) > int(total_laps):
        raise ValueError(f"forecast response 'lap' ({int(lap)}) cannot exceed 'total_laps' ({int(total_laps)})")

    if "session_key" in payload:
        if not _is_int(payload["session_key"]) or int(payload["session_key"]) <= 0:
            raise ValueError(f"forecast response 'session_key' must be a positive integer, got {payload['session_key']!r}")
    if "race" in payload:
        if not isinstance(payload["race"], str) or not payload["race"].strip():
            raise ValueError(f"forecast response 'race' must be a non-empty string, got {payload['race']!r}")
    if "scenarios" in payload:
        if not _is_int(payload["scenarios"]) or int(payload["scenarios"]) < 1:
            raise ValueError(f"forecast response 'scenarios' must be an integer >= 1, got {payload['scenarios']!r}")
    if "regime_at_k" in payload:
        if not isinstance(payload["regime_at_k"], str) or payload["regime_at_k"] not in _VALID_REGIMES:
            raise ValueError(f"forecast response 'regime_at_k' must be one of {sorted(_VALID_REGIMES)}, got {payload['regime_at_k']!r}")
    if "generated_at" in payload:
        if not isinstance(payload["generated_at"], str) or not payload["generated_at"].strip():
            raise ValueError("forecast response 'generated_at' must be a non-empty string")

    drivers = payload["drivers"]
    if not isinstance(drivers, (list, tuple)) or len(drivers) == 0:
        raise ValueError("forecast response 'drivers' must be a non-empty list")
    n = len(drivers)

    matrix = _validate_square_probability_matrix(payload["matrix"], n, "matrix")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=atol):
        raise ValueError("forecast matrix rows must sum to 1.0")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=atol):
        raise ValueError("forecast matrix columns must sum to 1.0")

    seen_driver_ids = set()
    seen_driver_numbers = set()
    any_driver_number = any(isinstance(d, Mapping) and "driver_number" in d for d in drivers)
    positions = np.arange(1, n + 1, dtype=float)

    for idx, driver in enumerate(drivers):
        if not isinstance(driver, Mapping):
            raise ValueError(f"drivers[{idx}] must be a JSON object")
        driver_id = driver.get("driver")
        if not isinstance(driver_id, str) or not driver_id.strip():
            raise ValueError(f"drivers[{idx}] must include a non-empty string 'driver'")
        if driver_id in seen_driver_ids:
            raise ValueError(f"duplicate driver identifier {driver_id!r} in forecast response")
        seen_driver_ids.add(driver_id)

        if any_driver_number:
            dnum = driver.get("driver_number")
            if not _is_int(dnum) or int(dnum) <= 0:
                raise ValueError(f"driver {driver_id!r} must have a positive integer 'driver_number', got {dnum!r}")
            if int(dnum) in seen_driver_numbers:
                raise ValueError(f"duplicate driver_number {int(dnum)} in forecast response")
            seen_driver_numbers.add(int(dnum))

        if "p_dnf" not in driver:
            raise ValueError(f"driver {driver_id!r} missing required 'p_dnf'")
        p_dnf = driver["p_dnf"]
        if not _is_finite_real(p_dnf) or not (0.0 <= float(p_dnf) <= 1.0):
            raise ValueError(f"driver {driver_id!r} has invalid p_dnf={p_dnf!r}; must be in [0, 1]")

        if "retired_at_k" in driver and not isinstance(driver["retired_at_k"], (bool, np.bool_)):
            raise ValueError(f"driver {driver_id!r} 'retired_at_k' must be a boolean")

        if "p_position" in driver:
            p_pos = driver["p_position"]
            if not isinstance(p_pos, (list, tuple)) or len(p_pos) != n:
                raise ValueError(f"driver {driver_id!r} 'p_position' must be a list of length {n}")
            for j, val in enumerate(p_pos):
                if not _is_finite_real(val) or not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"driver {driver_id!r} p_position[{j}]={val!r} must be in [0, 1]")
            if not np.allclose(np.asarray(p_pos, dtype=float), matrix[idx], atol=atol):
                raise ValueError(f"driver {driver_id!r} 'p_position' does not match matrix row {idx}")

        for prob_key, expected in (
            ("p_win", float(matrix[idx, 0])),
            ("p_podium", float(matrix[idx, :3].sum())),
            ("p_points", float(matrix[idx, :10].sum())),
        ):
            if prob_key in driver:
                val = driver[prob_key]
                if not _is_finite_real(val) or not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"driver {driver_id!r} has invalid {prob_key}={val!r}; must be in [0, 1]")
                if abs(float(val) - expected) > max(atol, 1e-4):
                    raise ValueError(
                        f"driver {driver_id!r} {prob_key}={float(val):.6f} does not match matrix ({expected:.6f})"
                    )

        if "expected_position" in driver:
            exp_pos = driver["expected_position"]
            if not _is_finite_real(exp_pos):
                raise ValueError(f"driver {driver_id!r} has non-finite expected_position={exp_pos!r}")
            expected_e = float(matrix[idx] @ positions)
            if abs(float(exp_pos) - expected_e) > 1e-2:
                raise ValueError(
                    f"driver {driver_id!r} expected_position={float(exp_pos):.3f} does not match matrix ({expected_e:.3f})"
                )

    if "ahead" in payload:
        ahead = _validate_square_probability_matrix(payload["ahead"], n, "ahead")
        if not np.allclose(np.diag(ahead), 0.0, atol=atol):
            raise ValueError("forecast 'ahead' matrix diagonal must be 0.0")
        off_diag = ahead + ahead.T
        if not np.allclose(off_diag, 1.0 - np.eye(n), atol=max(atol, 1e-3)):
            raise ValueError("forecast 'ahead' off-diagonal entries must satisfy ahead[i,j] + ahead[j,i] == 1.0")


# -- 7. Optional Lazy MLflow Tracking & Model Registry ------------------------


def _sqlite_uri(path: Path) -> str:
    resolved = path.resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _fallback_example_race_payload() -> Tuple[Dict[str, Any], int, int]:
    """Minimal self-contained race snapshot for MLflow signature inference when no race is passed."""
    from datetime import timedelta

    t0 = datetime(2025, 3, 16, 4, 0, tzinfo=timezone.utc)
    drivers = [
        {"driver_number": i + 1, "name_acronym": f"D{i + 1}", "team_name": f"Team{(i // 2) + 1}"}
        for i in range(4)
    ]
    laps: List[Dict[str, Any]] = []
    intervals: List[Dict[str, Any]] = []
    stints: List[Dict[str, Any]] = []
    for i in range(4):
        d = i + 1
        t = t0 + timedelta(seconds=0.5 * i)
        for lap in range(1, 5):
            dur = 90.0 + 0.2 * i
            laps.append(
                {
                    "driver_number": d,
                    "lap_number": lap,
                    "lap_duration": dur,
                    "date_start": t.isoformat(),
                    "is_pit_out_lap": False,
                }
            )
            t = t + timedelta(seconds=dur)
            intervals.append(
                {
                    "date": t.isoformat(),
                    "driver_number": d,
                    "interval": None if i == 0 else 2.0,
                    "gap_to_leader": 0.0 if i == 0 else 2.0 * i,
                }
            )
        stints.append(
            {
                "driver_number": d,
                "stint_number": 1,
                "compound": "MEDIUM",
                "lap_start": 1,
                "lap_end": 4,
                "tyre_age_at_start": 0,
            }
        )
    payload = {
        "ref": {
            "year": 2025,
            "meeting_key": 1,
            "session_key": 1,
            "location": "ExampleTrack",
            "date_start": t0.isoformat(),
            "circuit_key": None,
            "circuit_type": None,
            "qualifying_session_key": None,
        },
        "drivers": drivers,
        "laps": laps,
        "stints": stints,
        "pit": [],
        "race_control": [],
        "intervals": intervals,
        "starting_grid": [],
    }
    return payload, 3, 10


def log_race_model_to_mlflow(
    *,
    settings: Any,
    artifact: Any,
    evaluation_report: Mapping[str, Any],
    markdown_report: str,
    model_bytes: bytes,
    promoted: bool,
    verification: Optional[Mapping[str, Any]] = None,
    register_model: bool = True,
    example_race: Optional[Any] = None,
    example_lap: Optional[int] = None,
    example_total_laps: Optional[int] = None,
) -> Dict[str, Any]:
    """Optionally log a race-model evaluation run & predictive Pyfunc artifact to local MLflow.

    MLflow and ``RaceModelPyfuncModel`` are imported strictly lazily inside this
    function so importing ``race_model`` modules never loads MLflow. Uses the
    exact same local SQLite tracking/registry store as ``modeling/training.py``
    so LinearRegression, TabularMLP, and RaceModel appear side-by-side in ``mlflow ui``.
    """
    try:
        import mlflow
        import mlflow.pyfunc
        from mlflow.models import infer_signature
        import scipy
        from f1_race_intelligence.race_model.mlflow_pyfunc import RaceModelPyfuncModel
    except ImportError as exc:
        raise RuntimeError(
            f"MLflow logging for race_model requires the optional `training` extra; install `{MLFLOW_INSTALL_HINT}` first"
        ) from exc

    from types import SimpleNamespace
    from f1_race_intelligence.race_model.racedata import race_to_forecast_payload

    training_cfg = settings.training
    race_cfg = settings.race_model
    db_path = Path(training_cfg.tracking_db_path)
    artifact_root = Path(training_cfg.artifact_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = _sqlite_uri(db_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(training_cfg.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            training_cfg.experiment_name,
            artifact_location=_file_uri(artifact_root),
        )
    else:
        experiment_id = experiment.experiment_id

    prov = evaluation_report.get("provenance") or getattr(artifact, "provenance", None) or {}
    split = str(evaluation_report.get("split", "validation"))
    reg_name = race_cfg.registered_model_name if register_model else None

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"m7-race-model-{artifact.version}",
    ) as active_run:
        run_id = active_run.info.run_id
        tags = {
            "project": "f1-race-intelligence",
            "stage": "m7-race-model",
            "model_type": MODEL_TYPE,
            "model_version": str(artifact.version),
            "evaluated_split": split,
            "promoted": str(bool(promoted)).lower(),
            "test_sealed": str(bool(prov.get("test_sealed", True))).lower(),
            "input_fingerprint": str(prov.get("input_fingerprint") or ""),
            "source_commit": str(prov.get("source_commit") or ""),
        }
        mlflow.set_tags(tags)

        season_params = (artifact.season or {}).get("state", {}).get("params") or {}
        params: Dict[str, Any] = {
            "model_type": MODEL_TYPE,
            "estimator": ESTIMATOR_NAME,
            "reference_baseline": race_cfg.promote_requires_beating,
            "random_seed": race_cfg.random_seed,
            "burn_in_races": race_cfg.burn_in_races,
            "scenarios": artifact.forecast.get("scenarios"),
            "noise_scale": artifact.forecast.get("noise_scale"),
            "noise_scale_per_phase": race_cfg.noise_scale_per_phase,
            "level1_phi": round(float(artifact.level1.get("phi", 0.0)), 6),
            "level1_sigma_eps": round(float(artifact.level1.get("sigma_eps", 0.0)), 6),
            "input_fingerprint": prov.get("input_fingerprint"),
            "config_fingerprint": prov.get("config_fingerprint"),
            "source_commit": prov.get("source_commit"),
        }
        for key in ("q_gamma", "q_gamma_season", "q_delta", "s_c", "s_w", "sigma_q", "lam"):
            if key in season_params:
                params[f"season_{key}"] = round(float(season_params[key]), 6)
        mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

        metrics_to_log: Dict[str, float] = {}
        by_bucket = evaluation_report.get("by_bucket") or {}
        for bucket, vals in by_bucket.items():
            tag = "overall" if bucket == "all" else bucket.replace("%", "pct")
            for src_key, dst_key in (
                ("model_rps", f"{split}.race_model.rps.{tag}"),
                ("persist_rps", f"{split}.persist.rps.{tag}"),
                ("transition_rps", f"{split}.transition.rps.{tag}"),
                ("model_pll", f"{split}.race_model.pairwise_log_loss.{tag}"),
                ("transition_pll", f"{split}.transition.pairwise_log_loss.{tag}"),
            ):
                if _is_finite_real(vals.get(src_key)):
                    metrics_to_log[dst_key] = float(vals[src_key])

        calib = evaluation_report.get("calibration") or {}
        for ev in ("win", "podium", "points", "dnf"):
            brier_val = (calib.get(ev) or {}).get("brier")
            if _is_finite_real(brier_val):
                metrics_to_log[f"{split}.race_model.brier.{ev}"] = float(brier_val)

        prior_check = (artifact.metrics or {}).get("prior_check") or {}
        for pc_split in ("train", "validation"):
            pc = prior_check.get(pc_split) or {}
            if _is_finite_real(pc.get("rmse")):
                metrics_to_log[f"{pc_split}.race_model.prior_rmse"] = float(pc["rmse"])
            if _is_finite_real(pc.get("z2")):
                metrics_to_log[f"{pc_split}.race_model.prior_z2"] = float(pc["z2"])

        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / MODEL_FILENAME
            model_path.write_bytes(model_bytes)

            def _write_json(name: str, obj: Any) -> Path:
                p = tmp / name
                p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
                return p

            mlflow.log_artifact(str(_write_json("data_provenance.json", prov)))
            mlflow.log_artifact(str(_write_json("training_config.json", artifact.settings)))
            mlflow.log_artifact(str(_write_json("metrics.json", metrics_to_log)))
            mlflow.log_artifact(str(_write_json(f"evaluate_{split}.json", evaluation_report)))
            if evaluation_report.get("interpretability"):
                mlflow.log_artifact(str(_write_json("interpretability.json", evaluation_report["interpretability"])))
            if verification:
                mlflow.log_artifact(str(_write_json("package_verification.json", verification)))
            md_path = tmp / f"evaluate_{split}.md"
            md_path.write_text(markdown_report, encoding="utf-8")
            mlflow.log_artifact(str(md_path))

            if example_race is not None:
                ex_payload = race_to_forecast_payload(example_race)
                ex_lap = int(example_lap or getattr(example_race, "truncated_at", None) or example_race.leader_laps())
                ex_total = int(example_total_laps or max(ex_lap, example_race.leader_laps()))
            else:
                ex_payload, ex_lap, ex_total = _fallback_example_race_payload()

            example_in = pd.DataFrame(
                [
                    {
                        "race_json": canonical_json_bytes(ex_payload).decode("utf-8"),
                        "lap": ex_lap,
                        "total_laps": ex_total,
                        "seed": int(race_cfg.random_seed),
                    }
                ]
            )
            probe = RaceModelPyfuncModel()
            probe.load_context(SimpleNamespace(artifacts={"model_json": str(model_path)}))
            example_out = probe.predict(None, example_in)
            signature = infer_signature(example_in, example_out)
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RaceModelPyfuncModel(),
                artifacts={"model_json": str(model_path)},
                registered_model_name=reg_name,
                signature=signature,
                input_example=example_in,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"numpy=={np.__version__}",
                    f"pandas=={pd.__version__}",
                    f"scipy=={scipy.__version__}",
                ],
            )

    registered_version: Optional[str] = None
    model_uri = f"runs:/{run_id}/model"
    if reg_name:
        versions = client.search_model_versions(f"name='{reg_name}' and run_id='{run_id}'")
        if versions:
            registered_version = str(versions[0].version)
            model_uri = f"models:/{reg_name}/{registered_version}"

    return {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": training_cfg.experiment_name,
        "registered_model_name": reg_name,
        "registered_version": registered_version,
        "model_uri": model_uri,
        "logged_model_uri": getattr(model_info, "model_uri", f"runs:/{run_id}/model"),
    }
