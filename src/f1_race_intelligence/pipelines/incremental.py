"""Verified per-session completion state for incremental pipeline planning."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.ingestion.client import _is_retryable
from f1_race_intelligence.ingestion.exceptions import OpenF1HTTPError
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends.base import join_key
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore

STATE_SCHEMA_VERSION = 2
_COMPLETED_STAGES = {"m2", "m3", "m4", "m5"}
_TRACKED_STAGES = ("m3", "m4", "m5")


class SessionStateError(ValueError):
    """The stored session state cannot prove that the session is reusable."""


def record_stage_provenance(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    stage: str,
    *,
    expected_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Bind a stage report/output to the exact inputs observed after the stage ran."""
    if stage not in _TRACKED_STAGES:
        return False, f"unsupported stage provenance {stage!r}"
    if not scope.is_session or scope.year is None or scope.meeting_key is None:
        return False, "session scope is incomplete"

    try:
        m2_key, m2_manifest = _latest_session_report(
            store,
            store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path),
            scope,
            "historical_extraction",
        )
        raw_artifacts, absent = _raw_artifacts(store, settings, scope, m2_key, m2_manifest)
        m2_hash = _hash_artifact(store, m2_key)["sha256"]
        current_inputs = {
            "raw_artifacts": raw_artifacts,
            "absent_endpoints": absent,
            "m2_manifest_sha256": m2_hash,
        }
        if stage in {"m4", "m5"}:
            prerequisite = "m3" if stage == "m4" else "m4"
            prerequisite_root, prerequisite_name = _stage_report_location(settings, store, prerequisite)
            prerequisite_key, _ = _latest_session_report(store, prerequisite_root, scope, prerequisite_name)
            current_inputs["prerequisite_report"] = _hash_artifact(store, prerequisite_key)
        if stage == "m5":
            consolidated_key = store.layout.session_dataset_key(
                store.root(Layer.CONSOLIDATED, settings.consolidation.output_path),
                scope.year,
                scope.session_key,
            )
            current_inputs["prerequisite_output"] = _hash_artifact(store, consolidated_key)
        if expected_inputs is not None and expected_inputs != current_inputs:
            return False, f"{stage.upper()} inputs changed while the stage was running"
        report_root, pipeline_name = _stage_report_location(settings, store, stage)
        report_key, report = _latest_session_report(store, report_root, scope, pipeline_name)
        _validate_stage_report(store, stage, report, scope, raw_artifacts, m2_key, settings)
        record: Dict[str, Any] = {
            **current_inputs,
            "report": _hash_artifact(store, report_key),
        }
        stages_key = _stage_state_key(store, settings, scope)
        try:
            stage_state = store.backend.read_json(stages_key)
        except FileNotFoundError:
            stage_state = {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
        if not isinstance(stage_state, dict) or stage_state.get("schema_version") != STATE_SCHEMA_VERSION:
            stage_state = {"schema_version": STATE_SCHEMA_VERSION, "stages": {}}
        stages = stage_state.get("stages")
        if not isinstance(stages, dict):
            stages = {}

        if stage in {"m4", "m5"}:
            prerequisite = "m3" if stage == "m4" else "m4"
            previous = stages.get(prerequisite)
            if not isinstance(previous, dict) or not _same_inputs(previous, record):
                return False, f"{prerequisite.upper()} provenance does not match current inputs"
            if stage == "m4" and previous.get("report") != record.get("prerequisite_report"):
                return False, "M3 report changed before M4 completed"
            if stage == "m5" and previous.get("report") != record.get("prerequisite_report"):
                return False, "M4 report changed before M5 completed"
        if stage == "m4":
            validation = report.get("source_validation")
            m3_report = stages["m3"]["report"]
            config = report.get("config")
            if not isinstance(validation, dict) or not _same_location(
                validation.get("report"), store.backend.uri(m3_report["key"])
            ):
                return False, "M4 report does not reference the verified M3 report"
            raw_uri = store.backend.uri(store.root(Layer.RAW, settings.consolidation.raw_path))
            if not isinstance(config, dict) or not (
                _same_location(config.get("raw_path"), settings.consolidation.raw_path)
                or _same_location(config.get("raw_path"), raw_uri)
            ):
                return False, "M4 report does not identify the configured raw input"
            out_root = store.root(Layer.CONSOLIDATED, settings.consolidation.output_path)
            if not (
                _same_location(config.get("output_path"), settings.consolidation.output_path)
                or _same_location(config.get("output_path"), store.backend.uri(out_root))
            ):
                return False, "M4 report does not identify the configured output location"
        if stage == "m5":
            config = report.get("config")
            if not isinstance(config, dict) or config.get("input_path") != settings.features.input_path:
                return False, "M5 report does not identify the configured M4 input path"

        if stage in {"m4", "m5"}:
            layer = Layer.CONSOLIDATED if stage == "m4" else Layer.FEATURES
            configured = settings.consolidation.output_path if stage == "m4" else settings.features.output_path
            artifact_key = store.layout.session_dataset_key(store.root(layer, configured), scope.year, scope.session_key)
            record["output_artifact"] = _hash_artifact(store, artifact_key)
            expected_uri = store.backend.uri(artifact_key)
            session_record = _session_report_record(report, scope)
            if not session_record or not _same_location(session_record.get("output_path"), expected_uri):
                return False, f"{stage.upper()} report does not identify the expected output"
        if stage == "m5":
            previous = stages["m4"]
            if previous.get("output_artifact") != _hash_artifact(
                store,
                store.layout.session_dataset_key(
                    store.root(Layer.CONSOLIDATED, settings.consolidation.output_path),
                    scope.year,
                    scope.session_key,
                ),
            ):
                return False, "M4 output changed after its provenance was recorded"
            if previous.get("output_artifact") != record.get("prerequisite_output"):
                return False, "M4 output changed while M5 was running"

        stages[stage] = record
        store.backend.write_json(stages_key, stage_state)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)
    return True, f"{stage.upper()} input/output provenance recorded"


def capture_stage_inputs(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    stage: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Snapshot a stage's inputs before execution for a before/after comparison."""
    if stage not in _TRACKED_STAGES or not scope.is_session or scope.year is None or scope.meeting_key is None:
        return None, "stage or session scope is incomplete"
    try:
        m2_key, manifest = _latest_session_report(
            store,
            store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path),
            scope,
            "historical_extraction",
        )
        raw, absent = _raw_artifacts(store, settings, scope, m2_key, manifest)
        snapshot: Dict[str, Any] = {
            "raw_artifacts": raw,
            "absent_endpoints": absent,
            "m2_manifest_sha256": _hash_artifact(store, m2_key)["sha256"],
        }
        if stage in {"m4", "m5"}:
            prerequisite = "m3" if stage == "m4" else "m4"
            root, pipeline_name = _stage_report_location(settings, store, prerequisite)
            report_key, _ = _latest_session_report(store, root, scope, pipeline_name)
            snapshot["prerequisite_report"] = _hash_artifact(store, report_key)
        if stage == "m5":
            consolidated_key = store.layout.session_dataset_key(
                store.root(Layer.CONSOLIDATED, settings.consolidation.output_path),
                scope.year,
                scope.session_key,
            )
            snapshot["prerequisite_output"] = _hash_artifact(store, consolidated_key)
        return snapshot, "stage inputs captured"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, str(exc)


def record_session_completion(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
) -> Tuple[bool, str]:
    """Record hashes and stage provenance after successful per-session M5."""
    if not scope.is_session or scope.year is None or scope.meeting_key is None:
        return False, "session scope is incomplete"

    try:
        m2_key, m2_manifest = _latest_session_report(
            store,
            store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path),
            scope,
            "historical_extraction",
        )
        raw_artifacts, absent_endpoints = _raw_artifacts(store, settings, scope, m2_key, m2_manifest)
        reports = _stage_reports(store, settings, scope, raw_artifacts, m2_key)
        derived_artifacts = _derived_artifacts(store, settings, scope)
        _verify_stage_provenance(store, settings, scope, m2_key, raw_artifacts, absent_endpoints)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "session": scope.to_dict(),
        "config_fingerprint": _config_fingerprint(settings, store),
        "completed_stages": sorted(_COMPLETED_STAGES),
        "raw_artifacts": raw_artifacts,
        "absent_endpoints": absent_endpoints,
        "derived_artifacts": derived_artifacts,
        "reports": reports,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    store.backend.write_json(_state_key(store, settings, scope), state)
    return True, "verified completion state recorded"


def session_reuse_decision(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
) -> Tuple[bool, str]:
    """Return whether the latest per-session M2-M5 result is safe to omit."""
    if not scope.is_session or scope.year is None or scope.meeting_key is None:
        return False, "session scope is incomplete"

    key = _state_key(store, settings, scope)
    try:
        state = store.backend.read_json(key)
    except FileNotFoundError:
        return False, "no verified completion provenance"
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return False, f"completion state is unreadable: {exc}"

    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
        return False, "completion state schema is unknown"
    if state.get("session") != scope.to_dict():
        return False, "completion state belongs to a different session"
    if state.get("completed_stages") != sorted(_COMPLETED_STAGES):
        return False, "one or more M2-M5 stages lack completion provenance"
    if state.get("config_fingerprint") != _config_fingerprint(settings, store):
        return False, "pipeline configuration changed"

    try:
        m2_key, m2_manifest = _latest_session_report(
            store,
            store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path),
            scope,
            "historical_extraction",
        )
        current_raw, current_absent = _raw_artifacts(store, settings, scope, m2_key, m2_manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"raw artifacts are missing or unverifiable: {exc}"
    if state.get("raw_artifacts") != current_raw or state.get("absent_endpoints") != current_absent:
        return False, "raw input artifacts changed"

    try:
        current_derived = _derived_artifacts(store, settings, scope)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"derived artifacts are missing or unverifiable: {exc}"
    if state.get("derived_artifacts") != current_derived:
        return False, "derived artifacts are missing or changed"

    try:
        current_reports = _stage_reports(store, settings, scope, current_raw, m2_key)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"stage provenance is missing or unverifiable: {exc}"
    if state.get("reports") != current_reports:
        return False, "stage reports changed or no longer verify"
    try:
        _verify_stage_provenance(store, settings, scope, m2_key, current_raw, current_absent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"stage input/output provenance is missing or stale: {exc}"
    return True, "all M2-M5 artifacts and provenance match"


def _verify_stage_provenance(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    m2_key: str,
    raw_artifacts: List[Dict[str, Any]],
    absent: List[Dict[str, Any]],
) -> None:
    stage_state = store.backend.read_json(_stage_state_key(store, settings, scope))
    if not isinstance(stage_state, dict) or stage_state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise SessionStateError("stage state schema is missing or unknown")
    stages = stage_state.get("stages")
    if not isinstance(stages, dict):
        raise SessionStateError("stage provenance is missing")
    m2_hash = _hash_artifact(store, m2_key)["sha256"]
    expected_inputs = {"raw_artifacts": raw_artifacts, "absent_endpoints": absent, "m2_manifest_sha256": m2_hash}
    for stage in _TRACKED_STAGES:
        evidence = stages.get(stage)
        if not isinstance(evidence, dict) or not _same_inputs(evidence, expected_inputs):
            raise SessionStateError(f"{stage.upper()} stage was not verified against the current M2/raw inputs")
        root, pipeline_name = _stage_report_location(settings, store, stage)
        report_key, _ = _latest_session_report(store, root, scope, pipeline_name)
        if evidence.get("report") != _hash_artifact(store, report_key):
            raise SessionStateError(f"{stage.upper()} report changed after its provenance was recorded")
    m4, m5 = stages["m4"], stages["m5"]
    if m4.get("prerequisite_report") != stages["m3"].get("report"):
        raise SessionStateError("M4 provenance does not reference the verified M3 report")
    if (
        m5.get("prerequisite_report") != m4.get("report")
        or m5.get("prerequisite_output") != m4.get("output_artifact")
    ):
        raise SessionStateError("M5 provenance does not reference the verified M4 report and output")
    consolidated_key = store.layout.session_dataset_key(
        store.root(Layer.CONSOLIDATED, settings.consolidation.output_path), scope.year, scope.session_key
    )
    feature_key = store.layout.session_dataset_key(
        store.root(Layer.FEATURES, settings.features.output_path), scope.year, scope.session_key
    )
    if m4.get("output_artifact") != _hash_artifact(store, consolidated_key):
        raise SessionStateError("M4 output no longer matches its stage provenance")
    if m5.get("output_artifact") != _hash_artifact(store, feature_key):
        raise SessionStateError("M5 output no longer matches its stage provenance")


def _raw_artifacts(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    m2_key: str,
    m2_manifest: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    extraction = settings.historical_extraction
    raw_root = store.root(Layer.RAW, extraction.output_path)
    artifacts: Dict[str, Dict[str, Any]] = {}
    absent: Dict[Tuple[str, int], Dict[str, Any]] = {}
    endpoints = list(extraction.endpoints)
    base = {"year": scope.year, "meeting_key": scope.meeting_key, "session_key": scope.session_key}

    def add_artifact(artifact: Dict[str, Any]) -> None:
        # One entry per storage key: an object is enumerated once, however
        # many endpoints refer to it. Same key with other bytes is a real conflict.
        previous = artifacts.setdefault(artifact["key"], artifact)
        if previous != artifact:
            raise SessionStateError(f"{artifact['key']}: enumerated twice with different content")

    def add_absence(endpoint: str, driver_number: Optional[int] = None) -> None:
        absent[(endpoint, -1 if driver_number is None else driver_number)] = _absence_record(
            endpoint, scope, m2_key, m2_manifest, driver_number
        )

    for endpoint in endpoints:
        if endpoint == "car_data":
            if "drivers" not in endpoints:
                raise SessionStateError("car_data cannot be verified without the configured drivers endpoint")
            driver_key = store.layout.raw_key(raw_root, "drivers", base, f"session_{scope.session_key}.json")
            # The drivers object itself is enumerated by the "drivers" endpoint;
            # here it only supplies the expected per-driver car_data files.
            drivers, driver_artifact = _read_or_verify_absence(
                store, driver_key, "drivers", base, scope, m2_key, m2_manifest
            )
            if driver_artifact is None:
                # No driver list means M2 could not plan car_data either.
                continue
            driver_numbers = sorted(
                {
                    item["driver_number"]
                    for item in drivers
                    if isinstance(item, Mapping) and isinstance(item.get("driver_number"), int)
                }
            )
            if not driver_numbers:
                raise SessionStateError("car_data driver list is empty; expected drivers cannot be verified")
            for driver_number in driver_numbers:
                driver_parameters = {**base, "driver_number": driver_number}
                key = store.layout.raw_key(raw_root, endpoint, driver_parameters, f"driver_{driver_number}.json")
                _, artifact = _read_or_verify_absence(
                    store, key, endpoint, driver_parameters, scope, m2_key, m2_manifest, driver_number
                )
                if artifact is None:
                    add_absence(endpoint, driver_number)
                else:
                    add_artifact(artifact)
            continue

        key = store.layout.raw_key(raw_root, endpoint, base, f"session_{scope.session_key}.json")
        _, artifact = _read_or_verify_absence(store, key, endpoint, base, scope, m2_key, m2_manifest)
        if artifact is None:
            add_absence(endpoint)
        else:
            add_artifact(artifact)
    return (
        [artifacts[key] for key in sorted(artifacts)],
        [absent[key] for key in sorted(absent)],
    )


def _read_or_verify_absence(
    store: DataStore,
    key: str,
    endpoint: str,
    expected_parameters: Mapping[str, Any],
    scope: SessionScope,
    m2_key: str,
    m2_manifest: Dict[str, Any],
    driver_number: Optional[int] = None,
) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
    try:
        raw_bytes = store.backend.read_bytes(key)
    except FileNotFoundError:
        _verify_missing_endpoint(m2_manifest, endpoint, scope, driver_number)
        return [], None
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionStateError(f"{key}: cannot read raw envelope: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("endpoint") != endpoint:
        raise SessionStateError(f"{key}: raw endpoint provenance does not match")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict) or any(
        parameters.get(name) != value for name, value in expected_parameters.items()
    ):
        raise SessionStateError(f"{key}: raw request parameters do not match the session")
    if not isinstance(payload.get("retrieved_at"), str) or not isinstance(payload.get("data"), list):
        raise SessionStateError(f"{key}: raw envelope is incomplete")
    return payload["data"], {
        "key": key,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }


def _verify_missing_endpoint(
    manifest: Dict[str, Any],
    endpoint: str,
    scope: SessionScope,
    driver_number: Optional[int],
) -> None:
    if (
        manifest.get("pipeline_name") != "historical_extraction"
        or not manifest.get("started_at")
        or not manifest.get("finished_at")
        or scope.session_key not in manifest.get("sessions_processed", [])
    ):
        raise SessionStateError(f"{endpoint}: M2 manifest does not prove a completed session request")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise SessionStateError("M2 manifest has no entries")
    match = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("endpoint") == endpoint
            and entry.get("year") == scope.year
            and entry.get("meeting_key") == scope.meeting_key
            and entry.get("session_key") == scope.session_key
            and entry.get("driver_number") == driver_number
        ),
        None,
    )
    if (
        not match
        or match.get("status") != "failed"
        or match.get("status_code") != 404
        or match.get("error_type") != "OpenF1HTTPError"
        or not match.get("error")
        or _is_retryable(OpenF1HTTPError(404, "verified manifest response", endpoint=endpoint))
    ):
        raise SessionStateError(f"{endpoint}: missing raw data has no verified non-retryable M2 HTTP 404")


def _absence_record(
    endpoint: str,
    scope: SessionScope,
    m2_key: str,
    manifest: Dict[str, Any],
    driver_number: Optional[int] = None,
) -> Dict[str, Any]:
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    record = {
        "endpoint": endpoint,
        "status": "not_available",
        "year": scope.year,
        "meeting_key": scope.meeting_key,
        "session_key": scope.session_key,
        "manifest_key": m2_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "status_code": 404,
        "retryable": False,
    }
    if driver_number is not None:
        record["driver_number"] = driver_number
    return record


def _derived_artifacts(store: DataStore, settings: AppSettings, scope: SessionScope) -> List[Dict[str, Any]]:
    artifacts = []
    for layer, configured_path in (
        (Layer.CONSOLIDATED, settings.consolidation.output_path),
        (Layer.FEATURES, settings.features.output_path),
    ):
        root = store.root(layer, configured_path)
        key = store.layout.session_dataset_key(root, scope.year, scope.session_key)
        try:
            data = store.backend.read_bytes(key)
        except OSError as exc:
            raise SessionStateError(f"{key}: Parquet artifact is unreadable: {exc}") from exc
        artifact = {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        try:
            import pandas as pd

            pd.read_parquet(io.BytesIO(data), engine="pyarrow")
        except Exception as exc:
            raise SessionStateError(f"{key}: Parquet artifact is unreadable: {exc}") from exc
        artifacts.append(artifact)
    return artifacts


def _stage_reports(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    raw_artifacts: List[Dict[str, Any]],
    m2_key: str,
) -> List[Dict[str, Any]]:
    extraction_root = store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path)
    validation_root = store.root(Layer.VALIDATION_REPORTS, settings.validation.report_path)
    consolidation_root = store.root(Layer.CONSOLIDATION_REPORTS, settings.consolidation.report_path)
    feature_root = store.root(Layer.FEATURE_REPORTS, settings.features.report_path)
    candidates = [
        ("m2", extraction_root, "historical_extraction"),
        ("m3", validation_root, "data_validation"),
        ("m4", consolidation_root, "data_consolidation"),
        ("m5", feature_root, "feature_engineering"),
    ]
    reports = []
    for stage, root, pipeline_name in candidates:
        key, payload = _latest_session_report(store, root, scope, pipeline_name)
        _validate_stage_report(store, stage, payload, scope, raw_artifacts, m2_key, settings)
        reports.append({"stage": stage, **_hash_artifact(store, key)})
    return reports


def _stage_report_location(settings: AppSettings, store: DataStore, stage: str) -> Tuple[str, str]:
    if stage == "m3":
        return store.root(Layer.VALIDATION_REPORTS, settings.validation.report_path), "data_validation"
    if stage == "m4":
        return store.root(Layer.CONSOLIDATION_REPORTS, settings.consolidation.report_path), "data_consolidation"
    if stage == "m5":
        return store.root(Layer.FEATURE_REPORTS, settings.features.report_path), "feature_engineering"
    raise ValueError(f"unknown stage {stage!r}")


def _same_location(left: Any, right: Any) -> bool:
    """Compare two storage locations ignoring path-separator and trailing-slash style."""
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    return left.replace("\\", "/").rstrip("/") == right.replace("\\", "/").rstrip("/")


def _same_inputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("raw_artifacts", "absent_endpoints", "m2_manifest_sha256")
    )


def _session_report_record(payload: Dict[str, Any], scope: SessionScope) -> Optional[Dict[str, Any]]:
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return None
    return next(
        (
            item
            for item in sessions
            if isinstance(item, dict)
            and item.get("year") == scope.year
            and item.get("session_key") == scope.session_key
        ),
        None,
    )


def _latest_session_report(
    store: DataStore,
    root: str,
    scope: SessionScope,
    pipeline_name: str,
) -> Tuple[str, Dict[str, Any]]:
    prefix = store.layout.records_prefix(root, year=scope.year, session_key=scope.session_key)
    keys = [key for key in store.backend.list(prefix) if key.endswith(".json")]
    for key in reversed(keys):
        try:
            payload = store.backend.read_json(key)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("pipeline_name") == pipeline_name:
            if _report_mentions_session(stage=pipeline_name, payload=payload, scope=scope):
                return key, payload
    raise SessionStateError(f"no verifiable {pipeline_name} report for session {scope.session_key}")


def _report_mentions_session(stage: str, payload: Dict[str, Any], scope: SessionScope) -> bool:
    if stage == "data_validation":
        source = payload.get("source")
        report_scope = source.get("filter") if isinstance(source, dict) else None
        return isinstance(report_scope, dict) and report_scope == scope.to_dict()
    if stage == "historical_extraction":
        entries = payload.get("entries")
        return isinstance(entries, list) and any(
            isinstance(entry, dict)
            and entry.get("year") == scope.year
            and entry.get("meeting_key") == scope.meeting_key
            and entry.get("session_key") == scope.session_key
            for entry in entries
        )
    sessions = payload.get("sessions")
    return isinstance(sessions, list) and any(
        isinstance(item, dict)
        and item.get("year") == scope.year
        and item.get("session_key") == scope.session_key
        for item in sessions
    )


def _validate_stage_report(
    store: DataStore,
    stage: str,
    payload: Dict[str, Any],
    scope: SessionScope,
    raw_artifacts: List[Dict[str, Any]],
    m2_key: str,
    settings: AppSettings,
) -> None:
    if stage == "m2":
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise SessionStateError("M2 extraction manifest has no entries")
        expected = {store.backend.uri(item["key"]) for item in raw_artifacts}
        recorded = {
            entry.get("file_path")
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("year") == scope.year
            and entry.get("meeting_key") == scope.meeting_key
            and entry.get("session_key") == scope.session_key
            and entry.get("status") in {"downloaded", "skipped"}
        }
        if len(expected) != len(raw_artifacts):
            raise SessionStateError("M2 raw inputs are not completely enumerated")
        if expected and not expected <= recorded:
            raise SessionStateError("M2 manifest does not prove every raw artifact")
        if not payload.get("started_at") or not payload.get("finished_at"):
            raise SessionStateError("M2 manifest does not have execution provenance")
        if scope.session_key not in payload.get("sessions_processed", []):
            raise SessionStateError("M2 manifest does not identify this session as processed")
        return

    if stage == "m3":
        if payload.get("global_status") not in {"PASS", "WARNING"}:
            raise SessionStateError(f"M3 validation is not reusable ({payload.get('global_status')!r})")
        source = payload.get("source")
        manifest = source.get("manifest") if isinstance(source, dict) else None
        if not isinstance(source, dict) or not _same_location(
            source.get("raw_path"), store.backend.uri(store.root(Layer.RAW, settings.validation.raw_path))
        ):
            raise SessionStateError("M3 report does not identify the configured raw input root")
        if not isinstance(manifest, dict) or not _same_location(manifest.get("path"), store.backend.uri(m2_key)):
            raise SessionStateError("M3 report does not reference the current M2 manifest")
        return

    sessions = payload.get("sessions")
    record = next(
        (
            item
            for item in sessions
            if isinstance(item, dict)
            and item.get("year") == scope.year
            and item.get("session_key") == scope.session_key
        ),
        None,
    ) if isinstance(sessions, list) else None
    if not record or record.get("error") or record.get("blocking_failures"):
        raise SessionStateError(f"{stage.upper()} report does not show successful processing")
    if stage == "m4" and record.get("output_path") is None:
        raise SessionStateError("M4 report has no consolidated artifact path")
    if stage == "m5" and record.get("output_path") is None:
        raise SessionStateError("M5 report has no feature artifact path")


def _config_fingerprint(settings: AppSettings, store: DataStore) -> str:
    config = {
        # ``years`` only selects which seasons a run covers (a backfill phase sets it
        # through F1_EXTRACTION_YEARS); it does not change how a session is processed,
        # so it must not separate the state recorded by one phase from another's.
        "historical_extraction": settings.historical_extraction.model_dump(mode="json", exclude={"overwrite", "years"}),
        "validation": settings.validation.model_dump(mode="json"),
        "consolidation": settings.consolidation.model_dump(mode="json", exclude={"overwrite", "years"}),
        "features": settings.features.model_dump(mode="json", exclude={"overwrite", "years"}),
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_artifact(store: DataStore, key: str) -> Dict[str, Any]:
    try:
        data = store.backend.read_bytes(key)
    except OSError as exc:
        raise SessionStateError(f"{key}: artifact is missing or unreadable: {exc}") from exc
    return {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _state_directory(store: DataStore, settings: AppSettings, scope: SessionScope) -> str:
    # Legacy layout: next to the configured extraction manifests (inside the
    # store root, never the process working directory by accident). Lake layout:
    # the fixed manifests/incremental prefix. The configuration fingerprint keeps
    # incompatible configurations apart; the store root itself separates stores.
    configured = join_key(settings.historical_extraction.manifest_path, "incremental")
    root = store.root(Layer.INCREMENTAL_STATE, configured)
    return f"{root}/config={_config_fingerprint(settings, store)}/year={scope.year}/session={scope.session_key}"


def _state_key(store: DataStore, settings: AppSettings, scope: SessionScope) -> str:
    return f"{_state_directory(store, settings, scope)}/state.json"


def _stage_state_key(store: DataStore, settings: AppSettings, scope: SessionScope) -> str:
    return f"{_state_directory(store, settings, scope)}/stages.json"


def history_completeness(
    store: DataStore,
    settings: AppSettings,
    required_years: List[int],
) -> Dict[str, Any]:
    """Prove, from stored evidence only, whether every required season is complete.

    A season is complete when its cached meetings/sessions catalogue lists at
    least one configured session and every one of those sessions passes
    :func:`session_reuse_decision` (hash-verified raw, M2 manifest, M3-M5 reports
    and outputs, current configuration). Anything missing, unreadable or altered
    makes the season incomplete; nothing is taken from a flag or a summary.
    """
    from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline

    seasons: Dict[str, Any] = {}
    with HistoricalExtractionPipeline(settings, client=_NoClient(), store=store) as pipeline:
        for year in sorted(set(required_years)):
            units, catalog_reason = pipeline.cached_work_units(year)
            if catalog_reason is not None:
                seasons[str(year)] = {"status": "incomplete", "reason": catalog_reason, "sessions": 0, "pending": []}
                continue
            pending = []
            for unit in units:
                scope = SessionScope(year=year, meeting_key=unit["meeting_key"], session_key=unit["session_key"])
                ok, reason = session_reuse_decision(store, settings, scope)
                if not ok:
                    pending.append({"session_key": unit["session_key"], "reason": reason})
            seasons[str(year)] = {
                "status": "complete" if not pending else "incomplete",
                "reason": None if not pending else f"{len(pending)} session(s) without verified M2-M5 evidence",
                "sessions": len(units),
                "pending": pending,
            }
    complete = bool(seasons) and all(item["status"] == "complete" for item in seasons.values())
    return {"complete": complete, "required_years": sorted(set(required_years)), "seasons": seasons}


class _NoClient:
    """Stand-in client: evidence checks must never reach OpenF1."""

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"history verification must not call OpenF1 ({name})")
