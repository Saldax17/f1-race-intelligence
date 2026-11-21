"""Synthetic coverage for incremental per-session planning and provenance."""

from unittest.mock import MagicMock
from datetime import datetime, timezone

import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.ingestion.exceptions import OpenF1HTTPError
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.pipelines.incremental import (
    _stage_state_key,
    _state_key,
    capture_stage_inputs,
    record_session_completion,
    record_stage_provenance,
)
from f1_race_intelligence.pipelines.manifest import ExecutionManifest, ExtractionStatus, ManifestEntry
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from f1_race_intelligence.storage.store import DataStore

YEAR = 2023
MEETING = 1141
SESSION_A = 7953
SESSION_B = 7954


def settings() -> AppSettings:
    return AppSettings(
        historical_extraction={
            "years": [YEAR],
            "session_types": ["Race"],
            "endpoints": ["laps"],
            "output_path": "raw",
            "manifest_path": "manifests/extraction",
        },
        validation={"raw_path": "raw", "manifest_path": "manifests/extraction", "report_path": "validation"},
        consolidation={
            "raw_path": "raw",
            "output_path": "processed",
            "report_path": "consolidation",
            "validation_path": "validation",
        },
        features={"input_path": "processed", "output_path": "features", "report_path": "feature_reports"},
    )


def add_catalogs(store: DataStore, sessions=(SESSION_A,)) -> None:
    raw = RawDataStorage("raw", store=store)
    raw.save("meetings", {"year": YEAR}, [{"meeting_key": MEETING, "year": YEAR}], file_name="meetings.json")
    raw.save(
        "sessions",
        {"year": YEAR, "meeting_key": MEETING},
        [{"session_key": key, "session_name": "Race", "session_type": "Race"} for key in sessions],
        file_name="sessions.json",
    )


def add_completed_session(
    store: DataStore,
    config: AppSettings,
    session_key: int,
    *,
    m2_404_endpoint: str = "",
) -> SessionScope:
    scope = SessionScope(year=YEAR, meeting_key=MEETING, session_key=session_key)
    raw = RawDataStorage("raw", store=store)
    parameters = {"year": YEAR, "meeting_key": MEETING, "session_key": session_key}
    manifest_root = store.root(Layer.EXTRACTION_MANIFESTS, config.historical_extraction.manifest_path)
    if m2_404_endpoint:
        config.historical_extraction.endpoints = ["laps", m2_404_endpoint]
        client = MagicMock(spec=F1Client)
        client.get_laps.return_value = [{"lap_number": 1, "lap_duration": 90.0}]
        getattr(client, {"position": "get_positions"}[m2_404_endpoint]).side_effect = OpenF1HTTPError(
            404, "no data", endpoint=m2_404_endpoint
        )
        with HistoricalExtractionPipeline(config, client=client, store=store) as pipeline:
            manifest = pipeline.run_session(YEAR, session_key, meeting_key=MEETING)
        assert any(
            entry.endpoint == m2_404_endpoint and entry.status_code == 404 for entry in manifest.errors
        )
    else:
        path = raw.save(
            "laps",
            parameters,
            [{"lap_number": 1, "lap_duration": 90.0}],
            file_name=f"session_{session_key}.json",
        )
        manifest = ExecutionManifest(
            pipeline_name="historical_extraction",
            pipeline_version="1.0.0",
            config={},
            entries=[
                ManifestEntry(
                    endpoint="laps",
                    status=ExtractionStatus.DOWNLOADED,
                    year=YEAR,
                    meeting_key=MEETING,
                    session_key=session_key,
                    file_path=str(path),
                )
            ],
            sessions_processed=[session_key],
        )
        manifest.finished_at = datetime.now(timezone.utc)
        manifest.save(manifest_root, backend=store.backend)
    m2_key = store.backend.list(manifest_root)[-1]

    validation_root = store.root(Layer.VALIDATION_REPORTS, config.validation.report_path)
    store.backend.write_json(
        f"{validation_root}/validation_report_{session_key}.json",
        {
            "pipeline_name": "data_validation",
            "global_status": "PASS",
            "source": {
                "filter": scope.to_dict(),
                "raw_path": store.backend.uri(store.root(Layer.RAW, config.validation.raw_path)),
                "manifest": {"path": store.backend.uri(m2_key)},
            },
        },
    )

    for layer, configured, report_layer, report_path, pipeline, report in (
        (
            Layer.CONSOLIDATED,
            config.consolidation.output_path,
            Layer.CONSOLIDATION_REPORTS,
            config.consolidation.report_path,
            "data_consolidation",
            {
                "session_key": session_key,
                "year": YEAR,
                "output_path": store.backend.uri(
                    store.layout.session_dataset_key(store.root(Layer.CONSOLIDATED, config.consolidation.output_path), YEAR, session_key)
                ),
            },
        ),
        (
            Layer.FEATURES,
            config.features.output_path,
            Layer.FEATURE_REPORTS,
            config.features.report_path,
            "feature_engineering",
            {
                "session_key": session_key,
                "year": YEAR,
                "output_path": store.backend.uri(
                    store.layout.session_dataset_key(store.root(Layer.FEATURES, config.features.output_path), YEAR, session_key)
                ),
            },
        ),
    ):
        root = store.root(layer, configured)
        key = store.layout.session_dataset_key(root, YEAR, session_key)
        store.backend.write_parquet(key, pd.DataFrame({"value": [1.0]}))
        report_root = store.root(report_layer, report_path)
        store.backend.write_json(
            f"{report_root}/{pipeline}_{session_key}.json",
            {
                "pipeline_name": pipeline,
                "source_validation": {"report": store.backend.uri(f"{validation_root}/validation_report_{session_key}.json")}
                if pipeline == "data_consolidation"
                else {},
                "config": (
                    {"raw_path": config.consolidation.raw_path, "output_path": config.consolidation.output_path}
                    if pipeline == "data_consolidation"
                    else {"input_path": config.features.input_path}
                    if pipeline == "feature_engineering"
                    else {}
                ),
                "sessions": [report],
            },
        )
    for stage in ("m3", "m4", "m5"):
        assert record_stage_provenance(store, config, scope, stage)[0]
    return scope


def plan(store, config):
    client = MagicMock(spec=F1Client)
    pipeline = HistoricalExtractionPipeline(config, client=client, store=store)
    units = pipeline.plan()
    return units, pipeline.plan_decisions, client


def test_first_plan_includes_every_pending_session_without_refreshing_cached_catalogs(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A, SESSION_B))

    units, decisions, client = plan(store, config)

    assert [unit["session_key"] for unit in units] == [SESSION_A, SESSION_B]
    assert all(unit["process_reason"] == "no verified completion provenance" for unit in units)
    assert [item["decision"] for item in decisions] == ["include", "include"]
    client.get_meetings.assert_not_called()
    client.get_sessions.assert_not_called()


def test_second_plan_omits_only_sessions_with_complete_hash_verified_provenance(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    assert record_session_completion(store, config, scope)[0]

    units, decisions, _ = plan(store, config)

    assert units == []
    assert decisions == [
        {
            "year": YEAR,
            "meeting_key": MEETING,
            "session_key": SESSION_A,
            "session_name": "Race",
            "decision": "omit",
            "reason": "verified_existing_artifacts",
        }
    ]


def test_new_catalog_session_is_added_without_reprocessing_verified_sessions(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    assert record_session_completion(store, config, scope)[0]
    RawDataStorage("raw", store=store).save(
        "sessions",
        {"year": YEAR, "meeting_key": MEETING},
        [
            {"session_key": SESSION_A, "session_name": "Race", "session_type": "Race"},
            {"session_key": SESSION_B, "session_name": "Race", "session_type": "Race"},
        ],
        file_name="sessions.json",
    )

    units, decisions, client = plan(store, config)

    assert [unit["session_key"] for unit in units] == [SESSION_B]
    assert [item["decision"] for item in decisions] == ["omit", "include"]
    client.get_sessions.assert_not_called()


def test_changed_raw_input_reprocesses_only_its_session(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A, SESSION_B))
    for session_key in (SESSION_A, SESSION_B):
        scope = add_completed_session(store, config, session_key)
        assert record_session_completion(store, config, scope)[0]

    key = store.layout.raw_key(
        store.root(Layer.RAW, "raw"),
        "laps",
        {"year": YEAR, "meeting_key": MEETING, "session_key": SESSION_A},
        f"session_{SESSION_A}.json",
    )
    payload = store.backend.read_json(key)
    payload["data"][0]["lap_duration"] = 91.0
    store.backend.write_json(key, payload)

    units, decisions, _ = plan(store, config)

    assert [unit["session_key"] for unit in units] == [SESSION_A]
    assert decisions[0]["reason"] == "raw input artifacts changed"
    assert decisions[1]["decision"] == "omit"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_derived_artifacts_are_not_considered_complete(tmp_path, damage):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    assert record_session_completion(store, config, scope)[0]
    feature_key = store.layout.session_dataset_key(store.root(Layer.FEATURES, "features"), YEAR, SESSION_A)
    if damage == "missing":
        store.backend.local_path(feature_key).unlink()
    else:
        store.backend.write_bytes(feature_key, b"not parquet")

    units, decisions, _ = plan(store, config)

    assert [unit["session_key"] for unit in units] == [SESSION_A]
    assert decisions[0]["reason"].startswith("derived artifacts are missing or unverifiable:")


def test_unknown_provenance_never_causes_a_session_to_be_omitted(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    raw = RawDataStorage("raw", store=store)
    raw.save(
        "laps",
        {"year": YEAR, "meeting_key": MEETING, "session_key": SESSION_A},
        [{"lap_number": 1, "lap_duration": 90.0}],
        file_name=f"session_{SESSION_A}.json",
    )

    units, decisions, _ = plan(store, config)

    assert [unit["session_key"] for unit in units] == [SESSION_A]
    assert decisions[0]["reason"] == "no verified completion provenance"


def test_state_uses_independent_store_roots_and_configuration_fingerprints(tmp_path):
    root_a, root_b = tmp_path / "lake-a", tmp_path / "lake-b"
    store_a, store_b = DataStore.local(str(root_a)), DataStore.local(str(root_b))
    config_a, config_b = settings(), settings()
    config_b.features.output_path = "alternate-features"
    scope = SessionScope(year=YEAR, meeting_key=MEETING, session_key=SESSION_A)

    state_a = _state_key(store_a, config_a, scope)
    state_b = _state_key(store_b, config_a, scope)
    state_other_config = _state_key(store_a, config_b, scope)
    stages_a = _stage_state_key(store_a, config_a, scope)

    # The logical key carries no physical path; the store root separates stores.
    assert state_a == state_b
    assert state_a != state_other_config
    assert store_a.backend.local_path(state_a).is_relative_to(root_a)
    assert store_b.backend.local_path(state_b).is_relative_to(root_b)
    assert store_a.backend.local_path(state_a) != store_b.backend.local_path(state_b)
    assert store_a.backend.local_path(stages_a).is_relative_to(root_a)


def test_manual_m5_after_raw_change_cannot_refresh_completion_without_m3_m4(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    assert record_session_completion(store, config, scope)[0]

    key = store.layout.raw_key(
        store.root(Layer.RAW, "raw"),
        "laps",
        {"year": YEAR, "meeting_key": MEETING, "session_key": SESSION_A},
        f"session_{SESSION_A}.json",
    )
    payload = store.backend.read_json(key)
    payload["data"][0]["lap_duration"] = 92.0
    store.backend.write_json(key, payload)

    recorded, reason = record_stage_provenance(store, config, scope, "m5")
    assert not recorded
    assert "M4 provenance does not match current inputs" in reason
    assert not record_session_completion(store, config, scope)[0]

    units, _, _ = plan(store, config)
    assert [unit["session_key"] for unit in units] == [SESSION_A]


@pytest.mark.parametrize(
    ("stage", "layer", "report_path", "report_name", "tamper"),
    [
        ("m4", Layer.CONSOLIDATION_REPORTS, "consolidation", "data_consolidation", "output_path"),
        ("m5", Layer.FEATURE_REPORTS, "feature_reports", "feature_engineering", "input_path"),
    ],
)
def test_m4_m5_require_expected_input_and_output_links(
    tmp_path, stage, layer, report_path, report_name, tamper
):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    root = store.root(layer, report_path)
    key = f"{root}/{report_name}_{SESSION_A}.json"
    payload = store.backend.read_json(key)
    if stage == "m4":
        payload["sessions"][0]["output_path"] = "wrong-output.parquet"
    else:
        payload["config"]["input_path"] = "wrong-input"
    store.backend.write_json(key, payload)

    recorded, reason = record_stage_provenance(store, config, scope, stage)

    assert not recorded
    assert "expected output" in reason if tamper == "output_path" else "input path" in reason
    assert not record_session_completion(store, config, scope)[0]


def test_stage_snapshot_rejects_raw_change_during_execution(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A)
    snapshot, reason = capture_stage_inputs(store, config, scope, "m3")
    assert snapshot is not None, reason

    key = store.layout.raw_key(
        store.root(Layer.RAW, "raw"),
        "laps",
        {"year": YEAR, "meeting_key": MEETING, "session_key": SESSION_A},
        f"session_{SESSION_A}.json",
    )
    payload = store.backend.read_json(key)
    payload["data"][0]["lap_duration"] = 93.0
    store.backend.write_json(key, payload)

    recorded, reason = record_stage_provenance(
        store, config, scope, "m3", expected_inputs=snapshot
    )
    assert not recorded
    assert "inputs changed while the stage was running" in reason


def test_real_m2_404_is_explicitly_tracked_and_only_reused_while_evidence_matches(tmp_path):
    store, config = DataStore.local(str(tmp_path)), settings()
    add_catalogs(store, (SESSION_A,))
    scope = add_completed_session(store, config, SESSION_A, m2_404_endpoint="position")
    manifest_root = store.root(Layer.EXTRACTION_MANIFESTS, config.historical_extraction.manifest_path)
    m2_key = store.backend.list(manifest_root)[-1]
    manifest = store.backend.read_json(m2_key)
    missing_entry = next(
        entry for entry in manifest["entries"] if entry["endpoint"] == "position" and entry.get("status_code") == 404
    )
    assert missing_entry["status"] == "failed"
    assert missing_entry["error_type"] == "OpenF1HTTPError"

    assert record_session_completion(store, config, scope)[0]
    state = store.backend.read_json(_state_key(store, config, scope))
    absence = state["absent_endpoints"][0]
    assert (absence["endpoint"], absence["status_code"], absence["retryable"]) == ("position", 404, False)
    assert state["reports"][0]["stage"] == "m2"
    assert plan(store, config)[0] == []

    manifest["entries"] = [
        {**entry, "status_code": 503} if entry.get("endpoint") == "position" else entry
        for entry in manifest["entries"]
    ]
    store.backend.write_json(m2_key, manifest)
    units, decisions, _ = plan(store, config)
    assert [unit["session_key"] for unit in units] == [SESSION_A]
    assert decisions[0]["decision"] == "include"
