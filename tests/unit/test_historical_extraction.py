"""Unit tests for the historical extraction pipeline.

These never touch the network: ``F1Client`` is the seam the pipeline
depends on, so it is mocked directly (with ``spec=`` so the mock stays
honest about which methods actually exist). Storage is real, against
``tmp_path``, because file layout and idempotency are exactly what these
tests are about.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.ingestion.exceptions import OpenF1ConnectionError, OpenF1HTTPError
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.pipelines.manifest import ExtractionStatus

MEETINGS = [
    {"meeting_key": 1219, "meeting_name": "Bahrain Grand Prix", "year": 2023},
    {"meeting_key": 1220, "meeting_name": "Saudi Arabian Grand Prix", "year": 2023},
]

SESSIONS = [
    {"session_key": 9158, "session_name": "Race", "session_type": "Race"},
    {"session_key": 9157, "session_name": "Qualifying", "session_type": "Qualifying"},
    {"session_key": 9156, "session_name": "Practice 3", "session_type": "Practice"},
]


def make_settings(tmp_path: Path, **overrides: Any) -> AppSettings:
    config: Dict[str, Any] = {
        "years": [2023],
        "session_types": ["Race"],
        "endpoints": ["laps"],
        "output_path": str(tmp_path / "raw"),
        "manifest_path": str(tmp_path / "manifests"),
        "overwrite": False,
    }
    config.update(overrides)
    return AppSettings(historical_extraction=config)


def make_client(**overrides: Any) -> MagicMock:
    client = MagicMock(spec=F1Client)
    client.get_meetings.return_value = [MEETINGS[0]]
    client.get_sessions.return_value = SESSIONS
    client.get_drivers.return_value = [{"driver_number": 1}, {"driver_number": 44}]
    client.get_laps.return_value = [{"lap_number": 1}, {"lap_number": 2}]
    client.get_weather.return_value = [{"air_temperature": 20}]
    client.get_car_data.return_value = [{"speed": 301}]
    client.get_positions.return_value = []
    client.get_intervals.return_value = []
    client.get_stints.return_value = []
    client.get_pit.return_value = []
    client.get_race_control.return_value = []

    for name, value in overrides.items():
        getattr(client, name).return_value = value
    return client


def raw_path(tmp_path: Path, *parts: str) -> Path:
    return tmp_path.joinpath("raw", *parts)


def statuses_for(manifest, endpoint: str) -> List[ExtractionStatus]:
    return [entry.status for entry in manifest.entries if entry.endpoint == endpoint]


# -- discovery ------------------------------------------------------------


def test_discover_meetings_queries_client_and_saves_raw_response(tmp_path: Path) -> None:
    client = make_client()
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path), client=client, storage=None)

    manifest = pipeline.run()

    client.get_meetings.assert_called_once_with(year=2023)
    saved = raw_path(tmp_path, "meetings", "year=2023", "meetings.json")
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["data"] == [MEETINGS[0]]
    assert manifest.meetings_processed == [1219]


def test_discover_sessions_queries_client_per_meeting_and_saves_raw_response(tmp_path: Path) -> None:
    client = make_client(get_meetings=MEETINGS)
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path), client=client)

    manifest = pipeline.run()

    assert client.get_sessions.call_count == 2
    client.get_sessions.assert_any_call(year=2023, meeting_key=1219)
    client.get_sessions.assert_any_call(year=2023, meeting_key=1220)
    assert raw_path(tmp_path, "sessions", "year=2023", "meeting_key=1219", "sessions.json").exists()
    assert manifest.meetings_processed == [1219, 1220]


def test_meetings_are_processed_in_deterministic_key_order(tmp_path: Path) -> None:
    client = make_client(get_meetings=list(reversed(MEETINGS)))
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path), client=client)

    manifest = pipeline.run()

    assert manifest.meetings_processed == [1219, 1220]


# -- session filtering ----------------------------------------------------


def test_filter_sessions_keeps_only_configured_session_names(tmp_path: Path) -> None:
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path), client=make_client())

    kept = pipeline.filter_sessions(SESSIONS)

    assert [session["session_key"] for session in kept] == [9158]


def test_filter_sessions_also_matches_the_coarser_session_type(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, session_types=["practice"])
    pipeline = HistoricalExtractionPipeline(settings, client=make_client())

    kept = pipeline.filter_sessions(SESSIONS)

    assert [session["session_key"] for session in kept] == [9156]


def test_filter_sessions_without_configured_types_keeps_everything(tmp_path: Path) -> None:
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path, session_types=[]), client=make_client())

    assert len(pipeline.filter_sessions(SESSIONS)) == 3


# -- extraction and storage ----------------------------------------------


def test_run_extracts_configured_endpoints_only_for_matching_sessions(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["laps", "weather"])
    client = make_client()
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    client.get_laps.assert_called_once_with(session_key=9158)
    client.get_weather.assert_called_once_with(session_key=9158)
    partition = ("year=2023", "meeting_key=1219", "session_key=9158")
    assert raw_path(tmp_path, "laps", *partition, "session_9158.json").exists()
    assert raw_path(tmp_path, "weather", *partition, "session_9158.json").exists()
    assert manifest.sessions_processed == [9158]


def test_saved_raw_file_keeps_the_response_and_its_request_context(tmp_path: Path) -> None:
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path), client=make_client())

    pipeline.run()

    saved = raw_path(tmp_path, "laps", "year=2023", "meeting_key=1219", "session_key=9158", "session_9158.json")
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["endpoint"] == "laps"
    assert payload["parameters"] == {"year": 2023, "meeting_key": 1219, "session_key": 9158}
    assert payload["data"] == [{"lap_number": 1}, {"lap_number": 2}]
    assert "retrieved_at" in payload


def test_car_data_is_extracted_once_per_driver(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["car_data"])
    client = make_client()
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    client.get_drivers.assert_called_once_with(session_key=9158)
    client.get_car_data.assert_any_call(session_key=9158, driver_number=1)
    client.get_car_data.assert_any_call(session_key=9158, driver_number=44)
    session_dir = raw_path(tmp_path, "car_data", "year=2023", "meeting_key=1219", "session_key=9158")
    assert (session_dir / "driver_1.json").exists()
    assert (session_dir / "driver_44.json").exists()
    assert [entry.driver_number for entry in manifest.entries if entry.endpoint == "car_data"] == [1, 44]


def test_car_data_reuses_the_saved_drivers_file_instead_of_querying_again(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["drivers", "car_data"])
    client = make_client()

    HistoricalExtractionPipeline(settings, client=client).run()

    # drivers was fetched once as an endpoint; car_data planning read it off disk.
    client.get_drivers.assert_called_once_with(session_key=9158)


# -- idempotency ----------------------------------------------------------


def test_second_run_skips_existing_files_when_overwrite_is_false(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["laps", "weather"])
    HistoricalExtractionPipeline(settings, client=make_client()).run()

    second_client = make_client()
    manifest = HistoricalExtractionPipeline(settings, client=second_client).run()

    second_client.get_meetings.assert_not_called()
    second_client.get_sessions.assert_not_called()
    second_client.get_laps.assert_not_called()
    second_client.get_weather.assert_not_called()
    assert {entry.status for entry in manifest.entries} == {ExtractionStatus.SKIPPED}
    assert manifest.summary()["downloaded"] == 0
    assert manifest.summary()["skipped"] == 4  # meetings + sessions + laps + weather


def test_second_run_redownloads_everything_when_overwrite_is_true(tmp_path: Path) -> None:
    first_settings = make_settings(tmp_path)
    HistoricalExtractionPipeline(first_settings, client=make_client()).run()

    second_client = make_client()
    overwriting = make_settings(tmp_path, overwrite=True)
    manifest = HistoricalExtractionPipeline(overwriting, client=second_client).run()

    second_client.get_meetings.assert_called_once_with(year=2023)
    second_client.get_laps.assert_called_once_with(session_key=9158)
    assert statuses_for(manifest, "laps") == [ExtractionStatus.DOWNLOADED]
    assert manifest.summary()["skipped"] == 0


def test_rerun_does_not_create_duplicate_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    HistoricalExtractionPipeline(settings, client=make_client()).run()
    HistoricalExtractionPipeline(make_settings(tmp_path, overwrite=True), client=make_client()).run()

    session_dir = raw_path(tmp_path, "laps", "year=2023", "meeting_key=1219", "session_key=9158")
    assert [path.name for path in session_dir.iterdir()] == ["session_9158.json"]


def test_unreadable_existing_file_is_refetched_instead_of_breaking_the_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    HistoricalExtractionPipeline(settings, client=make_client()).run()
    corrupted = raw_path(tmp_path, "meetings", "year=2023", "meetings.json")
    corrupted.write_text("{not json", encoding="utf-8")

    second_client = make_client()
    manifest = HistoricalExtractionPipeline(settings, client=second_client).run()

    second_client.get_meetings.assert_called_once_with(year=2023)
    assert statuses_for(manifest, "meetings") == [ExtractionStatus.DOWNLOADED]


# -- manifest -------------------------------------------------------------


def test_manifest_is_written_to_disk_and_describes_the_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["laps", "weather"])
    pipeline = HistoricalExtractionPipeline(settings, client=make_client())

    manifest = pipeline.run()

    manifest_files = list((tmp_path / "manifests").glob("manifest_*.json"))
    assert len(manifest_files) == 1

    payload = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert payload["pipeline_name"] == "historical_extraction"
    assert payload["pipeline_version"] == manifest.pipeline_version
    assert payload["config"]["years"] == [2023]
    assert payload["config"]["session_types"] == ["Race"]
    assert payload["config"]["endpoints"] == ["laps", "weather"]
    assert payload["config"]["overwrite"] is False
    assert payload["summary"]["downloaded"] == 4
    assert payload["meetings_processed"] == [1219]
    assert payload["sessions_processed"] == [9158]
    assert payload["duration_seconds"] is not None
    assert payload["errors"] == []

    laps_entry = next(entry for entry in payload["entries"] if entry["endpoint"] == "laps")
    assert laps_entry["status"] == "downloaded"
    assert laps_entry["record_count"] == 2
    assert laps_entry["session_key"] == 9158
    assert laps_entry["file_path"].endswith("session_9158.json")


def test_each_run_writes_its_own_manifest(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    HistoricalExtractionPipeline(settings, client=make_client()).run()
    HistoricalExtractionPipeline(settings, client=make_client()).run()

    assert len(list((tmp_path / "manifests").glob("manifest_*.json"))) == 2


# -- error handling -------------------------------------------------------


def test_endpoint_failure_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["weather", "laps"])
    client = make_client()
    client.get_weather.side_effect = OpenF1HTTPError(status_code=500, message="boom", endpoint="weather")
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    client.get_laps.assert_called_once_with(session_key=9158)
    assert raw_path(tmp_path, "laps", "year=2023", "meeting_key=1219", "session_key=9158", "session_9158.json").exists()

    failure = next(entry for entry in manifest.errors)
    assert failure.endpoint == "weather"
    assert failure.session_key == 9158
    assert failure.error_type == "OpenF1HTTPError"
    assert manifest.summary()["failed"] == 1


def test_failing_endpoint_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["weather"])
    client = make_client()
    client.get_weather.side_effect = OpenF1ConnectionError("network down")

    HistoricalExtractionPipeline(settings, client=client).run()

    assert not raw_path(tmp_path, "weather").exists()


def test_failure_in_one_session_does_not_stop_the_next_session(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, session_types=["Race", "Qualifying"])
    client = make_client()
    # Qualifying (9157) is processed first because sessions are sorted by key.
    client.get_laps.side_effect = [OpenF1HTTPError(status_code=503, message="boom"), [{"lap_number": 1}]]
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    assert statuses_for(manifest, "laps") == [ExtractionStatus.FAILED, ExtractionStatus.DOWNLOADED]
    assert raw_path(tmp_path, "laps", "year=2023", "meeting_key=1219", "session_key=9158", "session_9158.json").exists()


def test_meeting_discovery_failure_skips_that_year_but_continues(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, years=[2023, 2024])
    client = make_client()

    def meetings_by_year(*, year: int) -> List[Dict[str, Any]]:
        if year == 2023:
            raise OpenF1HTTPError(status_code=502, message="boom", endpoint="meetings")
        return [MEETINGS[0]]

    client.get_meetings.side_effect = meetings_by_year
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    assert statuses_for(manifest, "meetings") == [ExtractionStatus.FAILED, ExtractionStatus.DOWNLOADED]
    assert client.get_sessions.call_count == 1
    assert manifest.summary()["failed"] == 1
    assert manifest.sessions_processed == [9158]


def test_session_discovery_failure_skips_that_meeting_but_continues(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = make_client(get_meetings=MEETINGS)
    client.get_sessions.side_effect = [
        OpenF1HTTPError(status_code=500, message="boom", endpoint="sessions"),
        SESSIONS,
    ]
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    assert statuses_for(manifest, "sessions") == [ExtractionStatus.FAILED, ExtractionStatus.DOWNLOADED]
    assert client.get_laps.call_count == 1


def test_car_data_planning_failure_is_recorded_as_an_endpoint_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["car_data"])
    client = make_client()
    client.get_drivers.side_effect = OpenF1ConnectionError("network down")
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    client.get_car_data.assert_not_called()
    failure = manifest.errors[0]
    assert failure.endpoint == "car_data"
    assert failure.session_key == 9158


def test_unbounded_query_guard_is_recorded_without_stopping_the_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["car_data", "laps"])
    client = make_client()
    client.get_car_data.side_effect = ValueError("get_car_data requires 'driver_number'")
    pipeline = HistoricalExtractionPipeline(settings, client=client)

    manifest = pipeline.run()

    assert manifest.summary()["failed"] == 2  # one per driver
    assert statuses_for(manifest, "laps") == [ExtractionStatus.DOWNLOADED]


# -- configuration --------------------------------------------------------


def test_unknown_endpoint_fails_before_any_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, endpoints=["laps", "telemetry"])
    client = make_client()

    with pytest.raises(ValueError, match="telemetry"):
        HistoricalExtractionPipeline(settings, client=client)

    client.get_meetings.assert_not_called()


def test_empty_year_list_produces_an_empty_but_valid_manifest(tmp_path: Path) -> None:
    pipeline = HistoricalExtractionPipeline(make_settings(tmp_path, years=[]), client=make_client())

    manifest = pipeline.run()

    assert manifest.entries == []
    assert manifest.summary()["downloaded"] == 0
    assert manifest.finished_at is not None


def test_pipeline_does_not_close_a_client_it_did_not_create(tmp_path: Path) -> None:
    client = make_client()

    with HistoricalExtractionPipeline(make_settings(tmp_path), client=client):
        pass

    client.close.assert_not_called()
