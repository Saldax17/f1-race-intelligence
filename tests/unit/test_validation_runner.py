"""End-to-end tests: a raw tree on disk in, a persisted report out."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from f1_race_intelligence.pipelines.manifest import ExecutionManifest, ExtractionStatus, ManifestEntry
from f1_race_intelligence.validation.models import GlobalStatus, RuleStatus, Severity
from f1_race_intelligence.validation.runner import ValidationRunner
from factories import MEETING_KEY, SESSION_KEY, car_data_record, driver_record, lap_record


def results_for(report, rule: str, severity: Severity = None) -> List[Any]:
    return [
        item
        for item in report.results
        if item.rule == rule and item.status is RuleStatus.FAIL and (severity is None or item.severity is severity)
    ]


def read_report(directory: Path) -> Dict[str, Any]:
    files = list(directory.glob("validation_report_*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


# -- happy path -----------------------------------------------------------


def test_a_consistent_dataset_passes(valid_dataset, validation_settings) -> None:
    valid_dataset()

    report = ValidationRunner(validation_settings()).run()

    assert report.global_status is GlobalStatus.PASS
    assert report.errors == []
    assert report.warnings == []
    assert report.scope["files_evaluated"] == 4
    assert report.scope["endpoints"] == ["drivers", "laps", "meetings", "sessions"]


def test_rules_that_found_nothing_are_still_recorded(valid_dataset, validation_settings) -> None:
    valid_dataset()

    report = ValidationRunner(validation_settings()).run()

    clean = {(item.endpoint, item.rule) for item in report.results if item.status is RuleStatus.PASS}
    assert ("laps", "types") in clean
    assert ("laps", "duplicates") in clean
    assert (None, "identifiers") in clean


def test_the_report_is_written_to_disk(valid_dataset, validation_settings, tmp_path: Path) -> None:
    valid_dataset()

    report = ValidationRunner(validation_settings()).run()

    payload = read_report(tmp_path / "validation")
    assert payload["pipeline_name"] == "data_validation"
    assert payload["global_status"] == "PASS"
    assert payload["summary"]["errors"] == 0
    assert payload["scope"]["files_evaluated"] == 4
    assert payload["config"]["fail_on_error"] is True
    assert payload["sampling"]["exhaustive"] is True
    assert payload["duration_seconds"] is not None
    assert report.global_status.value == payload["global_status"]


def test_a_csv_is_written_when_asked_for(valid_dataset, validation_settings, tmp_path: Path) -> None:
    valid_dataset()

    ValidationRunner(validation_settings(write_csv=True)).run()

    csv_files = list((tmp_path / "validation").glob("validation_report_*.csv"))
    assert len(csv_files) == 1
    assert "rule,severity,status" in csv_files[0].read_text(encoding="utf-8")


def test_raw_data_is_never_modified(valid_dataset, validation_settings, raw_root: Path) -> None:
    valid_dataset()
    before = {path: path.read_bytes() for path in sorted(raw_root.rglob("*.json"))}

    ValidationRunner(validation_settings()).run()

    after = {path: path.read_bytes() for path in sorted(raw_root.rglob("*.json"))}
    assert after == before


# -- global status --------------------------------------------------------


def test_a_warning_downgrades_the_run_to_warning(valid_dataset, validation_settings) -> None:
    valid_dataset(laps=[lap_record(1, 1, lap_duration=900.0)])

    report = ValidationRunner(validation_settings()).run()

    assert report.global_status is GlobalStatus.WARNING
    assert report.errors == []
    assert results_for(report, "values", Severity.WARNING)


def test_an_error_fails_the_run(valid_dataset, validation_settings) -> None:
    valid_dataset(laps=[lap_record(1, 1), lap_record(1, 1)])

    report = ValidationRunner(validation_settings()).run()

    assert report.global_status is GlobalStatus.FAIL
    assert results_for(report, "duplicates", Severity.ERROR)


def test_fail_on_error_false_downgrades_errors_to_a_warning(valid_dataset, validation_settings) -> None:
    valid_dataset(laps=[lap_record(1, 1), lap_record(1, 1)])

    report = ValidationRunner(validation_settings(fail_on_error=False)).run()

    assert report.global_status is GlobalStatus.WARNING
    assert report.errors, "the error is still reported, it just does not block"


def test_info_findings_alone_keep_the_run_passing(valid_dataset, validation_settings, write_raw) -> None:
    valid_dataset()
    write_raw("car_data", [car_data_record(1, 1, throttle=104)], meeting_key=MEETING_KEY,
              session_key=SESSION_KEY, file_name="driver_1.json")

    report = ValidationRunner(validation_settings()).run()

    assert results_for(report, "values", Severity.INFO)
    assert report.global_status is GlobalStatus.PASS


# -- resilience -----------------------------------------------------------


def test_an_unreadable_file_is_reported_and_the_run_continues(
    valid_dataset, validation_settings, write_raw, raw_root: Path
) -> None:
    valid_dataset()
    broken = write_raw("weather", [], meeting_key=MEETING_KEY, session_key=SESSION_KEY)
    broken.write_text("{not json", encoding="utf-8")

    report = ValidationRunner(validation_settings()).run()

    structure_errors = results_for(report, "structure", Severity.ERROR)
    assert len(structure_errors) == 1
    assert structure_errors[0].endpoint == "weather"
    # The rest of the tree was still validated.
    assert report.scope["files_evaluated"] == 5
    assert any(item.endpoint == "laps" and item.status is RuleStatus.PASS for item in report.results)


def test_an_empty_file_is_reported_as_an_error(valid_dataset, validation_settings, write_raw) -> None:
    valid_dataset()
    empty = write_raw("weather", [], meeting_key=MEETING_KEY, session_key=SESSION_KEY)
    empty.write_text("", encoding="utf-8")

    report = ValidationRunner(validation_settings()).run()

    assert results_for(report, "structure", Severity.ERROR)
    assert report.global_status is GlobalStatus.FAIL


def test_an_empty_raw_tree_is_a_warning_not_a_pass(validation_settings) -> None:
    report = ValidationRunner(validation_settings()).run()

    assert report.global_status is GlobalStatus.WARNING
    assert results_for(report, "coverage", Severity.WARNING)
    assert report.scope["files_evaluated"] == 0


def test_multiple_files_across_endpoints_are_all_processed(
    valid_dataset, validation_settings, write_raw
) -> None:
    valid_dataset()
    for driver in (1, 44):
        write_raw(
            "car_data",
            [car_data_record(driver, 1), car_data_record(driver, 2)],
            meeting_key=MEETING_KEY,
            session_key=SESSION_KEY,
            file_name=f"driver_{driver}.json",
        )

    report = ValidationRunner(validation_settings()).run()

    assert report.scope["files_evaluated"] == 6
    assert report.scope["records_inspected"] == 1 + 1 + 2 + 4 + 2 + 2


# -- manifest integration -------------------------------------------------


def write_manifest(directory: Path, entries: List[ManifestEntry], endpoints: List[str]) -> Path:
    manifest = ExecutionManifest(
        pipeline_name="historical_extraction",
        pipeline_version="1.0.0",
        config={"years": [2023], "session_types": ["Race"], "endpoints": endpoints},
    )
    for entry in entries:
        manifest.add(entry)
    manifest.record_session(SESSION_KEY)
    return manifest.save(directory)


def test_a_404_from_the_source_is_reported_as_info(
    valid_dataset, validation_settings, tmp_path: Path
) -> None:
    valid_dataset()
    write_manifest(
        tmp_path / "manifests",
        [
            ManifestEntry(
                endpoint="pit",
                status=ExtractionStatus.FAILED,
                year=2023,
                session_key=SESSION_KEY,
                error="OpenF1 returned HTTP 404 for pit",
                error_type="OpenF1HTTPError",
                status_code=404,
            )
        ],
        endpoints=["drivers", "laps", "pit"],
    )

    report = ValidationRunner(validation_settings()).run()

    unavailable = [item for item in results_for(report, "coverage") if item.endpoint == "pit"]
    assert [item.severity for item in unavailable] == [Severity.INFO]
    assert unavailable[0].description == "pit data unavailable for 2023 in source"
    assert report.global_status is GlobalStatus.PASS


def test_a_server_side_failure_is_reported_as_a_warning(
    valid_dataset, validation_settings, tmp_path: Path
) -> None:
    valid_dataset()
    write_manifest(
        tmp_path / "manifests",
        [
            ManifestEntry(
                endpoint="weather",
                status=ExtractionStatus.FAILED,
                year=2023,
                session_key=SESSION_KEY,
                error_type="OpenF1HTTPError",
                status_code=502,
            )
        ],
        endpoints=["drivers", "laps", "weather"],
    )

    report = ValidationRunner(validation_settings()).run()

    failed = [item for item in results_for(report, "coverage") if item.endpoint == "weather"]
    assert [item.severity for item in failed] == [Severity.WARNING]
    assert report.global_status is GlobalStatus.WARNING


def test_a_timeout_without_a_status_code_is_still_a_warning(
    valid_dataset, validation_settings, tmp_path: Path
) -> None:
    """Manifests from before status_code existed must keep working."""
    valid_dataset()
    write_manifest(
        tmp_path / "manifests",
        [
            ManifestEntry(
                endpoint="stints",
                status=ExtractionStatus.FAILED,
                year=2023,
                session_key=SESSION_KEY,
                error_type="OpenF1TimeoutError",
            )
        ],
        endpoints=["drivers", "laps", "stints"],
    )

    report = ValidationRunner(validation_settings()).run()

    failed = [item for item in results_for(report, "coverage") if item.endpoint == "stints"]
    assert [item.severity for item in failed] == [Severity.WARNING]
    assert failed[0].details["reason"] == "OpenF1TimeoutError"


def test_the_manifest_can_be_ignored(valid_dataset, validation_settings, tmp_path: Path) -> None:
    valid_dataset()
    write_manifest(tmp_path / "manifests", [], endpoints=["drivers", "laps", "pit"])

    report = ValidationRunner(validation_settings(use_manifest=False)).run()

    assert report.source["manifest"] is None
    # Without the manifest, pit is not expected at all, so nothing is missing.
    assert [item for item in results_for(report, "coverage") if item.endpoint == "pit"] == []


# -- sampling -------------------------------------------------------------


def test_sampling_is_disclosed_and_never_reports_a_clean_pass(
    valid_dataset, validation_settings, tmp_path: Path
) -> None:
    valid_dataset()

    report = ValidationRunner(validation_settings(max_records_per_file=1)).run()

    assert report.sampling["enabled"] is True
    assert report.sampling["exhaustive"] is False
    assert report.sampling["max_records_per_file"] == 1
    assert report.sampling["records_inspected"] < report.sampling["records_available"]
    assert "NOT represent an exhaustive validation" in report.sampling["disclaimer"]
    assert report.global_status is GlobalStatus.WARNING
    assert results_for(report, "sampling", Severity.WARNING)

    payload = read_report(tmp_path / "validation")
    # drivers (2 records) and laps (4) exceed the limit; meetings and sessions hold one each.
    assert payload["sampling"]["files_sampled"] == 2


def test_a_full_run_is_marked_exhaustive(valid_dataset, validation_settings) -> None:
    valid_dataset()

    report = ValidationRunner(validation_settings()).run()

    assert report.sampling == {
        "enabled": False,
        "max_records_per_file": None,
        "files_sampled": 0,
        "records_inspected": 8,
        "records_available": 8,
        "exhaustive": True,
    }


def test_sampling_is_deterministic(valid_dataset, validation_settings) -> None:
    valid_dataset()
    settings = validation_settings(max_records_per_file=1)

    first = ValidationRunner(settings).run()
    second = ValidationRunner(settings).run()

    assert [item.to_dict() for item in first.results] == [item.to_dict() for item in second.results]


# -- configuration --------------------------------------------------------


def test_an_unknown_rule_is_rejected_before_anything_is_read(validation_settings) -> None:
    with pytest.raises(ValueError, match="telemetry_sanity"):
        ValidationRunner(validation_settings(enabled_rules=["structure", "telemetry_sanity"]))


def test_disabled_rules_do_not_run(valid_dataset, validation_settings) -> None:
    valid_dataset(laps=[lap_record(1, 1), lap_record(1, 1)])

    report = ValidationRunner(validation_settings(enabled_rules=["structure", "types"])).run()

    assert results_for(report, "duplicates") == []
    assert report.global_status is GlobalStatus.PASS


def test_two_runs_over_the_same_data_agree(valid_dataset, validation_settings) -> None:
    valid_dataset(laps=[lap_record(1, 1), lap_record(44, 1, st_speed=None)])

    first = ValidationRunner(validation_settings()).run()
    second = ValidationRunner(validation_settings()).run()

    assert [item.to_dict() for item in first.results] == [item.to_dict() for item in second.results]
    assert first.global_status is second.global_status
