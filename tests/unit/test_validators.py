"""Rule-level tests: each check, against records built to trip exactly it."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from f1_race_intelligence.validation import validators
from f1_race_intelligence.validation.models import RuleStatus, Severity
from f1_race_intelligence.validation.specs import ENDPOINT_SPECS, EndpointSpec, FieldSpec, get_spec
from f1_race_intelligence.validation.validators import (
    RULE_COVERAGE,
    RULE_DUPLICATES,
    DatasetIndex,
    FileContext,
)

from factories import MEETING_KEY, SESSION_KEY, car_data_record, driver_record, lap_record


def context_for(endpoint: str, records: Sequence[Any], **overrides: Any) -> FileContext:
    return FileContext(
        endpoint=endpoint,
        path=Path(f"data/raw/{endpoint}/session.json"),
        partition=overrides.pop("partition", {"year": 2023, "meeting_key": MEETING_KEY, "session_key": SESSION_KEY}),
        spec=overrides.pop("spec", get_spec(endpoint)),
        records=records,
        total_records=len(records),
        **overrides,
    )


def run_rule(rule: str, context: FileContext) -> List[Any]:
    analysis = validators.analyze_records(context.records, context.spec)
    return validators.FILE_RULE_FUNCTIONS[rule](context, analysis)


def severities(results: Sequence[Any]) -> List[Severity]:
    return [item.severity for item in results]


# -- structure ------------------------------------------------------------


def test_valid_records_produce_no_findings() -> None:
    context = context_for("laps", [lap_record(1, 1), lap_record(1, 2)])

    findings = [item for rule in validators.FILE_RULES for item in run_rule(rule, context)]

    assert [item for item in findings if item.status is RuleStatus.FAIL] == []


def test_non_object_entries_are_an_error() -> None:
    context = context_for("laps", [lap_record(1, 1), "not a record", 42])

    findings = run_rule(validators.RULE_STRUCTURE, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].affected_records == 2


def test_an_empty_payload_is_a_warning() -> None:
    findings = run_rule(validators.RULE_STRUCTURE, context_for("laps", []))

    assert severities(findings) == [Severity.WARNING]
    assert "no records" in findings[0].description


def test_records_with_different_field_sets_are_reported_as_info() -> None:
    partial = lap_record(1, 2)
    partial.pop("st_speed")
    context = context_for("laps", [lap_record(1, 1), partial])

    findings = run_rule(validators.RULE_STRUCTURE, context)

    assert severities(findings) == [Severity.INFO]
    assert findings[0].details["distinct_field_sets"] == 2


# -- schema ---------------------------------------------------------------


def test_a_missing_required_field_is_an_error() -> None:
    broken = lap_record(1, 1)
    broken.pop("lap_number")
    context = context_for("laps", [lap_record(1, 2), broken])

    findings = run_rule(validators.RULE_SCHEMA, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["field"] == "lap_number"
    assert findings[0].affected_records == 1


def test_fields_the_project_does_not_know_are_reported_as_info() -> None:
    context = context_for("laps", [lap_record(1, 1, tyre_pressure=22.5)])

    findings = run_rule(validators.RULE_SCHEMA, context)

    assert severities(findings) == [Severity.INFO]
    assert findings[0].details["fields"] == ["tyre_pressure"]


# -- types ----------------------------------------------------------------


def test_a_wrong_type_is_an_error() -> None:
    context = context_for("laps", [{**lap_record(1, 1), "lap_number": "5"}])

    findings = run_rule(validators.RULE_TYPES, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["field"] == "lap_number"
    assert findings[0].details["observed_types"] == ["str"]


def test_a_boolean_does_not_pass_as_an_integer() -> None:
    context = context_for("laps", [{**lap_record(1, 1), "lap_number": True}])

    findings = run_rule(validators.RULE_TYPES, context)

    assert severities(findings) == [Severity.ERROR]


def test_lapped_cars_reporting_a_text_gap_are_not_a_type_error() -> None:
    """gap_to_leader is '+1 LAP' for lapped drivers in real OpenF1 data."""
    records = [
        {"session_key": SESSION_KEY, "driver_number": 1, "date": "2023-03-05T15:00:00+00:00",
         "gap_to_leader": 12.3, "interval": 1.2},
        {"session_key": SESSION_KEY, "driver_number": 44, "date": "2023-03-05T15:00:01+00:00",
         "gap_to_leader": "+1 LAP", "interval": None},
    ]
    context = context_for("intervals", records)

    findings = [item for rule in validators.FILE_RULES for item in run_rule(rule, context)]

    assert [item for item in findings if item.severity is Severity.ERROR] == []


def test_an_unrecognized_text_gap_is_reported_as_info() -> None:
    records = [
        {"session_key": SESSION_KEY, "driver_number": 1, "date": "2023-03-05T15:00:00+00:00",
         "gap_to_leader": "MAYBE A LAP", "interval": None},
    ]

    findings = run_rule(validators.RULE_VALUES, context_for("intervals", records))

    assert severities(findings) == [Severity.INFO]
    assert findings[0].details["observed"] == ["MAYBE A LAP"]


# -- values ---------------------------------------------------------------


def test_impossible_values_are_errors() -> None:
    context = context_for("car_data", [car_data_record(1, 1, speed=-5)])

    findings = run_rule(validators.RULE_VALUES, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["field"] == "speed"


def test_a_zero_duration_is_an_error_because_the_minimum_is_exclusive() -> None:
    findings = run_rule(validators.RULE_VALUES, context_for("laps", [lap_record(1, 1, lap_duration=0)]))

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["field"] == "lap_duration"


def test_an_implausible_but_possible_duration_is_a_warning() -> None:
    findings = run_rule(validators.RULE_VALUES, context_for("laps", [lap_record(1, 1, lap_duration=900.0)]))

    assert severities(findings) == [Severity.WARNING]
    assert findings[0].details["plausible"] == [40.0, 400.0]


def test_telemetry_above_its_documented_range_is_info_not_an_error() -> None:
    """throttle and brake really do reach 104 in OpenF1 telemetry."""
    context = context_for("car_data", [car_data_record(1, 1, throttle=104, brake=104)])

    findings = run_rule(validators.RULE_VALUES, context)

    assert severities(findings) == [Severity.INFO, Severity.INFO]
    assert {item.details["field"] for item in findings} == {"throttle", "brake"}


def test_an_unknown_drs_code_is_info_because_drs_is_a_state_code() -> None:
    findings = run_rule(validators.RULE_VALUES, context_for("car_data", [car_data_record(1, 1, drs=99)]))

    assert severities(findings) == [Severity.INFO]
    assert findings[0].details["observed"] == [99]


def test_a_stint_that_ends_before_it_starts_is_an_error() -> None:
    records = [
        {"session_key": SESSION_KEY, "driver_number": 1, "stint_number": 1,
         "lap_start": 20, "lap_end": 5, "compound": "SOFT", "tyre_age_at_start": 0},
    ]

    findings = run_rule(validators.RULE_VALUES, context_for("stints", records))

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["examples"] == [{"lap_start": 20, "lap_end": 5}]


def test_an_unknown_tyre_compound_is_info() -> None:
    records = [
        {"session_key": SESSION_KEY, "driver_number": 1, "stint_number": 1,
         "lap_start": 1, "lap_end": 20, "compound": "ULTRASOFT", "tyre_age_at_start": 0},
    ]

    findings = run_rule(validators.RULE_VALUES, context_for("stints", records))

    assert severities(findings) == [Severity.INFO]


# -- missing values -------------------------------------------------------


def test_a_null_in_a_field_that_must_have_a_value_is_an_error() -> None:
    records = [{**lap_record(1, 1), "driver_number": None}]
    findings = run_rule(validators.RULE_MISSING_VALUES, context_for("laps", records))

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["field"] == "driver_number"


def test_nulls_above_the_threshold_are_a_warning() -> None:
    records = [lap_record(1, 1, st_speed=None), lap_record(1, 2, st_speed=None), lap_record(1, 3)]
    context = context_for("laps", records, missing_value_threshold=0.5)

    findings = run_rule(validators.RULE_MISSING_VALUES, context)

    assert severities(findings) == [Severity.WARNING]
    assert findings[0].details["null_ratio"] == pytest.approx(0.6667, abs=1e-4)


def test_nulls_below_the_threshold_are_not_reported() -> None:
    records = [lap_record(1, 1, st_speed=None)] + [lap_record(1, lap) for lap in range(2, 11)]
    context = context_for("laps", records, missing_value_threshold=0.5)

    assert run_rule(validators.RULE_MISSING_VALUES, context) == []


def test_fields_that_are_sparse_by_nature_are_info_whatever_the_ratio() -> None:
    """race_control names a driver only for driver-specific messages."""
    records = [
        {"session_key": SESSION_KEY, "date": "2023-03-05T15:00:00+00:00", "message": "GREEN LIGHT",
         "driver_number": None, "flag": None, "scope": None, "sector": None, "qualifying_phase": None},
    ]

    findings = run_rule(validators.RULE_MISSING_VALUES, context_for("race_control", records))

    assert set(severities(findings)) == {Severity.INFO}


# -- duplicates -----------------------------------------------------------


def test_duplicates_on_the_natural_key_are_an_error() -> None:
    context = context_for("laps", [lap_record(1, 1), lap_record(1, 1), lap_record(1, 2)])

    findings = run_rule(RULE_DUPLICATES, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].affected_records == 1
    assert findings[0].details["natural_key"] == ["session_key", "driver_number", "lap_number"]


def test_duplicate_detection_is_skipped_when_no_natural_key_is_defined() -> None:
    keyless = EndpointSpec(name="mystery", fields={"value": FieldSpec(types=(int,))}, natural_key=None)
    context = context_for("mystery", [{"value": 1}, {"value": 1}], spec=keyless)

    findings = run_rule(RULE_DUPLICATES, context)

    assert findings[0].status is RuleStatus.SKIPPED
    assert findings[0].severity is Severity.INFO


# -- temporal -------------------------------------------------------------


def test_an_unparseable_timestamp_is_an_error() -> None:
    context = context_for("car_data", [car_data_record(1, 1, date="not-a-timestamp")])

    findings = run_rule(validators.RULE_TEMPORAL, context)

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["example"] == "not-a-timestamp"


def test_records_out_of_temporal_order_are_a_warning() -> None:
    context = context_for("car_data", [car_data_record(1, 10), car_data_record(1, 2)])

    findings = run_rule(validators.RULE_TEMPORAL, context)

    assert severities(findings) == [Severity.WARNING]
    assert findings[0].affected_records == 1


def test_order_is_checked_per_driver_not_across_them() -> None:
    """Interleaved drivers are normal; each driver's own sequence is what matters."""
    records = [lap_record(1, 1), lap_record(1, 2), lap_record(44, 1), lap_record(44, 2)]

    assert run_rule(validators.RULE_TEMPORAL, context_for("laps", records)) == []


# -- identifiers ----------------------------------------------------------


def build_index(files: Sequence[FileContext]) -> DatasetIndex:
    index = DatasetIndex()
    for context in files:
        index.observe(context)
    return index


def test_a_session_missing_from_the_catalog_is_an_error() -> None:
    sessions = context_for("sessions", [{"session_key": SESSION_KEY, "meeting_key": MEETING_KEY, "year": 2023}])
    laps = context_for("laps", [lap_record(1, 1, session_key=9999)])

    findings = validators.validate_identifiers(build_index([sessions, laps]))

    assert severities(findings) == [Severity.ERROR]
    assert findings[0].details["unknown_session_keys"] == [9999]


def test_session_checks_are_skipped_when_there_is_no_sessions_catalog() -> None:
    findings = validators.validate_identifiers(build_index([context_for("laps", [lap_record(1, 1)])]))

    assert findings[0].status is RuleStatus.SKIPPED


def test_a_driver_in_laps_but_not_in_drivers_is_a_warning_not_an_error() -> None:
    sessions = context_for("sessions", [{"session_key": SESSION_KEY, "meeting_key": MEETING_KEY, "year": 2023}])
    drivers = context_for("drivers", [driver_record(1, "VER", "Red Bull")])
    laps = context_for("laps", [lap_record(1, 1), lap_record(44, 1)])

    findings = validators.validate_identifiers(build_index([sessions, drivers, laps]))

    assert severities(findings) == [Severity.WARNING]
    assert findings[0].details["driver_numbers"] == [44]


def test_a_session_referencing_an_unknown_meeting_is_an_error() -> None:
    meetings = context_for("meetings", [{"meeting_key": 1, "year": 2023}], partition={"year": 2023})
    sessions = context_for("sessions", [{"session_key": SESSION_KEY, "meeting_key": 999, "year": 2023}])

    findings = validators.validate_identifiers(build_index([meetings, sessions]))

    assert [item.details.get("unknown_meeting_keys") for item in findings if item.severity is Severity.ERROR] == [[999]]


# -- coverage -------------------------------------------------------------


class FakeManifest:
    def __init__(self, failures: Dict[Any, Dict[str, Any]]) -> None:
        self._failures = failures

    def failure_for(self, endpoint: str, session_key: int) -> Optional[Dict[str, Any]]:
        return self._failures.get((endpoint, session_key))


def index_with_one_session() -> DatasetIndex:
    sessions = context_for("sessions", [{"session_key": SESSION_KEY, "meeting_key": MEETING_KEY, "year": 2023}])
    laps = context_for("laps", [lap_record(1, 1)])
    return build_index([sessions, laps])


def test_data_absent_at_the_source_is_reported_as_info() -> None:
    manifest = FakeManifest({("pit", SESSION_KEY): {"status_code": 404, "error_type": "OpenF1HTTPError"}})

    findings = validators.validate_coverage(index_with_one_session(), ["laps", "pit"], manifest)

    unavailable = [item for item in findings if item.endpoint == "pit"]
    assert severities(unavailable) == [Severity.INFO]
    assert unavailable[0].description == "pit data unavailable for 2023 in source"
    assert "404" in unavailable[0].details["reason"]


def test_data_missing_because_extraction_failed_is_a_warning() -> None:
    manifest = FakeManifest({("weather", SESSION_KEY): {"status_code": 500, "error_type": "OpenF1HTTPError"}})

    findings = validators.validate_coverage(index_with_one_session(), ["laps", "weather"], manifest)

    failed = [item for item in findings if item.endpoint == "weather"]
    assert severities(failed) == [Severity.WARNING]
    assert "re-running" in failed[0].description


def test_data_that_was_never_attempted_is_a_warning() -> None:
    findings = validators.validate_coverage(index_with_one_session(), ["laps", "stints"], FakeManifest({}))

    missing = [item for item in findings if item.endpoint == "stints"]
    assert severities(missing) == [Severity.WARNING]
    assert "never recorded" in missing[0].description


def test_lap_coverage_is_described_not_asserted_to_be_uniform() -> None:
    """Drivers legitimately complete different numbers of laps."""
    sessions = context_for("sessions", [{"session_key": SESSION_KEY, "meeting_key": MEETING_KEY, "year": 2023}])
    laps = context_for("laps", [lap_record(1, 1), lap_record(1, 2), lap_record(44, 1)])

    findings = validators.validate_coverage(build_index([sessions, laps]), ["laps"], None)

    summary = next(item for item in findings if item.details.get("laps_min") is not None)
    assert summary.status is RuleStatus.PASS
    assert summary.details["laps_min"] == 1
    assert summary.details["laps_max"] == 2
    assert summary.details["drivers_with_laps"] == 2


def test_sessions_the_extraction_never_targeted_are_not_demanded() -> None:
    """A race-only extraction must not be faulted for having no practice data."""
    sessions = context_for(
        "sessions",
        [
            {"session_key": SESSION_KEY, "meeting_key": MEETING_KEY, "year": 2023, "session_name": "Race"},
            {"session_key": 7765, "meeting_key": MEETING_KEY, "year": 2023, "session_name": "Practice 1"},
        ],
    )
    laps = context_for("laps", [lap_record(1, 1)])
    index = build_index([sessions, laps])

    findings = validators.validate_coverage(index, ["laps"], None, expected_sessions=[SESSION_KEY])

    assert [item for item in findings if item.status is RuleStatus.FAIL] == []
    skipped = next(item for item in findings if item.details.get("sessions_not_extracted"))
    assert skipped.severity is Severity.INFO
    assert skipped.details["session_keys"] == [7765]


def test_an_empty_dataset_is_reported_as_a_coverage_warning() -> None:
    findings = validators.validate_coverage(DatasetIndex(), ["laps"], None)

    assert severities(findings) == [Severity.WARNING]
    assert "No sessions found" in findings[0].description


# -- spec sanity ----------------------------------------------------------


@pytest.mark.parametrize("endpoint", sorted(ENDPOINT_SPECS))
def test_every_spec_references_fields_it_declares(endpoint: str) -> None:
    """Guards against a typo in specs.py silently disabling a check."""
    spec = ENDPOINT_SPECS[endpoint]
    declared = set(spec.fields)

    assert spec.required_fields, f"{endpoint} declares no required field"
    assert set(spec.natural_key or ()) <= declared
    assert set(spec.timestamp_fields) <= declared
    if spec.order_field:
        assert spec.order_field in declared
    if spec.order_group:
        assert spec.order_group in declared
    if spec.natural_key:
        assert spec.natural_key_note, f"{endpoint} has a natural key with no justification"
