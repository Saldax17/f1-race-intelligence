"""Tests for M4: a small controlled raw tree in, one lap-grain table out.

Timings come from the factories, where the race starts at 15:00:00 and
every lap is exactly 100 seconds. That makes it possible to state which
lap a sample at a given offset must land in, instead of asserting on
whatever the code happens to produce.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest

from f1_race_intelligence.consolidation.runner import ConsolidationRunner
from factories import (
    MEETING_KEY,
    MEETING_RECORD,
    SESSION_KEY,
    SESSION_RECORD,
    car_data_record,
    driver_record,
    interval_record,
    lap_offset,
    lap_record,
    pit_record,
    position_record,
    race_control_record,
    stint_record,
    weather_record,
)

DRIVERS = (1, 44)


@pytest.fixture
def build_raw(write_raw: Callable[..., Path]) -> Callable[..., None]:
    """Write a small but internally consistent session into the raw tree."""

    def _build(**sources: Optional[List[Dict[str, Any]]]) -> None:
        def value(name: str, default: Any) -> Any:
            return default if name not in sources else sources[name]

        laps = value("laps", [lap_record(d, lap) for d in DRIVERS for lap in (1, 2, 3)])

        write_raw("meetings", [MEETING_RECORD])
        write_raw("sessions", [SESSION_RECORD], meeting_key=MEETING_KEY)
        for endpoint, records in (
            ("laps", laps),
            ("drivers", value("drivers", [driver_record(d, f"D{d}", f"Team {d}") for d in DRIVERS])),
            ("stints", value("stints", [stint_record(d, 1, 1, 3) for d in DRIVERS])),
            ("pit", value("pit", [])),
            ("position", value("position", [position_record(d, -10, i + 1) for i, d in enumerate(DRIVERS)])),
            ("intervals", value("intervals", [])),
            ("weather", value("weather", [weather_record(-30), weather_record(70), weather_record(170)])),
            ("race_control", value("race_control", [])),
        ):
            if records is None:
                continue
            write_raw(endpoint, records, meeting_key=MEETING_KEY, session_key=SESSION_KEY)

        for driver, records in (value("car_data", None) or {}).items():
            write_raw(
                "car_data",
                records,
                meeting_key=MEETING_KEY,
                session_key=SESSION_KEY,
                file_name=f"driver_{driver}.json",
            )

    return _build


def consolidate(settings) -> pd.DataFrame:
    report = ConsolidationRunner(settings).run()
    path = Path(settings.consolidation.output_path)
    files = list(path.rglob("*.parquet"))
    assert files, "no parquet written"
    frame = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in sorted(files)], ignore_index=True)
    frame.attrs["report"] = report
    return frame


def row(frame: pd.DataFrame, driver: int, lap: int) -> pd.Series:
    match = frame[(frame.driver_number == driver) & (frame.lap_number == lap)]
    assert len(match) == 1, f"expected one row for driver {driver} lap {lap}, got {len(match)}"
    return match.iloc[0]


# -- grain, keys and joins ------------------------------------------------


def test_the_grain_is_one_row_per_driver_and_lap(build_raw, consolidation_settings) -> None:
    build_raw()

    frame = consolidate(consolidation_settings())

    assert len(frame) == 6
    assert not frame.duplicated(subset=["session_key", "driver_number", "lap_number"]).any()
    assert sorted(frame.driver_number.unique()) == list(DRIVERS)


def test_every_row_carries_the_keys_back_to_its_sources(build_raw, consolidation_settings) -> None:
    build_raw()

    frame = consolidate(consolidation_settings())

    for column, expected in (("year", 2023), ("meeting_key", MEETING_KEY), ("session_key", SESSION_KEY)):
        assert (frame[column] == expected).all()


def test_drivers_join_adds_identity_without_changing_the_row_count(build_raw, consolidation_settings) -> None:
    build_raw()

    frame = consolidate(consolidation_settings())

    assert row(frame, 1, 1).team_name == "Team 1"
    assert row(frame, 44, 3).driver_acronym == "D44"
    assert len(frame) == 6


# -- car_data -------------------------------------------------------------


def test_car_data_is_aggregated_per_lap(build_raw, consolidation_settings) -> None:
    samples = {
        1: [
            car_data_record(1, 10, speed=100, rpm=9000),
            car_data_record(1, 20, speed=300, rpm=12000),
            car_data_record(1, 110, speed=200, rpm=10000),
        ]
    }
    build_raw(car_data=samples)

    frame = consolidate(consolidation_settings())

    first = row(frame, 1, 1)
    assert first.car_data_samples == 2
    assert first.speed_min == 100
    assert first.speed_max == 300
    assert first.speed_mean == 200
    assert row(frame, 1, 2).car_data_samples == 1


def test_only_samples_inside_the_lap_window_are_aggregated(build_raw, consolidation_settings) -> None:
    """A lap is described by its own window, never by what came before or after."""
    samples = {
        1: [
            car_data_record(1, -50, speed=1),      # before the race: the formation lap / garage
            car_data_record(1, 50, speed=250),     # lap 1
            car_data_record(1, 150, speed=260),    # lap 2
            car_data_record(1, 5000, speed=2),     # long after the last lap ended
        ]
    }
    build_raw(car_data=samples)

    frame = consolidate(consolidation_settings())

    assert row(frame, 1, 1).car_data_samples == 1
    assert row(frame, 1, 1).speed_max == 250
    assert row(frame, 1, 2).car_data_samples == 1
    report = frame.attrs["report"].sessions[0]
    assert report.car_data["samples_read"] == 4
    assert report.car_data["samples_in_window"] == 2


def test_impossible_telemetry_is_excluded_from_aggregates_but_counted(build_raw, consolidation_settings) -> None:
    """The raw file keeps the value; the aggregate does not inherit it."""
    samples = {
        1: [
            car_data_record(1, 10, n_gear=7, throttle=100, brake=0),
            car_data_record(1, 20, n_gear=20, throttle=104, brake=104),
        ]
    }
    build_raw(car_data=samples)

    frame = consolidate(consolidation_settings())

    first = row(frame, 1, 1)
    assert first.n_gear_max == 7, "gear 20 must not become the maximum"
    assert first.throttle_mean == 100, "throttle 104 must not drag the mean"
    assert first.car_data_samples == 2
    assert first.car_data_invalid_samples == 1
    assert frame.attrs["report"].sessions[0].car_data["invalid_samples"] == 1


def test_brake_and_drs_shares_are_proportions_of_valid_samples(build_raw, consolidation_settings) -> None:
    samples = {
        1: [
            car_data_record(1, 10, brake=0, drs=0),
            car_data_record(1, 20, brake=100, drs=12),
            car_data_record(1, 30, brake=0, drs=8),
            car_data_record(1, 40, brake=100, drs=14),
        ]
    }
    build_raw(car_data=samples)

    frame = consolidate(consolidation_settings())

    first = row(frame, 1, 1)
    assert first.brake_applied_share == 0.5
    assert first.drs_active_share == 0.5, "code 8 means eligible, not open"


# -- weather --------------------------------------------------------------


def test_weather_comes_from_before_the_lap_started(build_raw, consolidation_settings) -> None:
    """Never a reading taken after the lap began, even if it is closer in time."""
    build_raw(weather=[
        weather_record(-30, air_temperature=20.0),   # before lap 1 started (offset 0)
        weather_record(5, air_temperature=99.0),     # during lap 1, before lap 2 started (offset 100)
        weather_record(105, air_temperature=30.0),   # during lap 2, before lap 3 started (offset 200)
    ])

    frame = consolidate(consolidation_settings())

    # Lap 1 cannot use the reading taken 5 s after it began, even though that
    # one is nearer in time than the reading from 30 s before.
    assert row(frame, 1, 1).air_temperature == 20.0
    assert row(frame, 1, 2).air_temperature == 99.0
    assert row(frame, 1, 3).air_temperature == 30.0
    assert (frame.weather_age_s.dropna() >= 0).all()


def test_weather_further_away_than_the_tolerance_is_left_empty(build_raw, consolidation_settings) -> None:
    build_raw(weather=[weather_record(-500, air_temperature=20.0)])

    frame = consolidate(consolidation_settings(weather_tolerance_seconds=60))

    assert pd.isna(row(frame, 1, 1).air_temperature)
    assert pd.isna(row(frame, 1, 1).weather_age_s)


def test_a_session_without_weather_still_consolidates(build_raw, consolidation_settings) -> None:
    build_raw(weather=[])

    frame = consolidate(consolidation_settings())

    assert len(frame) == 6
    assert frame.air_temperature.isna().all()


# -- position -------------------------------------------------------------


def test_position_is_carried_forward_when_a_lap_has_no_record(build_raw, consolidation_settings) -> None:
    """Position is only reported on change, so most laps inherit the last known one."""
    build_raw(position=[
        position_record(1, -10, 3),
        position_record(1, 150, 2),   # a change during lap 2
    ])

    frame = consolidate(consolidation_settings())

    assert row(frame, 1, 1).position_at_lap_start == 3
    assert row(frame, 1, 2).position_at_lap_start == 3, "no change yet when lap 2 began"
    assert row(frame, 1, 2).position_at_lap_end == 2
    assert row(frame, 1, 3).position_at_lap_start == 2, "carried forward into lap 3"
    assert row(frame, 1, 2).position_changes_in_lap == 1
    assert row(frame, 1, 1).position_changes_in_lap == 0


def test_position_before_any_record_is_left_empty(build_raw, consolidation_settings) -> None:
    """No position is invented for laps that ran before the first report."""
    build_raw(position=[position_record(1, 250, 5)])   # during lap 3, which spans 200-300

    frame = consolidate(consolidation_settings())

    assert pd.isna(row(frame, 1, 1).position_at_lap_start)
    assert pd.isna(row(frame, 1, 3).position_at_lap_start), "not yet known when lap 3 began"
    assert row(frame, 1, 3).position_at_lap_end == 5


# -- intervals ------------------------------------------------------------


def test_intervals_take_the_value_at_the_end_of_the_lap(build_raw, consolidation_settings) -> None:
    build_raw(intervals=[
        interval_record(1, 10, gap_to_leader=5.0, interval=1.0),
        interval_record(1, 90, gap_to_leader=3.0, interval=0.5),
    ])

    frame = consolidate(consolidation_settings())

    first = row(frame, 1, 1)
    assert first.gap_to_leader_s == 3.0
    assert first.interval_s == 0.5
    assert first.interval_samples == 2


def test_a_lapped_car_is_flagged_and_has_no_numeric_gap(build_raw, consolidation_settings) -> None:
    """gap_to_leader is text ('+1 LAP') for lapped cars in real OpenF1 data."""
    build_raw(intervals=[
        interval_record(1, 10, gap_to_leader=5.0),
        interval_record(1, 90, gap_to_leader="+1 LAP", interval=None),
    ])

    frame = consolidate(consolidation_settings())

    first = row(frame, 1, 1)
    assert bool(first.is_lapped) is True
    assert pd.isna(first.gap_to_leader_s), "the flag and the gap must describe the same instant"


# -- stints and pit -------------------------------------------------------


def test_stints_map_onto_laps_and_tyre_age_follows(build_raw, consolidation_settings) -> None:
    build_raw(stints=[
        stint_record(1, 1, 1, 2, compound="SOFT", tyre_age_at_start=3),
        stint_record(1, 2, 3, 3, compound="HARD", tyre_age_at_start=0),
    ])

    frame = consolidate(consolidation_settings())

    assert row(frame, 1, 1).compound == "SOFT"
    assert row(frame, 1, 1).tyre_age == 3
    assert row(frame, 1, 2).tyre_age == 4
    assert row(frame, 1, 2).stint_lap == 2
    assert row(frame, 1, 3).compound == "HARD"
    assert row(frame, 1, 3).tyre_age == 0
    assert row(frame, 1, 3).stint_number == 2


def test_a_lap_beyond_the_last_stint_has_no_stint(build_raw, consolidation_settings) -> None:
    build_raw(stints=[stint_record(1, 1, 1, 2)])

    frame = consolidate(consolidation_settings())

    assert pd.isna(row(frame, 1, 3).stint_number)
    assert pd.isna(row(frame, 1, 3).tyre_age)


def test_a_pit_stop_lands_on_the_lap_the_car_entered_the_pits(build_raw, consolidation_settings) -> None:
    """pit.lap_number is the in-lap; the next lap is the one flagged pit-out."""
    laps = [lap_record(d, lap) for d in DRIVERS for lap in (1, 2, 3)]
    laps = [
        {**lap, "is_pit_out_lap": True} if lap["driver_number"] == 1 and lap["lap_number"] == 3 else lap
        for lap in laps
    ]
    build_raw(laps=laps, pit=[pit_record(1, 2, pit_duration=24.5)])

    frame = consolidate(consolidation_settings())

    assert bool(row(frame, 1, 2).pit_in_lap) is True
    assert row(frame, 1, 2).pit_duration == 24.5
    assert bool(row(frame, 1, 3).pit_in_lap) is False
    assert bool(row(frame, 1, 3).is_pit_out_lap) is True


def test_a_session_without_pit_data_keeps_the_pit_out_flag(build_raw, consolidation_settings) -> None:
    """2023 races have no pit endpoint at all; is_pit_out_lap is then the only trace."""
    laps = [lap_record(d, lap) for d in DRIVERS for lap in (1, 2, 3)]
    laps = [
        {**lap, "is_pit_out_lap": True} if lap["driver_number"] == 1 and lap["lap_number"] == 3 else lap
        for lap in laps
    ]
    build_raw(laps=laps, pit=None)

    frame = consolidate(consolidation_settings())

    assert frame.pit_duration.isna().all()
    assert bool(row(frame, 1, 3).is_pit_out_lap) is True
    assert not frame.pit_in_lap.any()


# -- race control ---------------------------------------------------------


def test_race_control_events_are_counted_per_lap_and_per_driver(build_raw, consolidation_settings) -> None:
    build_raw(race_control=[
        race_control_record(2),
        race_control_record(2, driver_number=1),
        race_control_record(3, driver_number=44),
    ])

    frame = consolidate(consolidation_settings())

    assert row(frame, 1, 2).rc_events_in_lap == 2
    assert row(frame, 44, 2).rc_events_in_lap == 2, "session-wide events count for every driver"
    assert row(frame, 1, 2).rc_driver_events_in_lap == 1
    assert row(frame, 44, 2).rc_driver_events_in_lap == 0
    assert row(frame, 1, 1).rc_events_in_lap == 0


# -- rows kept, rows lost -------------------------------------------------


def test_no_rows_are_lost_and_every_stage_says_so(build_raw, consolidation_settings) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})

    frame = consolidate(consolidation_settings())
    result = frame.attrs["report"].sessions[0]

    assert result.rows_in == 6
    assert result.rows_out == 6
    assert [stage.stage for stage in result.stages][-1] == "car_data"
    for stage in result.stages:
        assert stage.rows_out == stage.rows_in, f"stage {stage.stage} changed the row count"


def test_stage_counters_record_how_many_rows_a_source_matched(build_raw, consolidation_settings) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})

    frame = consolidate(consolidation_settings())
    stages = {stage.stage: stage for stage in frame.attrs["report"].sessions[0].stages}

    assert stages["car_data"].matched == 1, "only driver 1 lap 1 has telemetry"
    assert stages["car_data"].unmatched == 5
    assert stages["drivers"].matched == 6


def test_the_last_lap_without_a_closable_window_is_kept_not_dropped(build_raw, consolidation_settings) -> None:
    """Dropping it would bias the dataset towards drivers who finished."""
    laps = [lap_record(1, 1), lap_record(1, 2, lap_duration=None)]
    build_raw(laps=laps, car_data={1: [car_data_record(1, 10)]})

    frame = consolidate(consolidation_settings())

    assert len(frame) == 2
    last = row(frame, 1, 2)
    assert bool(last.is_last_lap_for_driver) is True
    assert bool(last.has_complete_window) is False
    assert pd.isna(last.car_data_samples)
    assert frame.attrs["report"].sessions[0].checks["laps_without_window"] == 1


# -- empty and partial sources -------------------------------------------


def test_a_session_with_only_laps_still_produces_the_dataset(build_raw, consolidation_settings) -> None:
    build_raw(drivers=[], stints=[], position=[], intervals=[], weather=[], race_control=[], pit=[])

    frame = consolidate(consolidation_settings())

    assert len(frame) == 6
    assert frame.team_name.isna().all()
    assert frame.compound.isna().all()
    assert frame.car_data_samples.isna().all()
    assert frame.rc_events_in_lap.eq(0).all()


def test_missing_endpoints_leave_columns_empty_rather_than_invented(build_raw, consolidation_settings) -> None:
    build_raw(stints=None, position=None, intervals=None)

    frame = consolidate(consolidation_settings())

    assert len(frame) == 6
    for column in ("stint_number", "tyre_age", "position_at_lap_start", "gap_to_leader_s"):
        assert frame[column].isna().all(), column


def test_a_session_without_laps_is_reported_and_skipped(write_raw, consolidation_settings) -> None:
    write_raw("meetings", [MEETING_RECORD])
    write_raw("sessions", [SESSION_RECORD], meeting_key=MEETING_KEY)

    report = ConsolidationRunner(consolidation_settings()).run()

    assert report.summary()["sessions_processed"] == 0


# -- report, checks and reproducibility -----------------------------------


def test_the_report_records_the_cardinality_and_the_checks(build_raw, consolidation_settings, tmp_path) -> None:
    build_raw()

    ConsolidationRunner(consolidation_settings()).run()

    files = list((tmp_path / "consolidation").glob("consolidation_report_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["pipeline_name"] == "data_consolidation"
    assert payload["summary"]["rows_in"] == 6
    assert payload["summary"]["rows_out"] == 6
    assert payload["summary"]["rows_lost"] == 0
    assert payload["checks"]["grain_unique"] is True
    assert payload["sessions"][0]["checks"]["grain_unique"] is True
    assert payload["sessions"][0]["stages"]


def test_two_runs_over_the_same_raw_data_produce_the_same_dataset(build_raw, consolidation_settings) -> None:
    build_raw(car_data={1: [car_data_record(1, 10), car_data_record(1, 20)]})

    first = consolidate(consolidation_settings())
    second = consolidate(consolidation_settings(overwrite=True))

    pd.testing.assert_frame_equal(first.drop(columns=[]), second.drop(columns=[]))


def test_an_already_consolidated_session_is_skipped(build_raw, consolidation_settings) -> None:
    build_raw()
    settings = consolidation_settings()
    ConsolidationRunner(settings).run()

    report = ConsolidationRunner(settings).run()

    assert report.summary()["sessions_skipped"] == 1
    assert report.summary()["sessions_processed"] == 0


# -- validation gate ------------------------------------------------------


def write_validation_report(directory: Path, status: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation_report_20230305T120000000000Z.json").write_text(
        json.dumps({"global_status": status, "summary": {"errors": 0}}), encoding="utf-8"
    )


def test_a_failing_validation_does_not_block_but_is_recorded(build_raw, consolidation_settings, tmp_path) -> None:
    build_raw()
    write_validation_report(tmp_path / "validation", "FAIL")

    report = ConsolidationRunner(consolidation_settings()).run()

    assert report.summary()["sessions_processed"] == 1
    assert report.source_validation["status"] == "FAIL"
    assert "consolidated anyway" in report.source_validation["note"].lower()


def test_require_validation_pass_stops_the_run(build_raw, consolidation_settings, tmp_path) -> None:
    build_raw()
    write_validation_report(tmp_path / "validation", "FAIL")

    report = ConsolidationRunner(consolidation_settings(require_validation_pass=True)).run()

    assert report.aborted is not None
    assert "FAIL" in report.aborted
    assert report.summary()["sessions_processed"] == 0
    assert not list((tmp_path / "processed").rglob("*.parquet"))


def test_require_validation_pass_allows_a_passing_run(build_raw, consolidation_settings, tmp_path) -> None:
    build_raw()
    write_validation_report(tmp_path / "validation", "PASS")

    report = ConsolidationRunner(consolidation_settings(require_validation_pass=True)).run()

    assert report.aborted is None
    assert report.summary()["sessions_processed"] == 1


# -- output ---------------------------------------------------------------


def test_the_output_is_parquet_partitioned_by_season(build_raw, consolidation_settings, tmp_path) -> None:
    build_raw()

    ConsolidationRunner(consolidation_settings()).run()

    expected = tmp_path / "processed" / "lap_dataset" / "2023" / f"session_{SESSION_KEY}.parquet"
    assert expected.exists()


def test_the_dataset_keeps_its_types_through_parquet(build_raw, consolidation_settings) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})

    frame = consolidate(consolidation_settings())

    assert frame.lap_number.dtype == "Int64"
    assert frame.is_pit_out_lap.dtype == "boolean"
    assert frame.lap_duration.dtype == "Float64"
    assert str(frame.lap_start_time.dtype).startswith("datetime64")


def test_raw_data_is_never_modified(build_raw, consolidation_settings, raw_root: Path) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})
    before = {path: path.read_bytes() for path in sorted(raw_root.rglob("*.json"))}

    ConsolidationRunner(consolidation_settings()).run()

    after = {path: path.read_bytes() for path in sorted(raw_root.rglob("*.json"))}
    assert after == before
