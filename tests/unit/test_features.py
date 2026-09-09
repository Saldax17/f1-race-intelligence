"""Tests for M5: a consolidated lap dataset in, a modelling dataset out.

The anti-leakage guards get tested from both sides. Confirming they pass on
correct data proves nothing on its own — a check that always passes would
too — so each one is also fed data that leaks, and is required to fail.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.features import checks, selection
from f1_race_intelligence.features.runner import FeatureRunner
from f1_race_intelligence.features.selection import FeatureRole, TARGET_COLUMN
from f1_race_intelligence.features.target import add_target
from f1_race_intelligence.features.temporal import add_temporal_features

RACE_START = pd.Timestamp("2024-03-02T15:00:00Z")


def lap_rows(
    session_key: int = 9472,
    year: int = 2024,
    drivers: tuple = (1, 44),
    laps: int = 6,
    durations: Optional[Dict[int, List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Build rows shaped like M4's consolidated output."""
    rows = []
    for driver in drivers:
        values = (durations or {}).get(driver) or [95.0 + lap for lap in range(1, laps + 1)]
        for index, duration in enumerate(values, start=1):
            rows.append(
                {
                    "year": year,
                    "meeting_key": 1229,
                    "session_key": session_key,
                    "driver_number": driver,
                    "lap_number": index,
                    "team_name": f"Team {driver}",
                    "driver_acronym": f"D{driver}",
                    "lap_start_time": RACE_START + pd.Timedelta(seconds=100 * (index - 1)),
                    "lap_duration": duration,
                    "compound": "SOFT",
                    "tyre_age": index,
                    "is_pit_out_lap": False,
                    "pit_in_lap": False,
                    "pit_duration": None,
                    "pit_lane_duration": None,
                    "is_last_lap_for_driver": index == len(values),
                    "has_complete_window": True,
                    "position_at_lap_start": 1,
                    "speed_mean": 200.0,
                    "air_temperature": 20.0,
                    "rc_events_in_lap": 0,
                }
            )
    return rows


@pytest.fixture
def lap_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Write a consolidated dataset the way M4 would."""

    def _write(rows: List[Dict[str, Any]], year: int = 2024, session_key: int = 9472) -> Path:
        base = tmp_path / "lap_dataset" / str(year)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"session_{session_key}.parquet"
        pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
        return path

    return _write


@pytest.fixture
def feature_settings(tmp_path: Path) -> Callable[..., AppSettings]:
    def _settings(**overrides: Any) -> AppSettings:
        config: Dict[str, Any] = {
            "input_path": str(tmp_path / "lap_dataset"),
            "output_path": str(tmp_path / "features"),
            "report_path": str(tmp_path / "feature_reports"),
        }
        config.update(overrides)
        return AppSettings(features=config)

    return _settings


def build(settings) -> pd.DataFrame:
    report = FeatureRunner(settings).run()
    files = list(Path(settings.features.output_path).rglob("*.parquet"))
    assert files, "no parquet written"
    frame = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in sorted(files)], ignore_index=True)
    frame.attrs["report"] = report
    return frame


def driver_laps(frame: pd.DataFrame, driver: int) -> pd.DataFrame:
    return frame[frame.driver_number == driver].sort_values("lap_number")


# -- target ---------------------------------------------------------------


def test_the_target_is_the_next_lap_of_the_same_driver(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(durations={1: [90.0, 91.0, 92.0, 93.0], 44: [80.0, 81.0, 82.0, 83.0]}))

    frame = build(feature_settings())

    one = driver_laps(frame, 1)
    assert one.next_lap_time.tolist()[:3] == [91.0, 92.0, 93.0]
    assert driver_laps(frame, 44).next_lap_time.tolist()[:3] == [81.0, 82.0, 83.0]


def test_the_last_lap_has_no_target_and_is_kept(lap_dataset, feature_settings) -> None:
    """Dropping it silently would hide the drivers who retired."""
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0, 91.0, 92.0]}))

    frame = build(feature_settings())

    last = frame[frame.lap_number == 3].iloc[0]
    assert pd.isna(last.next_lap_time)
    assert bool(last.has_target) is False
    assert len(frame) == 3, "the row is kept, not dropped"


def test_a_retirement_does_not_borrow_a_target_from_another_driver(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(durations={1: [90.0, 91.0], 44: [80.0, 81.0, 82.0, 83.0]}))

    frame = build(feature_settings())

    one = driver_laps(frame, 1)
    assert len(one) == 2
    assert one.next_lap_time.tolist() == [91.0, None] or pd.isna(one.next_lap_time.iloc[1])
    assert one.next_lap_time.iloc[0] == 91.0


def test_the_target_never_crosses_a_session(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(session_key=1, year=2023, drivers=(1,), durations={1: [90.0, 91.0]}), year=2023, session_key=1)
    lap_dataset(lap_rows(session_key=2, year=2024, drivers=(1,), durations={1: [70.0, 71.0]}), year=2024, session_key=2)

    frame = build(feature_settings())

    first = frame[(frame.session_key == 1) & (frame.lap_number == 2)].iloc[0]
    assert pd.isna(first.next_lap_time), "the next session must not supply a target"


def test_a_gap_in_the_lap_sequence_produces_no_target() -> None:
    """Lap 5 must not be paired with lap 7 as if they were consecutive."""
    rows = pd.DataFrame(
        [
            {"session_key": 1, "driver_number": 1, "lap_number": 5, "lap_duration": 90.0},
            {"session_key": 1, "driver_number": 1, "lap_number": 7, "lap_duration": 95.0},
        ]
    )

    result, breakdown = add_target(rows)

    assert pd.isna(result[TARGET_COLUMN].iloc[0])
    assert breakdown["without_target_lap_sequence_gap"] == 1


# -- temporal features ----------------------------------------------------


def test_previous_lap_time_is_the_lap_before(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0, 92.0, 91.0, 93.0]}))

    one = driver_laps(build(feature_settings()), 1)

    assert pd.isna(one.lap_time_prev_1.iloc[0]), "there is no lap before the first"
    assert one.lap_time_prev_1.tolist()[1:] == [90.0, 92.0, 91.0]
    assert one.lap_time_delta_1.tolist()[1:] == [2.0, -1.0, 2.0]


def test_rolling_windows_cover_the_current_lap_and_the_ones_before(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0, 100.0, 110.0, 120.0, 130.0]}))

    one = driver_laps(build(feature_settings()), 1)

    # Lap 3's window of 3 covers laps 1-3.
    assert one.lap_time_roll_mean_3.iloc[2] == pytest.approx(100.0)
    # Lap 5's window of 3 covers laps 3-5, not the whole race.
    assert one.lap_time_roll_mean_3.iloc[4] == pytest.approx(120.0)
    assert one.lap_time_roll_min_5.iloc[4] == pytest.approx(90.0)
    assert one.lap_time_expanding_mean.iloc[4] == pytest.approx(110.0)


def test_the_first_lap_behaves_as_documented(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0, 95.0, 96.0]}))

    first = driver_laps(build(feature_settings()), 1).iloc[0]

    assert pd.isna(first.lap_time_prev_1)
    assert pd.isna(first.lap_time_delta_1)
    assert pd.isna(first.lap_time_roll_std_3), "a spread needs two laps"
    assert first.lap_time_roll_mean_3 == 90.0
    assert first.lap_time_roll_min_5 == 90.0
    assert first.lap_time_vs_roll_mean_5 == 0.0
    assert first.lap_time_expanding_mean == 90.0


def test_rolling_features_never_mix_drivers(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(durations={1: [90.0, 90.0, 90.0], 44: [200.0, 200.0, 200.0]}))

    frame = build(feature_settings())

    assert driver_laps(frame, 1).lap_time_roll_mean_3.tolist() == [90.0, 90.0, 90.0]
    assert driver_laps(frame, 44).lap_time_roll_mean_3.tolist() == [200.0, 200.0, 200.0]


def test_rolling_features_never_mix_sessions(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(session_key=1, year=2023, drivers=(1,), durations={1: [90.0, 90.0]}), year=2023, session_key=1)
    lap_dataset(lap_rows(session_key=2, year=2024, drivers=(1,), durations={1: [200.0, 200.0]}), year=2024, session_key=2)

    frame = build(feature_settings())

    later = frame[(frame.session_key == 2) & (frame.lap_number == 1)].iloc[0]
    assert later.lap_time_expanding_mean == 200.0, "the earlier race must not bleed in"


def test_temporal_features_are_computed_in_lap_order_whatever_the_input_order() -> None:
    shuffled = pd.DataFrame(
        [
            {"session_key": 1, "driver_number": 1, "lap_number": 3, "lap_duration": 92.0},
            {"session_key": 1, "driver_number": 1, "lap_number": 1, "lap_duration": 90.0},
            {"session_key": 1, "driver_number": 1, "lap_number": 2, "lap_duration": 91.0},
        ]
    )

    result = add_temporal_features(shuffled).sort_values("lap_number")

    assert result.lap_time_prev_1.tolist()[1:] == [90.0, 91.0]


# -- anti-leakage guards, proven to fail when they should ------------------


def feature_frame() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {"session_key": 1, "driver_number": 1, "lap_number": lap, "lap_duration": 90.0 + lap,
             "is_last_lap_for_driver": lap == 4}
            for lap in (1, 2, 3, 4)
        ]
    )
    with_target, _ = add_target(rows)
    return add_temporal_features(with_target)


def test_the_guards_pass_on_a_correctly_built_frame() -> None:
    frame = feature_frame()

    results = checks.run_all(frame, ["lap_duration", "lap_time_prev_1"])

    assert checks.blocking_failures(results) == []


def test_the_forbidden_column_guard_fails_when_a_leaky_column_is_offered() -> None:
    result = checks.check_no_forbidden_features(["lap_duration", "pit_duration"])

    assert result.passed is False
    assert result.details["forbidden_present"] == ["pit_duration"]


def test_the_target_guard_catches_a_target_taken_from_another_driver() -> None:
    frame = feature_frame()
    frame.loc[frame.lap_number == 2, TARGET_COLUMN] = 999.0

    result = checks.check_target_independently(frame)

    assert result.passed is False
    assert result.details["mismatched_rows"] == 1


def test_the_last_lap_guard_catches_an_invented_target() -> None:
    frame = feature_frame()
    frame.loc[frame.is_last_lap_for_driver, TARGET_COLUMN] = 95.0

    result = checks.check_last_lap_has_no_target(frame)

    assert result.passed is False
    assert result.details["last_laps_with_target"] == 1


def test_the_rolling_guard_catches_a_window_that_saw_the_future() -> None:
    """A centred window is the classic mistake; the guard must reject it."""
    frame = feature_frame()
    frame["lap_time_roll_mean_3"] = (
        frame.sort_values("lap_number")["lap_duration"].astype("float64").rolling(3, center=True, min_periods=1).mean()
    )

    result = checks.check_rolling_uses_only_the_past(frame)

    assert result.passed is False
    assert result.details["mismatch_count"] > 0


def test_the_correlation_guard_flags_a_feature_that_is_the_target() -> None:
    frame = feature_frame()
    frame["copy_of_the_target"] = frame[TARGET_COLUMN]

    result = checks.check_correlation_with_target(frame, ["copy_of_the_target"], threshold=0.99)

    assert result.passed is False
    assert result.details["flagged"][0]["feature"] == "copy_of_the_target"
    assert result.blocking is False, "a diagnostic, not a rule that drops features"


def test_the_grain_guard_catches_duplicates() -> None:
    frame = pd.concat([feature_frame(), feature_frame()], ignore_index=True)

    result = checks.check_grain_unique(frame)

    assert result.passed is False
    assert result.details["duplicate_rows"] == 4


def test_a_session_that_fails_a_guard_is_not_written(lap_dataset, feature_settings, tmp_path, monkeypatch) -> None:
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0, 91.0, 92.0]}))
    monkeypatch.setattr(
        checks,
        "check_grain_unique",
        lambda frame: checks.CheckResult("grain_unique", False, "forced failure for the test"),
    )

    report = FeatureRunner(feature_settings()).run()

    assert report.summary()["sessions_with_blocking_failures"] == [9472]
    assert not list((tmp_path / "features").rglob("*.parquet")), "a leaking dataset must not be published"


# -- the catalogue --------------------------------------------------------


def test_no_forbidden_column_is_classified_as_a_feature() -> None:
    """Guards the catalogue itself against a future edit."""
    assert selection.catalogue_conflicts() == []


def test_the_leaky_columns_are_the_ones_the_evidence_named() -> None:
    leaky = {item.name for item in selection.CATALOGUE if item.role is FeatureRole.LEAKY}

    assert "pit_duration" in leaky
    assert "pit_lane_duration" in leaky
    assert "pit_in_lap" not in leaky, "pit entry precedes the end of lap t, so it is observable"


def test_every_catalogue_entry_carries_a_reason() -> None:
    without_reason = [item.name for item in selection.CATALOGUE if not item.reason.strip()]

    assert without_reason == []


def test_excluded_columns_do_not_reach_the_output(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())

    frame = build(feature_settings())

    for column in ("pit_duration", "pit_lane_duration", "is_last_lap_for_driver", "has_complete_window"):
        assert column not in frame.columns, column


# -- output, missing values and reproducibility ---------------------------


def test_categoricals_stay_categorical_without_being_encoded(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())

    frame = build(feature_settings())

    for column in ("compound", "team_name", "driver_acronym"):
        assert str(frame[column].dtype) == "category", column
    assert not [c for c in frame.columns if c.startswith("compound_")], "no one-hot encoding in M5"


def test_nulls_are_kept_rather_than_imputed(lap_dataset, feature_settings) -> None:
    rows = lap_rows(drivers=(1,), durations={1: [90.0, 91.0, 92.0]})
    for row in rows:
        row["speed_mean"] = None
    lap_dataset(rows)

    frame = build(feature_settings())

    assert frame.speed_mean.isna().all(), "M5 must not fill in a missing measurement"
    assert pd.isna(frame.sort_values("lap_number").lap_time_prev_1.iloc[0])


def test_traceability_columns_survive(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())

    frame = build(feature_settings())

    for column in ("year", "meeting_key", "session_key", "driver_number", "lap_number", "session_date"):
        assert column in frame.columns, column
    assert frame.session_date.notna().all()


def test_the_grain_is_unique(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())

    frame = build(feature_settings())

    assert not frame.duplicated(subset=["session_key", "driver_number", "lap_number"]).any()


def test_two_runs_produce_the_same_dataset(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())

    first = build(feature_settings())
    second = build(feature_settings(overwrite=True))

    pd.testing.assert_frame_equal(first, second)


def test_the_report_records_the_decisions(lap_dataset, feature_settings, tmp_path) -> None:
    lap_dataset(lap_rows())

    FeatureRunner(feature_settings()).run()

    files = list((tmp_path / "feature_reports").glob("feature_engineering_report_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["pipeline_name"] == "feature_engineering"
    assert payload["target"]["definition"] == "lap_duration(t+1)"
    assert payload["rolling_windows"]["windows"] == [3, 5]
    assert payload["rolling_windows"]["includes_current_lap"] is True
    assert "does not impute" in payload["missing_value_policy"]
    excluded = {item["column"]: item for item in payload["excluded_columns"]}
    assert excluded["pit_duration"]["role"] == "leaky"
    assert "0.989" in excluded["pit_duration"]["reason"]


# -- edge cases -----------------------------------------------------------


def test_an_empty_input_directory_produces_no_dataset(feature_settings) -> None:
    report = FeatureRunner(feature_settings()).run()

    assert report.summary()["sessions_processed"] == 0
    assert report.summary()["rows_out"] == 0


def test_a_race_of_a_single_lap_yields_a_row_without_a_target(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows(drivers=(1,), durations={1: [90.0]}))

    frame = build(feature_settings())

    assert len(frame) == 1
    assert bool(frame.has_target.iloc[0]) is False
    assert frame.lap_time_roll_mean_3.iloc[0] == 90.0


def test_a_session_already_built_is_skipped(lap_dataset, feature_settings) -> None:
    lap_dataset(lap_rows())
    settings = feature_settings()
    FeatureRunner(settings).run()

    report = FeatureRunner(settings).run()

    assert report.summary()["sessions_skipped"] == 1
    assert report.summary()["sessions_processed"] == 0


def test_the_consolidated_dataset_is_not_modified(lap_dataset, feature_settings, tmp_path) -> None:
    path = lap_dataset(lap_rows())
    before = path.read_bytes()

    FeatureRunner(feature_settings()).run()

    assert path.read_bytes() == before
