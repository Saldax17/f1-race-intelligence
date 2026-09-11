"""Tests for M6: the M5 feature dataset in, train/validation/test out.

The guards are tested from both sides. Showing that they pass on a
correctly built dataset proves nothing on its own — a guard that always
passed would do the same — so each one is also handed a dataset that leaks
and is required to fail.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.features.selection import TARGET_COLUMN
from f1_race_intelligence.modeling import checks, preprocessing
from f1_race_intelligence.modeling.loaders import FeatureDatasetLoader, feature_columns
from f1_race_intelligence.modeling.runner import KEY_PREFIX, ModelingRunner
from f1_race_intelligence.modeling.split import (
    TEST,
    TRAIN,
    VALIDATION,
    SessionAssignment,
    SplitStrategy,
    apply_assignments,
    assign_sessions,
)

RACE_START = pd.Timestamp("2023-03-05T15:00:00Z")


def feature_rows(
    session_key: int,
    year: int,
    day_offset: int,
    drivers: tuple = (1, 44),
    laps: int = 5,
    compound: str = "SOFT",
    team: str = "Red Bull",
) -> List[Dict[str, Any]]:
    """Rows shaped like M5's output, for one race."""
    start = RACE_START + pd.Timedelta(days=day_offset)
    rows = []
    for driver in drivers:
        for lap in range(1, laps + 1):
            rows.append(
                {
                    "year": year,
                    "meeting_key": 1000 + session_key,
                    "session_key": session_key,
                    "driver_number": driver,
                    "lap_number": lap,
                    "session_date": start,
                    "lap_duration": 90.0 + lap + driver * 0.1,
                    "tyre_age": lap,
                    "position_at_lap_start": 1,
                    "speed_mean": 200.0 + lap,
                    "gap_to_leader_s": None if lap == 1 else float(lap),
                    "lap_time_prev_1": None if lap == 1 else 90.0 + lap - 1,
                    "compound": compound,
                    "team_name": team,
                    "driver_acronym": f"D{driver}",
                    "is_pit_out_lap": False,
                    "pit_in_lap": False,
                    "is_lapped": False,
                    "next_lap_time": 91.0 + lap + driver * 0.1 if lap < laps else None,
                    "has_target": lap < laps,
                }
            )
    return rows


@pytest.fixture
def feature_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Write an M5-style dataset to disk."""

    def _write(rows: List[Dict[str, Any]], year: int, session_key: int) -> Path:
        base = tmp_path / "lap_features" / str(year)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"session_{session_key}.parquet"
        frame = pd.DataFrame(rows)
        for column in ("compound", "team_name", "driver_acronym"):
            frame[column] = frame[column].astype("category")
        frame.to_parquet(path, engine="pyarrow", index=False)
        return path

    return _write


@pytest.fixture
def three_races(feature_dataset) -> Callable[[], None]:
    """Three races on different days, so a 3-way split is possible."""

    def _build() -> None:
        feature_dataset(feature_rows(1, 2023, 0), 2023, 1)
        feature_dataset(feature_rows(2, 2024, 365), 2024, 2)
        feature_dataset(feature_rows(3, 2025, 730), 2025, 3)

    return _build


@pytest.fixture
def modeling_settings(tmp_path: Path) -> Callable[..., AppSettings]:
    def _settings(**overrides: Any) -> AppSettings:
        config: Dict[str, Any] = {
            "input_path": str(tmp_path / "lap_features"),
            "output_path": str(tmp_path / "modeling"),
            "report_path": str(tmp_path / "modeling" / "reports"),
            "train_fraction": 1 / 3,
            "validation_fraction": 1 / 3,
            "min_sessions_per_split": 1,
        }
        config.update(overrides)
        return AppSettings(modeling=config)

    return _settings


def load_matrix(settings, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(settings.modeling.output_path) / split / "features.parquet", engine="pyarrow")


# -- loading and target ---------------------------------------------------


def test_rows_without_a_target_are_dropped_and_counted(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()

    # 3 races x 2 drivers x 5 laps = 30 rows, of which the last lap of each
    # driver has no target.
    assert report.source["rows_in"] == 30
    assert report.source["rows_dropped_without_target"] == 6
    assert report.source["rows_out"] == 24


def test_the_target_is_never_part_of_the_feature_matrix(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()

    assert TARGET_COLUMN not in report.preprocessing["output_feature_names"]
    assert report.aborted is None


# -- the split ------------------------------------------------------------


def test_the_split_is_chronological(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()

    assert report.split[TRAIN]["years"] == [2023]
    assert report.split[VALIDATION]["years"] == [2024]
    assert report.split[TEST]["years"] == [2025]


def test_a_race_is_never_divided_between_splits(three_races, modeling_settings) -> None:
    """Laps of one race share a track, weather and safety cars."""
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()

    keys = {}
    for split in (TRAIN, VALIDATION, TEST):
        frame = pd.read_parquet(Path(settings.modeling.output_path) / split / "dataset.parquet")
        keys[split] = set(frame["session_key"].unique())
    assert keys[TRAIN] & keys[VALIDATION] == set()
    assert keys[TRAIN] & keys[TEST] == set()
    assert keys[VALIDATION] & keys[TEST] == set()


def test_splitting_by_year_assigns_whole_seasons(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings(
        split_strategy="years", train_years=[2023], validation_years=[2024], test_years=[2025]
    )

    report = ModelingRunner(settings).run()

    assert report.split[TRAIN]["session_keys"] == [1]
    assert report.split[TEST]["session_keys"] == [3]


def season_calendar(year: int, races: int, first_day: int, first_key: int) -> pd.DataFrame:
    """A season of `races` races, two weeks apart."""
    rows = []
    for index in range(races):
        rows.extend(feature_rows(first_key + index, year, first_day + index * 14, laps=2))
    return pd.DataFrame(rows)


def test_train_years_then_split_gives_whole_seasons_to_train_and_halves_the_rest() -> None:
    """Learn from the closed seasons, hold out the most recent one in two."""
    calendar = pd.concat(
        [
            season_calendar(2023, 4, 0, 100),
            season_calendar(2024, 4, 400, 200),
            season_calendar(2025, 4, 800, 300),
        ],
        ignore_index=True,
    )

    assignments = assign_sessions(
        calendar, strategy=SplitStrategy.TRAIN_YEARS_THEN_SPLIT, train_years=[2023, 2024]
    )
    by_split: Dict[str, List[int]] = {}
    for item in assignments:
        by_split.setdefault(item.split, []).append(item.session_key)

    assert sorted(by_split[TRAIN]) == [100, 101, 102, 103, 200, 201, 202, 203]
    assert sorted(by_split[VALIDATION]) == [300, 301], "the earlier half of 2025"
    assert sorted(by_split[TEST]) == [302, 303], "the later half of 2025"


def test_an_odd_number_of_remaining_races_leaves_the_extra_one_in_validation() -> None:
    calendar = pd.concat(
        [season_calendar(2023, 2, 0, 100), season_calendar(2025, 5, 800, 300)], ignore_index=True
    )

    assignments = assign_sessions(
        calendar, strategy=SplitStrategy.TRAIN_YEARS_THEN_SPLIT, train_years=[2023]
    )
    counts = {split: sum(1 for i in assignments if i.split == split) for split in (TRAIN, VALIDATION, TEST)}

    assert counts == {TRAIN: 2, VALIDATION: 3, TEST: 2}


def test_train_years_then_split_stays_chronological() -> None:
    calendar = pd.concat(
        [season_calendar(2023, 2, 0, 100), season_calendar(2025, 4, 800, 300)], ignore_index=True
    )

    assignments = assign_sessions(
        calendar, strategy=SplitStrategy.TRAIN_YEARS_THEN_SPLIT, train_years=[2023]
    )

    assert checks.check_chronological_split(assignments).passed is True


def test_train_years_then_split_needs_training_seasons() -> None:
    calendar = season_calendar(2025, 2, 0, 300)

    with pytest.raises(ValueError, match="needs train_years"):
        assign_sessions(calendar, strategy=SplitStrategy.TRAIN_YEARS_THEN_SPLIT, train_years=[])


def test_train_years_then_split_rejects_seasons_that_are_not_there() -> None:
    calendar = season_calendar(2025, 2, 0, 300)

    with pytest.raises(ValueError, match="No session belongs"):
        assign_sessions(calendar, strategy=SplitStrategy.TRAIN_YEARS_THEN_SPLIT, train_years=[2019])


def test_a_season_in_two_splits_is_a_configuration_error() -> None:
    frame = pd.DataFrame(feature_rows(1, 2023, 0))

    with pytest.raises(ValueError, match="more than one split"):
        assign_sessions(frame, strategy=SplitStrategy.YEARS, train_years=[2023], validation_years=[2023])


def test_a_season_nobody_assigned_is_a_configuration_error() -> None:
    frame = pd.DataFrame(feature_rows(1, 2023, 0))

    with pytest.raises(ValueError, match="not named in the split configuration"):
        assign_sessions(frame, strategy=SplitStrategy.YEARS, train_years=[2024])


def test_sessions_are_ordered_by_date_not_by_key() -> None:
    """A later race with a lower key must still come later."""
    frame = pd.concat(
        [pd.DataFrame(feature_rows(9, 2023, 0)), pd.DataFrame(feature_rows(2, 2024, 365))], ignore_index=True
    )

    assignments = assign_sessions(frame, train_fraction=0.5, validation_fraction=0.5)

    by_split = {item.split: item.session_key for item in assignments}
    assert by_split[TRAIN] == 9, "the 2023 race trains even though its key is higher"
    assert by_split[VALIDATION] == 2


# -- preprocessing --------------------------------------------------------


def test_preprocessing_statistics_come_from_train_alone(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()
    check = next(c for c in report.checks if c["check"] == "preprocessing_fitted_on_train_only")

    assert check["status"] == "PASS"


def test_missing_values_are_imputed_and_recorded(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()
    matrix = load_matrix(settings, TRAIN)

    features = [c for c in matrix.columns if not c.startswith(KEY_PREFIX) and c != TARGET_COLUMN]
    assert matrix[features].isna().sum().sum() == 0, "no nulls survive into the model input"
    assert any(name.startswith("missingindicator_") for name in features), "the absence itself is kept"


def test_categoricals_are_one_hot_encoded(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()
    names = report.preprocessing["output_feature_names"]

    assert "compound_SOFT" in names
    assert "compound" not in names


def test_a_category_unseen_in_training_does_not_break_transform(feature_dataset, modeling_settings) -> None:
    """A driver who debuts in a later season must not break the pipeline."""
    feature_dataset(feature_rows(1, 2023, 0, drivers=(1, 44)), 2023, 1)
    feature_dataset(feature_rows(2, 2024, 365, drivers=(1, 99), compound="WET", team="New Team"), 2024, 2)
    settings = modeling_settings(train_fraction=0.5, validation_fraction=0.5)

    report = ModelingRunner(settings).run()

    assert report.aborted is None
    check = next(c for c in report.checks if c["check"] == "categorical_unknown_handling")
    assert check["status"] == "PASS"
    matrix = load_matrix(settings, VALIDATION)
    assert "compound_WET" not in matrix.columns, "a category never seen in training gets no column"
    assert matrix["compound_SOFT"].sum() == 0, "the unseen compound encodes to zeros"


def test_a_key_that_is_also_a_feature_is_not_overwritten(three_races, modeling_settings) -> None:
    """lap_number identifies a row and also predicts; both must survive."""
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()
    matrix = load_matrix(settings, TRAIN)

    assert abs(float(matrix["lap_number"].mean())) < 1e-9, "the feature is standardized"
    assert int(matrix[f"{KEY_PREFIX}lap_number"].min()) == 1, "the raw key is kept separately"


def test_scaling_can_be_turned_off(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings(scale_numeric=False)

    report = ModelingRunner(settings).run()

    assert report.preprocessing["scaling"] == "none"
    assert load_matrix(settings, TRAIN)["tyre_age"].min() == 1.0


# -- guards, proven to fail when they should ------------------------------


def built_frame() -> pd.DataFrame:
    rows = pd.DataFrame(feature_rows(1, 2023, 0))
    return rows[rows["has_target"]].reset_index(drop=True)


def test_the_target_guard_rejects_the_target_as_a_feature() -> None:
    result = checks.check_target_not_in_features(["lap_duration", TARGET_COLUMN])

    assert result.passed is False
    assert result.details["found"] == [TARGET_COLUMN]


def test_the_forbidden_guard_rejects_a_leaky_column() -> None:
    result = checks.check_forbidden_columns_not_in_features(["lap_duration", "pit_duration"])

    assert result.passed is False
    assert "pit_duration" in result.details["found"]


def test_the_forbidden_guard_rejects_an_identifier() -> None:
    result = checks.check_forbidden_columns_not_in_features(["lap_duration", "session_key"])

    assert result.passed is False


def test_lap_number_is_allowed_because_m5_approves_it() -> None:
    """It is a key and a feature; the catalogue decides."""
    result = checks.check_forbidden_columns_not_in_features(["lap_number", "lap_duration"])

    assert result.passed is True


def test_the_traceability_guard_rejects_a_key_column_among_the_features() -> None:
    result = checks.check_traceability_columns_not_in_model_features(
        ["lap_duration", f"{KEY_PREFIX}session_key"]
    )

    assert result.passed is False
    assert result.details["keys_found_among_features"] == [f"{KEY_PREFIX}session_key"]


def test_the_traceability_guard_rejects_an_identifier_among_the_features() -> None:
    result = checks.check_traceability_columns_not_in_model_features(["lap_duration", "session_date"])

    assert result.passed is False
    assert "session_date" in result.details["identifiers_found_among_features"]


def test_the_traceability_guard_rejects_a_matrix_column_nobody_declared() -> None:
    """Anything in the file that is neither a feature, the target nor a key."""
    result = checks.check_traceability_columns_not_in_model_features(
        ["lap_duration"], matrix_columns=["lap_duration", TARGET_COLUMN, "a_stray_column"]
    )

    assert result.passed is False
    assert result.details["matrix_columns_unaccounted_for"] == ["a_stray_column"]


def test_the_traceability_guard_accepts_keys_that_stay_out_of_the_features() -> None:
    result = checks.check_traceability_columns_not_in_model_features(
        ["lap_duration", "lap_number"],
        matrix_columns=["lap_duration", "lap_number", TARGET_COLUMN, f"{KEY_PREFIX}session_key"],
    )

    assert result.passed is True, "lap_number is a key and an M5-approved feature"


def test_keys_are_present_in_the_matrix_but_not_among_the_model_features(
    three_races, modeling_settings
) -> None:
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()
    matrix = load_matrix(settings, TRAIN)
    declared = report.preprocessing["output_feature_names"]

    assert [c for c in matrix.columns if c.startswith(KEY_PREFIX)], "keys travel with the data"
    assert not [c for c in declared if c.startswith(KEY_PREFIX)], "but are not model inputs"
    assert TARGET_COLUMN not in declared


def test_the_overlap_guard_rejects_a_session_in_two_splits() -> None:
    assignments = [
        SessionAssignment(session_key=1, year=2023, session_date=RACE_START, split=TRAIN),
        SessionAssignment(session_key=1, year=2023, session_date=RACE_START, split=VALIDATION),
    ]

    result = checks.check_no_session_overlap_between_splits(assignments)

    assert result.passed is False
    assert "1" in result.details["overlapping"]


def test_the_row_overlap_guard_rejects_a_lap_in_two_splits() -> None:
    frame = built_frame()
    splits = {TRAIN: frame, VALIDATION: frame.copy(), TEST: frame.iloc[0:0]}

    result = checks.check_split_rows_do_not_overlap(splits)

    assert result.passed is False
    assert result.details["collisions"]


def test_the_chronology_guard_rejects_validation_before_train() -> None:
    assignments = [
        SessionAssignment(session_key=1, year=2024, session_date=pd.Timestamp("2024-05-01T00:00:00Z"), split=TRAIN),
        SessionAssignment(session_key=2, year=2023, session_date=pd.Timestamp("2023-05-01T00:00:00Z"), split=VALIDATION),
    ]

    result = checks.check_chronological_split(assignments)

    assert result.passed is False
    assert result.details["violations"][0]["earlier"] == TRAIN


def test_the_fitting_guard_rejects_a_transformer_fitted_on_train_plus_validation() -> None:
    """The classic mistake: fit on everything, then split."""
    train = pd.DataFrame({"speed_mean": [1.0, 2.0, 3.0], "compound": ["SOFT", "SOFT", "HARD"]})
    validation = pd.DataFrame({"speed_mean": [100.0, 200.0], "compound": ["SOFT", "HARD"]})

    contaminated = preprocessing.build_preprocessor(["speed_mean"], ["compound"])
    contaminated.fit(pd.concat([train, validation], ignore_index=True))

    result = checks.check_no_validation_statistics_used_in_training(contaminated, train, validation)

    assert result.passed is False
    assert result.details["matches_train_only_fit"] is False


def test_the_fitting_guard_rejects_a_transformer_fitted_on_test() -> None:
    train = pd.DataFrame({"speed_mean": [1.0, 2.0, 3.0], "compound": ["SOFT", "SOFT", "HARD"]})
    test = pd.DataFrame({"speed_mean": [500.0, 600.0], "compound": ["SOFT", "HARD"]})

    contaminated = preprocessing.build_preprocessor(["speed_mean"], ["compound"])
    contaminated.fit(pd.concat([train, test], ignore_index=True))

    result = checks.check_no_test_statistics_used_in_training(contaminated, train, test)

    assert result.passed is False


def test_the_fitting_guard_accepts_a_transformer_fitted_on_train() -> None:
    train = pd.DataFrame({"speed_mean": [1.0, 2.0, 3.0], "compound": ["SOFT", "SOFT", "HARD"]})
    validation = pd.DataFrame({"speed_mean": [100.0, 200.0], "compound": ["SOFT", "HARD"]})

    correct = preprocessing.build_preprocessor(["speed_mean"], ["compound"])
    correct.fit(train)

    result = checks.check_no_validation_statistics_used_in_training(correct, train, validation)

    assert result.passed is True
    assert result.details["adding_split_changes_parameters"] is True, "the comparison can discriminate"


def test_the_schema_guard_rejects_mismatched_columns() -> None:
    left = pd.DataFrame({"a": [1.0], "b": [2.0]})
    right = pd.DataFrame({"b": [2.0], "a": [1.0]})

    result = checks.check_consistent_feature_schema({TRAIN: left, VALIDATION: right})

    assert result.passed is False
    assert result.details["mismatched"][VALIDATION]["same_set_different_order"] is True


def test_the_alignment_guard_rejects_a_shifted_target() -> None:
    frame = built_frame()
    shifted = frame[TARGET_COLUMN].shift(1).fillna(0.0)

    result = checks.check_target_alignment({TRAIN: frame}, {TRAIN: shifted}, frame)

    assert result.passed is False


def test_a_run_that_fails_a_guard_writes_nothing(three_races, modeling_settings, tmp_path, monkeypatch) -> None:
    three_races()
    settings = modeling_settings()
    monkeypatch.setattr(
        checks,
        "check_chronological_split",
        lambda assignments: checks.CheckResult("chronological_split", False, "forced failure"),
    )

    report = ModelingRunner(settings).run()

    assert report.aborted is not None
    assert not list((tmp_path / "modeling").rglob("*.parquet")), "a leaking dataset must not be published"


# -- report, artifacts and reproducibility --------------------------------


def test_the_report_records_the_configuration_and_the_split(three_races, modeling_settings, tmp_path) -> None:
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()

    files = list((tmp_path / "modeling" / "reports").glob("modeling_report_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["pipeline_name"] == "modeling_dataset"
    assert payload["config"]["split_strategy"] == "fraction"
    assert payload["preprocessing"]["fitted_on"] == "train split only"
    assert payload["target"]["definition"].startswith("lap_duration(t+1)")
    assert payload["split"][TRAIN]["sessions"] == 1
    assert all(item["status"] == "PASS" for item in payload["checks"])


def test_the_fitted_preprocessor_is_persisted(three_races, modeling_settings, tmp_path) -> None:
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()

    directory = tmp_path / "modeling" / "preprocessing"
    assert (directory / "preprocessor.joblib").exists()
    schema = list(directory.glob("feature_schema*.json"))
    assert schema, "the encoded column names are written for the next stage"
    payload = json.loads(schema[0].read_text(encoding="utf-8"))
    assert payload["target"] == TARGET_COLUMN
    assert payload["key_columns_in_matrix"] == [f"{KEY_PREFIX}{c}" for c in ("session_key", "driver_number", "lap_number")]


def test_two_runs_produce_the_same_matrices(three_races, modeling_settings) -> None:
    three_races()
    settings = modeling_settings()

    ModelingRunner(settings).run()
    first = load_matrix(settings, TRAIN)
    ModelingRunner(settings).run()
    second = load_matrix(settings, TRAIN)

    pd.testing.assert_frame_equal(first, second)


def test_a_thin_history_is_reported_as_insufficient(feature_dataset, modeling_settings) -> None:
    """Two races cannot support a three-way split, and the report says so."""
    feature_dataset(feature_rows(1, 2023, 0), 2023, 1)
    feature_dataset(feature_rows(2, 2024, 365), 2024, 2)
    settings = modeling_settings(min_sessions_per_split=3, train_fraction=0.5, validation_fraction=0.5)

    report = ModelingRunner(settings).run()

    assert report.warnings
    assert any("below the configured minimum" in item for item in report.warnings)
    assert report.aborted is None, "the pipeline still runs; it just refuses to pretend"


def test_an_empty_input_aborts_without_writing(modeling_settings, tmp_path) -> None:
    report = ModelingRunner(modeling_settings()).run()

    assert report.aborted is not None
    assert not list((tmp_path / "modeling").rglob("*.parquet"))


def test_the_feature_dataset_is_not_modified(three_races, modeling_settings, tmp_path) -> None:
    three_races()
    source = sorted((tmp_path / "lap_features").rglob("*.parquet"))
    before = {path: path.read_bytes() for path in source}

    ModelingRunner(modeling_settings()).run()

    assert {path: path.read_bytes() for path in source} == before


def test_features_come_from_the_m5_catalogue(three_races, modeling_settings) -> None:
    """M6 does not invent features; it consumes what M5 approved."""
    three_races()
    settings = modeling_settings()

    report = ModelingRunner(settings).run()

    frame = pd.read_parquet(Path(settings.modeling.output_path) / TRAIN / "dataset.parquet")
    columns = feature_columns(frame)
    assert set(report.features["numeric"]) == set(columns["numeric"])
    assert set(report.features["categorical"]) == set(columns["categorical"])
    assert report.features["source"].startswith("M5 catalogue")
