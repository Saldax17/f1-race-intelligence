"""The path contract: the legacy tree must not move, and the lake keys are an API."""

import pytest

from f1_race_intelligence.storage.layout import LAKE_ROOTS, LakeLayout, Layer, LegacyLayout, RawKey, get_layout

SESSION = {"year": 2023, "meeting_key": 1141, "session_key": 7953}


# -- legacy: exactly what the project has always written ----------------------


def test_legacy_raw_keys_are_the_historical_tree() -> None:
    layout = LegacyLayout()
    root = layout.root(Layer.RAW, "data/raw")

    assert layout.raw_key(root, "laps", SESSION, "session_7953.json") == (
        "data/raw/laps/year=2023/meeting_key=1141/session_key=7953/session_7953.json"
    )
    assert layout.raw_key(root, "meetings", {"year": 2023}, "meetings.json") == "data/raw/meetings/year=2023/meetings.json"
    assert layout.raw_directory(root, "/sessions", {}) == "data/raw/sessions"


def test_legacy_datasets_use_plain_season_directories() -> None:
    layout = LegacyLayout()

    assert layout.session_dataset_key("data/processed/lap_dataset", 2024, 9472) == (
        "data/processed/lap_dataset/2024/session_9472.parquet"
    )
    assert layout.session_dataset_key("out", None, 9472) == "out/session_9472.parquet"
    assert layout.split_prefix("data/modeling", "train") == "data/modeling/train"
    assert layout.preprocessing_prefix("data/modeling") == "data/modeling/preprocessing"


def test_legacy_ignores_scope_for_records_and_honours_configured_roots() -> None:
    layout = LegacyLayout()

    assert layout.root(Layer.EXTRACTION_MANIFESTS, "data/manifests") == "data/manifests"
    assert layout.records_prefix("data/validation", year=2023, session_key=7953) == "data/validation"


# -- lake: the S3 contract ------------------------------------------------------


def test_lake_roots_are_the_documented_layers() -> None:
    layout = LakeLayout()

    assert {layer: layout.root(layer, "ignored/path") for layer in Layer} == LAKE_ROOTS
    assert LAKE_ROOTS[Layer.RAW] == "raw"
    assert LAKE_ROOTS[Layer.VALIDATION_REPORTS] == "validated"
    assert LAKE_ROOTS[Layer.CONSOLIDATED] == "consolidated"
    assert LAKE_ROOTS[Layer.FEATURES] == "features"
    assert LAKE_ROOTS[Layer.MODELING] == "modeling"
    assert LAKE_ROOTS[Layer.MODELS] == "models"
    assert LAKE_ROOTS[Layer.LOGS] == "logs"
    assert LAKE_ROOTS[Layer.EXTRACTION_MANIFESTS].startswith("manifests/")


def test_lake_raw_keys_put_the_whole_session_under_one_prefix() -> None:
    layout = LakeLayout()

    assert layout.raw_key("raw", "laps", SESSION, "session_7953.json") == (
        "raw/year=2023/meeting=1141/session=7953/endpoint=laps/session_7953.json"
    )
    assert layout.raw_key("raw", "car_data", {**SESSION, "driver_number": 44}, "driver_44.json") == (
        "raw/year=2023/meeting=1141/session=7953/endpoint=car_data/driver_44.json"
    )
    assert layout.raw_key("raw", "meetings", {"year": 2023}, "meetings.json") == "raw/year=2023/endpoint=meetings/meetings.json"
    assert layout.raw_key("raw", "sessions", {"year": 2023, "meeting_key": 1141}, "sessions.json") == (
        "raw/year=2023/meeting=1141/endpoint=sessions/sessions.json"
    )


def test_lake_dataset_split_and_record_keys() -> None:
    layout = LakeLayout()

    assert layout.session_dataset_key("consolidated", 2023, 7953) == "consolidated/year=2023/session=7953/session_7953.parquet"
    assert layout.session_dataset_key("features", 2024, 9472) == "features/year=2024/session=9472/session_9472.parquet"
    assert layout.split_prefix("modeling", "validation") == "modeling/split=validation"
    assert layout.records_prefix("validated", year=2023, session_key=7953) == "validated/year=2023/session=7953"
    assert layout.records_prefix("validated") == "validated"


def test_lake_listing_prefix_narrows_only_on_contiguous_keys() -> None:
    layout = LakeLayout()

    assert layout.raw_listing_prefix("raw") == "raw"
    assert layout.raw_listing_prefix("raw", year=2023) == "raw/year=2023"
    assert layout.raw_listing_prefix("raw", year=2023, meeting_key=1141, session_key=7953) == (
        "raw/year=2023/meeting=1141/session=7953"
    )
    # Without the meeting the session cannot be addressed directly.
    assert layout.raw_listing_prefix("raw", year=2023, session_key=7953) == "raw/year=2023"
    assert layout.session_dataset_listing_prefix("features", 2025) == "features/year=2025"


# -- parsing is the inverse of building ------------------------------------------


@pytest.mark.parametrize("layout_name,root", [("legacy", "data/raw"), ("lake", "raw")])
@pytest.mark.parametrize(
    "endpoint,partition,file_name",
    [
        ("laps", SESSION, "session_7953.json"),
        ("car_data", SESSION, "driver_1.json"),
        ("sessions", {"year": 2023, "meeting_key": 1141}, "sessions.json"),
        ("meetings", {"year": 2023}, "meetings.json"),
    ],
)
def test_raw_keys_parse_back_to_their_partition(layout_name, root, endpoint, partition, file_name) -> None:
    layout = get_layout(layout_name)

    parsed = layout.parse_raw_key(root, layout.raw_key(root, endpoint, partition, file_name))

    assert parsed == RawKey(endpoint=endpoint, **partition)


@pytest.mark.parametrize("layout_name,root", [("legacy", "data/features/lap_features"), ("lake", "features")])
def test_dataset_keys_parse_back(layout_name, root) -> None:
    layout = get_layout(layout_name)

    assert layout.parse_session_dataset_key(root, layout.session_dataset_key(root, 2024, 9472)) == (2024, 9472)
    assert layout.parse_session_dataset_key(root, f"{root}/2024/notes.parquet") is None
    assert layout.parse_session_dataset_key("elsewhere", layout.session_dataset_key(root, 2024, 9472)) is None


def test_a_key_outside_the_root_or_without_an_endpoint_is_not_raw() -> None:
    assert LakeLayout().parse_raw_key("raw", "raw/year=2023/meetings.json") is None
    assert LegacyLayout().parse_raw_key("data/raw", "data/raw/orphan.json") is None
    assert LegacyLayout().parse_raw_key("data/raw", "other/laps/x.json") is None


def test_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_layout("hive")
