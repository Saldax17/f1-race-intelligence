import json
from pathlib import Path
from typing import Callable

import pytest

from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFileError


def test_discover_returns_nothing_when_the_tree_does_not_exist(tmp_path: Path) -> None:
    catalog = RawDataCatalog(base_path=tmp_path / "missing")

    assert catalog.exists() is False
    assert catalog.discover() == []


def test_discover_recovers_partition_keys_from_the_path(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    write_raw("laps", [], meeting_key=1141, session_key=7953)

    found = RawDataCatalog(base_path=raw_root).discover()

    assert len(found) == 1
    raw_file = found[0]
    assert raw_file.endpoint == "laps"
    assert raw_file.partition == {"year": 2023, "meeting_key": 1141, "session_key": 7953}


def test_discover_recovers_the_driver_from_per_driver_file_names(
    raw_root: Path, write_raw: Callable[..., Path]
) -> None:
    write_raw("car_data", [], meeting_key=1141, session_key=7953, file_name="driver_44.json")

    raw_file = RawDataCatalog(base_path=raw_root).discover()[0]

    assert raw_file.driver_number == 44
    assert raw_file.partition["driver_number"] == 44


def test_discover_is_ordered_deterministically(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    write_raw("laps", [], meeting_key=1141, session_key=7953)
    write_raw("meetings", [])
    write_raw("car_data", [], meeting_key=1141, session_key=7953, file_name="driver_44.json")
    write_raw("car_data", [], meeting_key=1141, session_key=7953, file_name="driver_1.json")

    catalog = RawDataCatalog(base_path=raw_root)
    first = [(item.endpoint, item.path.name) for item in catalog.discover()]
    second = [(item.endpoint, item.path.name) for item in catalog.discover()]

    assert first == second
    assert first == [
        ("car_data", "driver_1.json"),
        ("car_data", "driver_44.json"),
        ("laps", "session_7953.json"),
        ("meetings", "meetings.json"),
    ]


def test_discover_can_be_limited_to_specific_endpoints(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    write_raw("laps", [], meeting_key=1141, session_key=7953)
    write_raw("weather", [], meeting_key=1141, session_key=7953)

    found = RawDataCatalog(base_path=raw_root).discover(endpoints=["laps"])

    assert [item.endpoint for item in found] == ["laps"]


def test_discover_skips_leftovers_from_an_interrupted_write(
    raw_root: Path, write_raw: Callable[..., Path]
) -> None:
    path = write_raw("laps", [], meeting_key=1141, session_key=7953)
    (path.parent / ".tmp-half-written.json").write_text("{", encoding="utf-8")

    found = RawDataCatalog(base_path=raw_root).discover()

    assert [item.path.name for item in found] == ["session_7953.json"]


def test_load_returns_the_envelope(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    write_raw("laps", [{"lap_number": 1}], meeting_key=1141, session_key=7953)
    catalog = RawDataCatalog(base_path=raw_root)

    envelope = catalog.load(catalog.discover()[0])

    assert envelope.endpoint == "laps"
    assert envelope.parameters["session_key"] == 7953
    assert envelope.records == [{"lap_number": 1}]
    assert envelope.retrieved_at is not None


def test_load_rejects_invalid_json(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    path = write_raw("laps", [], meeting_key=1141, session_key=7953)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RawFileError, match="Invalid JSON"):
        RawDataCatalog(base_path=raw_root).load(path)


def test_load_rejects_a_file_that_is_not_an_envelope(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    path = write_raw("laps", [], meeting_key=1141, session_key=7953)
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(RawFileError, match="envelope"):
        RawDataCatalog(base_path=raw_root).load(path)


def test_records_is_empty_when_the_payload_is_not_a_list(raw_root: Path, write_raw: Callable[..., Path]) -> None:
    write_raw("laps", {"unexpected": "shape"}, meeting_key=1141, session_key=7953)
    catalog = RawDataCatalog(base_path=raw_root)

    envelope = catalog.load(catalog.discover()[0])

    assert envelope.records == []
    assert envelope.data == {"unexpected": "shape"}
