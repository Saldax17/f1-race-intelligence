import json
from pathlib import Path

from f1_race_intelligence.storage.raw_storage import RawDataStorage


def test_save_writes_json_file_with_expected_payload(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("sessions", {"year": 2025}, data=[{"session_key": 1}])

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["endpoint"] == "sessions"
    assert payload["parameters"] == {"year": 2025}
    assert payload["data"] == [{"session_key": 1}]
    assert "retrieved_at" in payload


def test_save_partitions_by_year_meeting_and_session(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save(
        "laps",
        {"year": 2025, "meeting_key": 111, "session_key": 222, "driver_number": 1},
        data=[],
    )

    expected_dir = tmp_path / "laps" / "year=2025" / "meeting_key=111" / "session_key=222"
    assert path.parent == expected_dir


def test_save_without_partition_keys_falls_back_to_endpoint_only(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("meetings", {}, data=[])

    assert path.parent == tmp_path / "meetings"


def test_save_strips_leading_slash_from_endpoint(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("/sessions", {}, data=[])

    assert path.parent == tmp_path / "sessions"
