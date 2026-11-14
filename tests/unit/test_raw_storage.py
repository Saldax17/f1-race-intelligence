import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from f1_race_intelligence.storage import raw_storage
from f1_race_intelligence.storage.backends import base as storage_base
from f1_race_intelligence.storage.backends import local as local_backend
from f1_race_intelligence.storage.raw_storage import RawDataStorage


class _FrozenClock:
    """Stands in for ``datetime`` so ``now()`` always returns the same instant."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None) -> datetime:
        return self._moment


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


def test_save_with_explicit_file_name_is_deterministic(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    first = storage.save("laps", {"session_key": 9158}, data=[1], file_name="session_9158.json")
    second = storage.save("laps", {"session_key": 9158}, data=[2], file_name="session_9158.json")

    assert first == second
    assert first.name == "session_9158.json"
    assert json.loads(first.read_text(encoding="utf-8"))["data"] == [2]


def test_save_without_file_name_keeps_the_legacy_timestamped_name(tmp_path: Path) -> None:
    """Callers from M1 that pass no file name keep their original behaviour."""
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("laps", {"session_key": 9158}, data=[])

    assert re.fullmatch(r"\d{8}T\d{12}Z\.json", path.name)


def test_saves_on_the_same_clock_tick_do_not_overwrite_each_other(tmp_path: Path, monkeypatch) -> None:
    """The clock can repeat: two saves must still produce two files.

    Windows' clock granularity is ~15 ms, so consecutive calls really do
    get the same timestamp. Freezing it makes that deterministic here.
    """
    frozen = datetime(2026, 3, 5, 12, 0, 0, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(raw_storage, "datetime", _FrozenClock(frozen))
    storage = RawDataStorage(base_path=tmp_path)

    first = storage.save("laps", {"session_key": 9158}, data=[{"lap_number": 1}])
    second = storage.save("laps", {"session_key": 9158}, data=[{"lap_number": 2}])

    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["data"] == [{"lap_number": 1}]
    assert json.loads(second.read_text(encoding="utf-8"))["data"] == [{"lap_number": 2}]
    assert second.name == "20260305T120000123456Z-1.json"


def test_save_with_file_name_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("laps", {"session_key": 9158}, data=[1], file_name="session_9158.json")
    storage.save("laps", {"session_key": 9158}, data=[2], file_name="session_9158.json")

    assert [item.name for item in path.parent.iterdir()] == ["session_9158.json"]


@pytest.mark.parametrize(
    "module, attribute",
    [
        # Before anything reaches the disk: the payload cannot be serialised.
        (storage_base, "_json_bytes"),
        # After the temporary file is fully written: it cannot be moved into place.
        (os, "replace"),
    ],
    ids=["while-serialising", "while-replacing"],
)
def test_a_failed_write_leaves_neither_a_partial_nor_a_temporary_file(
    tmp_path: Path, monkeypatch, module, attribute
) -> None:
    storage = RawDataStorage(base_path=tmp_path)
    storage.save("laps", {"session_key": 9158}, data=["original"], file_name="session_9158.json")
    target = tmp_path / "laps" / "session_key=9158" / "session_9158.json"
    original = target.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, attribute, explode)

    with pytest.raises(OSError):
        storage.save("laps", {"session_key": 9158}, data=["replacement"], file_name="session_9158.json")

    assert [item.name for item in target.parent.iterdir()] == ["session_9158.json"]
    assert json.loads(target.read_text(encoding="utf-8"))["data"] == ["original"]
    assert target.read_bytes() == original


def test_a_failure_while_writing_the_binary_temporary_file_keeps_the_original(tmp_path: Path, monkeypatch) -> None:
    """The temporary file exists before its binary write fails and is then removed."""
    storage = RawDataStorage(base_path=tmp_path)
    storage.save("laps", {"session_key": 9158}, data=["original"], file_name="session_9158.json")
    target = tmp_path / "laps" / "session_key=9158" / "session_9158.json"
    original = target.read_bytes()
    real_named_temporary_file = local_backend.tempfile.NamedTemporaryFile

    class FailingTemporaryFile:
        def __init__(self, handle):
            self._handle = handle

        @property
        def name(self):
            return self._handle.name

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def write(self, data):
            raise OSError("disk full")

    def failing_named_temporary_file(*args, **kwargs):
        return FailingTemporaryFile(real_named_temporary_file(*args, **kwargs))

    monkeypatch.setattr(local_backend.tempfile, "NamedTemporaryFile", failing_named_temporary_file)

    with pytest.raises(OSError, match="disk full"):
        storage.save("laps", {"session_key": 9158}, data=["replacement"], file_name="session_9158.json")

    assert [item.name for item in target.parent.iterdir()] == ["session_9158.json"]
    assert target.read_bytes() == original


@pytest.mark.parametrize("file_name", ["session_9158.json", None], ids=["named", "timestamped"])
def test_saved_files_use_lf_line_endings_on_every_operating_system(tmp_path: Path, file_name) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    path = storage.save("laps", {"session_key": 9158}, data=[{"driver": "Pérez"}], file_name=file_name)

    stored = path.read_bytes()
    assert b"\r" not in stored
    assert stored.count(b"\n") > 0
    assert not stored.startswith(b"\xef\xbb\xbf") and not stored.endswith(b"\n")
    assert "Pérez".encode("utf-8") in stored


def test_a_raw_file_saved_with_crlf_by_an_earlier_version_still_loads_and_is_left_alone(tmp_path: Path) -> None:
    """Raw files downloaded on Windows before the fix carry CRLF and are never rewritten."""
    storage = RawDataStorage(base_path=tmp_path)
    target = storage.resolve_path("laps", {"session_key": 9158}, file_name="session_9158.json")
    target.parent.mkdir(parents=True)
    historical = b'{\r\n  "endpoint": "laps",\r\n  "data": [\r\n    {\r\n      "lap_number": 1\r\n    }\r\n  ]\r\n}'
    target.write_bytes(historical)

    assert storage.exists("laps", {"session_key": 9158}, file_name="session_9158.json")
    assert storage.load("laps", {"session_key": 9158}, file_name="session_9158.json") == [{"lap_number": 1}]
    assert target.read_bytes() == historical


def test_resolve_path_points_at_where_save_would_write_without_creating_it(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    resolved = storage.resolve_path("laps", {"year": 2023, "session_key": 9158}, file_name="session_9158.json")

    assert not resolved.exists()
    assert resolved == storage.save("laps", {"year": 2023, "session_key": 9158}, data=[], file_name="session_9158.json")


def test_load_returns_the_saved_data_payload(tmp_path: Path) -> None:
    storage = RawDataStorage(base_path=tmp_path)
    storage.save("drivers", {"session_key": 9158}, data=[{"driver_number": 1}], file_name="session_9158.json")

    loaded = storage.load("drivers", {"session_key": 9158}, file_name="session_9158.json")

    assert loaded == [{"driver_number": 1}]


@pytest.mark.parametrize("file_name", ["../escape.json", "nested/file.json", "", ".."])
def test_file_names_that_would_escape_the_directory_are_rejected(tmp_path: Path, file_name: str) -> None:
    storage = RawDataStorage(base_path=tmp_path)

    with pytest.raises(ValueError):
        storage.save("laps", {}, data=[], file_name=file_name)
