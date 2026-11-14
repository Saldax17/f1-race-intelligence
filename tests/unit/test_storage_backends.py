"""The storage contract, checked identically on local disk and on (stubbed) S3.

Every test in the first half runs against both backends: a stage written
against :class:`StorageBackend` must not be able to tell them apart. The S3
backend talks to an in-memory stub — no boto3, no network, no credentials.
"""

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from f1_race_intelligence.storage.backends import local as local_backend
from f1_race_intelligence.storage.backends import LocalStorageBackend, S3StorageBackend, StorageError, join_key
from f1_race_intelligence.storage.backends.base import relative_parts
from f1_race_intelligence.storage.store import DataStore, store_from_storage_settings, store_from_uri
from f1_race_intelligence.config.settings import StorageSettings
from s3_fakes import FakeClientError, FakeS3Client

BUCKET = "f1-test-bucket"


@pytest.fixture(params=["local", "s3"])
def backend(request, tmp_path: Path):
    if request.param == "local":
        return LocalStorageBackend(tmp_path)
    return S3StorageBackend(BUCKET, "dev", client=FakeS3Client())


# -- the shared contract ----------------------------------------------------


def test_bytes_round_trip_and_existence(backend) -> None:
    assert backend.exists("raw/a.json") is False

    backend.write_bytes("raw/a.json", b"payload")

    assert backend.exists("raw/a.json") is True
    assert backend.read_bytes("raw/a.json") == b"payload"
    assert backend.size("raw/a.json") == len(b"payload")


def test_a_write_replaces_the_whole_object(backend) -> None:
    backend.write_bytes("k.bin", b"first version, longer")
    backend.write_bytes("k.bin", b"second")

    assert backend.read_bytes("k.bin") == b"second"


def test_reading_a_missing_key_raises_file_not_found(backend) -> None:
    with pytest.raises(FileNotFoundError):
        backend.read_bytes("nowhere.json")


def test_listing_is_recursive_sorted_and_relative_to_the_backend(backend) -> None:
    for key in ("raw/b/2.json", "raw/a/1.json", "raw/top.json", "other/x.json"):
        backend.write_bytes(key, b"{}")

    assert backend.list("raw") == ["raw/a/1.json", "raw/b/2.json", "raw/top.json"]
    assert backend.list("raw", recursive=False) == ["raw/top.json"]
    assert backend.list("missing") == []


def test_a_listing_prefix_is_a_directory_not_a_stem(backend) -> None:
    backend.write_bytes("raw/laps/x.json", b"{}")
    backend.write_bytes("raw/laps_extra/y.json", b"{}")

    assert backend.list("raw/laps") == ["raw/laps/x.json"]


def test_hidden_objects_are_never_listed(backend) -> None:
    backend.write_bytes("raw/.tmp-leftover.json", b"{}")
    backend.write_bytes("raw/real.json", b"{}")

    assert backend.list("raw") == ["raw/real.json"]


def test_has_prefix(backend) -> None:
    assert backend.has_prefix("raw") is False
    backend.write_bytes("raw/deep/file.json", b"{}")
    assert backend.has_prefix("raw") is True
    assert backend.has_prefix("raw/deep") is True


def test_json_round_trip(backend) -> None:
    payload = {"endpoint": "laps", "data": [{"lap_number": 1, "name": "Pérez"}]}

    backend.write_json("raw/laps.json", payload)

    assert backend.read_json("raw/laps.json") == payload


# -- JSON bytes: one representation on every backend and operating system -----

JSON_PAYLOAD = {
    "endpoint": "laps",
    "driver": "Pérez",
    "nested": {"values": [1, 2.5, None, True], "empty": {}},
    "data": [{"lap_number": 1}, {"lap_number": 2}],
}


def canonical_json(payload) -> bytes:
    """The agreed JSON representation, written out here rather than imported.

    UTF-8 without a BOM, two-space indent, LF between lines, no trailing
    newline. Stating it independently means the backends are compared with
    the contract, not with their own implementation.
    """
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def assert_canonical_json(stored: bytes, payload) -> None:
    assert stored == canonical_json(payload)
    assert b"\r" not in stored, "no CRLF (or lone CR) line endings"
    assert stored.count(b"\n") > 0, "lines are separated by LF"
    assert not stored.startswith(b"\xef\xbb\xbf"), "no byte order mark"
    assert not stored.endswith(b"\n"), "no trailing newline"
    assert "Pérez".encode("utf-8") in stored, "non-ASCII text is stored as UTF-8, not escaped"
    assert json.loads(stored.decode("utf-8")) == payload


def test_json_is_stored_as_the_canonical_bytes(backend) -> None:
    backend.write_json("raw/laps.json", JSON_PAYLOAD)

    assert_canonical_json(backend.read_bytes("raw/laps.json"), JSON_PAYLOAD)


def test_unique_json_is_stored_as_the_canonical_bytes(backend) -> None:
    key = backend.write_json_unique("manifests", "manifest_x", JSON_PAYLOAD)

    assert_canonical_json(backend.read_bytes(key), JSON_PAYLOAD)


def test_local_disk_and_s3_store_identical_json_bytes(tmp_path: Path) -> None:
    """Compared byte for byte, straight from each medium, with nothing normalised."""
    client = FakeS3Client()
    local, s3 = LocalStorageBackend(tmp_path), S3StorageBackend(BUCKET, "dev", client=client)

    for store in (local, s3):
        store.write_json("reports/replaceable.json", JSON_PAYLOAD)
        assert store.write_json_unique("reports", "unique", JSON_PAYLOAD) == "reports/unique.json"

    for name in ("replaceable.json", "unique.json"):
        on_disk = (tmp_path / "reports" / name).read_bytes()
        on_s3 = client.objects[BUCKET][f"dev/reports/{name}"]
        assert on_disk == on_s3, name


def test_json_written_with_crlf_by_an_earlier_version_is_still_readable(backend) -> None:
    """Files written on Windows before the fix carry CRLF; they must stay readable and untouched."""
    payload = {"endpoint": "laps", "driver": "Pérez", "data": [{"lap_number": 1}]}
    historical = canonical_json(payload).replace(b"\n", b"\r\n")
    assert b"\r\n" in historical
    backend.write_bytes("raw/old.json", historical)

    assert backend.read_json("raw/old.json") == payload
    assert backend.read_bytes("raw/old.json") == historical, "reading never rewrites a historical file"


def test_write_if_absent_never_replaces(backend) -> None:
    assert backend.write_bytes_if_absent("records/r.json", b"one") is True
    assert backend.write_bytes_if_absent("records/r.json", b"two") is False
    assert backend.read_bytes("records/r.json") == b"one"


def test_format_writers_can_publish_once_without_replacing_existing_objects(backend) -> None:
    first_frame = pd.DataFrame({"value": [1, 2]})
    second_frame = pd.DataFrame({"value": [9]})

    assert backend.write_json_if_absent("runs/a/report.json", {"run": 1}) is True
    assert backend.write_json_if_absent("runs/a/report.json", {"run": 2}) is False
    assert backend.read_bytes("runs/a/report.json") == canonical_json({"run": 1})

    assert backend.write_parquet_if_absent("runs/a/data.parquet", first_frame) is True
    assert backend.write_parquet_if_absent("runs/a/data.parquet", second_frame) is False
    pd.testing.assert_frame_equal(backend.read_parquet("runs/a/data.parquet"), first_frame)

    assert backend.write_joblib_if_absent("runs/a/model.joblib", {"run": 1}) is True
    assert backend.write_joblib_if_absent("runs/a/model.joblib", {"run": 2}) is False
    assert backend.read_joblib("runs/a/model.joblib") == {"run": 1}


def test_unique_json_names_get_suffixes_instead_of_overwriting(backend) -> None:
    first = backend.write_json_unique("manifests", "manifest_20260101T000000000000Z", {"run": 1})
    second = backend.write_json_unique("manifests", "manifest_20260101T000000000000Z", {"run": 2})

    assert first == "manifests/manifest_20260101T000000000000Z.json"
    assert second == "manifests/manifest_20260101T000000000000Z-1.json"
    assert backend.read_json(first) == {"run": 1}
    assert backend.read_json(second) == {"run": 2}
    assert backend.read_bytes(first) == canonical_json({"run": 1}), "the first record is left exactly as written"


def test_parquet_round_trip_keeps_dtypes_and_invents_no_partition_columns(backend) -> None:
    frame = pd.DataFrame(
        {
            "lap_number": pd.array([1, 2, None], dtype="Int64"),
            "lap_duration": pd.array([90.1, None, 91.3], dtype="Float64"),
            "has_target": pd.array([True, False, None], dtype="boolean"),
            "compound": pd.Series(["SOFT", "HARD", "SOFT"], dtype="category"),
        }
    )
    # A Hive-style directory: reading it by path could add "year" and "session".
    key = "consolidated/year=2023/session=7953/session_7953.parquet"

    written = backend.write_parquet(key, frame)
    loaded = backend.read_parquet(key)

    assert written == backend.size(key)
    assert list(loaded.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(loaded, frame)


def test_joblib_round_trip(backend) -> None:
    backend.write_joblib("modeling/preprocessing/preprocessor.joblib", {"median": 91.5})

    assert backend.read_joblib("modeling/preprocessing/preprocessor.joblib") == {"median": 91.5}


# -- local specifics --------------------------------------------------------


def test_local_location_is_a_path_as_before(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    backend.write_bytes("raw/a.json", b"{}")

    assert backend.location("raw/a.json") == tmp_path / "raw" / "a.json"
    assert backend.uri("raw/a.json") == str(tmp_path / "raw" / "a.json")


def test_local_absolute_keys_stay_absolute(tmp_path: Path) -> None:
    backend = LocalStorageBackend(".")
    absolute = join_key(tmp_path, "data", "file.json")

    backend.write_bytes(absolute, b"x")

    assert (tmp_path / "data" / "file.json").read_bytes() == b"x"
    assert backend.list(join_key(tmp_path, "data")) == [absolute]


def test_local_failed_write_leaves_the_previous_file_and_no_temporary(tmp_path: Path, monkeypatch) -> None:
    backend = LocalStorageBackend(tmp_path)
    backend.write_bytes("raw/a.json", b"original")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        backend.write_bytes("raw/a.json", b"replacement")

    assert [item.name for item in (tmp_path / "raw").iterdir()] == ["a.json"]
    assert (tmp_path / "raw" / "a.json").read_bytes() == b"original"


def test_local_failed_json_write_leaves_the_previous_file_and_no_temporary(tmp_path: Path, monkeypatch) -> None:
    """JSON gets the same all-or-nothing replacement as any other object."""
    backend = LocalStorageBackend(tmp_path)
    backend.write_json("raw/a.json", {"version": "original"})
    original = (tmp_path / "raw" / "a.json").read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        backend.write_json("raw/a.json", {"version": "replacement"})

    assert [item.name for item in (tmp_path / "raw").iterdir()] == ["a.json"]
    assert (tmp_path / "raw" / "a.json").read_bytes() == original


def test_local_unique_json_is_created_exclusively(tmp_path: Path) -> None:
    """A name already taken on disk — by anyone — is never replaced."""
    backend = LocalStorageBackend(tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "manifest_x.json").write_bytes(b"written by another process")

    key = backend.write_json_unique("manifests", "manifest_x", {"run": 2})

    assert key == "manifests/manifest_x-1.json"
    assert (tmp_path / "manifests" / "manifest_x.json").read_bytes() == b"written by another process"
    assert sorted(item.name for item in (tmp_path / "manifests").iterdir()) == ["manifest_x-1.json", "manifest_x.json"]


def test_local_unique_json_write_failure_leaves_no_partial_file_or_temporary(
    tmp_path: Path, monkeypatch
) -> None:
    backend = LocalStorageBackend(tmp_path)
    real_fdopen = local_backend.os.fdopen
    directory = tmp_path / "manifests"
    partial_contents: list[bytes] = []

    class FailingBinaryFile:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def write(self, data):
            self._handle.write(data[:2])
            self._handle.flush()
            temp_files = [path for path in directory.iterdir() if path.name.startswith(".tmp-")]
            assert len(temp_files) == 1
            partial_contents.append(temp_files[0].read_bytes())
            raise OSError("disk full")

    def failing_fdopen(descriptor, mode):
        return FailingBinaryFile(real_fdopen(descriptor, mode))

    monkeypatch.setattr(local_backend.os, "fdopen", failing_fdopen)

    with pytest.raises(OSError, match="disk full"):
        backend.write_json_unique("manifests", "manifest_x", {"run": 1})

    assert partial_contents == [b"{\n"]
    assert not (directory / "manifest_x.json").exists()
    assert [path.name for path in directory.iterdir()] == []


def test_local_write_bytes_if_absent_allows_only_one_concurrent_publisher(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    barrier = Barrier(2)

    def publish(data: bytes) -> bool:
        barrier.wait(timeout=5)
        return backend.write_bytes_if_absent("records/shared.bin", data)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (b"writer one", b"writer two")))

    assert sorted(results) == [False, True]
    assert backend.read_bytes("records/shared.bin") in {b"writer one", b"writer two"}
    assert [path.name for path in (tmp_path / "records").iterdir()] == ["shared.bin"]


def test_local_write_bytes_if_absent_preserves_default_create_permissions(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    expected_mode_path = tmp_path / "ordinary-create.bin"
    with expected_mode_path.open("xb") as handle:
        handle.write(b"ordinary")

    backend.write_bytes_if_absent("records/created.bin", b"atomic")

    actual_mode = stat.S_IMODE((tmp_path / "records" / "created.bin").stat().st_mode)
    expected_mode = stat.S_IMODE(expected_mode_path.stat().st_mode)
    assert actual_mode == expected_mode


def test_local_unique_json_fails_safely_when_hard_link_publication_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    backend = LocalStorageBackend(tmp_path)

    def unsupported(*args, **kwargs):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(local_backend.os, "link", unsupported)

    with pytest.raises(OSError, match="hard links unsupported"):
        backend.write_json_unique("manifests", "manifest_x", {"run": 1})

    directory = tmp_path / "manifests"
    assert not (directory / "manifest_x.json").exists()
    assert list(directory.iterdir()) == []


# -- S3 specifics -------------------------------------------------------------


def test_s3_keys_live_under_bucket_and_prefix() -> None:
    client = FakeS3Client()
    backend = S3StorageBackend(BUCKET, "/dev/", client=client)

    backend.write_json("raw/x.json", {"a": 1})

    assert client.keys(BUCKET) == ["dev/raw/x.json"]
    assert client.content_types["dev/raw/x.json"] == "application/json"
    assert backend.uri("raw/x.json") == "s3://f1-test-bucket/dev/raw/x.json"
    assert backend.location("raw/x.json") == "s3://f1-test-bucket/dev/raw/x.json"
    assert backend.local_path("raw/x.json") is None


def test_s3_without_prefix_uses_the_bucket_root() -> None:
    client = FakeS3Client()
    backend = S3StorageBackend(BUCKET, client=client)

    backend.write_bytes("raw/x.json", b"{}")

    assert client.keys(BUCKET) == ["raw/x.json"]
    assert backend.list("raw") == ["raw/x.json"]


def test_s3_listing_follows_pagination() -> None:
    client = FakeS3Client(page_size=2)
    backend = S3StorageBackend(BUCKET, "dev", client=client)
    keys = [f"raw/file_{index:02d}.json" for index in range(7)]
    for key in keys:
        backend.write_bytes(key, b"{}")

    assert backend.list("raw") == keys
    assert client.calls.count("ListObjectsV2") == 4


def test_s3_unique_writes_are_conditional() -> None:
    client = FakeS3Client()
    backend = S3StorageBackend(BUCKET, client=client)

    backend.write_json_unique("manifests", "manifest_x", {})
    backend.write_json_unique("manifests", "manifest_x", {})

    assert client.keys(BUCKET) == ["manifests/manifest_x-1.json", "manifests/manifest_x.json"]


def test_s3_errors_other_than_not_found_are_storage_errors() -> None:
    client = FakeS3Client()
    client.fail_with["GetObject"] = FakeClientError("AccessDenied", 403, "GetObject")
    client.fail_with["HeadObject"] = FakeClientError("AccessDenied", 403, "HeadObject")
    backend = S3StorageBackend(BUCKET, client=client)

    with pytest.raises(StorageError) as read_error:
        backend.read_bytes("raw/x.json")
    with pytest.raises(StorageError):
        backend.exists("raw/x.json")

    # An OSError, so "treat as unreadable" handlers written for disk still apply.
    assert isinstance(read_error.value, OSError)
    assert "AccessDenied" in str(read_error.value)


@pytest.mark.parametrize("key", ["raw/../escape.json", "C:/Users/x.json", "./raw/x.json"])
def test_s3_rejects_keys_that_are_not_logical(key: str) -> None:
    backend = S3StorageBackend(BUCKET, client=FakeS3Client())

    with pytest.raises(ValueError):
        backend.write_bytes(key, b"{}")


@pytest.mark.parametrize("bucket", ["", "s3://bucket", "bucket/prefix"])
def test_s3_rejects_invalid_bucket_names(bucket: str) -> None:
    with pytest.raises(ValueError):
        S3StorageBackend(bucket, client=FakeS3Client())


def test_s3_backend_does_not_need_boto3_when_a_client_is_given() -> None:
    backend = S3StorageBackend(BUCKET, client=FakeS3Client())

    backend.write_bytes("k", b"v")

    assert backend.read_bytes("k") == b"v"


# -- keys and stores --------------------------------------------------------


def test_join_key_normalises_separators_and_keeps_absolute_roots() -> None:
    assert join_key("data\\raw", "/laps/", "", None, "x.json") == "data/raw/laps/x.json"
    assert join_key("/tmp/data", "raw") == "/tmp/data/raw"
    assert relative_parts("data/raw", "data/raw/laps/x.json") == ["laps", "x.json"]
    assert relative_parts("data/raw", "data/rawer/x.json") is None


def test_store_from_uri() -> None:
    s3_store = store_from_uri("s3://bucket-name/some/prefix", client=FakeS3Client())
    local_store = store_from_uri("data/lake", layout="lake")

    assert s3_store.backend.uri("raw") == "s3://bucket-name/some/prefix/raw"
    assert s3_store.layout.name == "lake"
    assert local_store.backend.name == "local"
    assert local_store.layout.name == "lake"
    assert store_from_uri("data").layout.name == "legacy"
    with pytest.raises(ValueError):
        store_from_uri("gs://elsewhere/prefix")


def test_store_from_settings_defaults_to_the_historical_local_tree() -> None:
    store = store_from_storage_settings(StorageSettings())

    assert store.backend.name == "local"
    assert store.layout.name == "legacy"
    assert store.backend.uri("data/raw") == str(Path("data/raw"))
    assert DataStore.local().describe() == {"backend": "local", "location": ".", "layout": "legacy"}


def test_store_from_settings_for_s3_and_for_a_local_lake() -> None:
    s3_store = store_from_storage_settings(
        StorageSettings(backend="s3", bucket="bucket-name", prefix="prod"), client=FakeS3Client()
    )
    local_lake = store_from_storage_settings(StorageSettings(layout="lake"))

    assert s3_store.layout.name == "lake"
    assert s3_store.backend.uri("") == "s3://bucket-name/prod"
    assert local_lake.backend.uri("raw") == str(Path("data/lake/raw"))
