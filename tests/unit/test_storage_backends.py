"""The storage contract, checked identically on local disk and on (stubbed) S3.

Every test in the first half runs against both backends: a stage written
against :class:`StorageBackend` must not be able to tell them apart. The S3
backend talks to an in-memory stub — no boto3, no network, no credentials.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

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


def test_write_if_absent_never_replaces(backend) -> None:
    assert backend.write_bytes_if_absent("records/r.json", b"one") is True
    assert backend.write_bytes_if_absent("records/r.json", b"two") is False
    assert backend.read_bytes("records/r.json") == b"one"


def test_unique_json_names_get_suffixes_instead_of_overwriting(backend) -> None:
    first = backend.write_json_unique("manifests", "manifest_20260101T000000000000Z", {"run": 1})
    second = backend.write_json_unique("manifests", "manifest_20260101T000000000000Z", {"run": 2})

    assert first == "manifests/manifest_20260101T000000000000Z.json"
    assert second == "manifests/manifest_20260101T000000000000Z-1.json"
    assert backend.read_json(first) == {"run": 1}
    assert backend.read_json(second) == {"run": 2}


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
