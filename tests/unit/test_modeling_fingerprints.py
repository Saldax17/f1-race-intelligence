"""Tests for the content fingerprints in the M6 report.

Every hash the report states is checked against bytes taken straight from
the medium — the files on disk or the fake bucket's dict — never through
the backend under test, so a backend that hashed the wrong thing could not
agree with itself. S3 is the in-memory stub: no boto3, no network, no AWS.
"""

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling import checks
from f1_race_intelligence.modeling.fingerprints import INPUT_FINGERPRINT_SCHEME, InputFile, input_fingerprint
from f1_race_intelligence.modeling.loaders import FeatureDatasetLoader
from f1_race_intelligence.modeling.runner import ModelingRunner
from f1_race_intelligence.modeling.split import SPLITS
from f1_race_intelligence.storage.backends import LocalStorageBackend, S3StorageBackend
from f1_race_intelligence.storage.backends.base import StorageError
from f1_race_intelligence.storage.layout import LakeLayout, Layer, LegacyLayout
from f1_race_intelligence.storage.store import DataStore
from s3_fakes import FakeS3Client
from test_modeling import feature_rows

BUCKET = "f1-test-bucket"
PREFIX = "dev"

MODELING = {"train_fraction": 1 / 3, "validation_fraction": 1 / 3, "min_sessions_per_split": 1}
RACES = ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730))
DEFAULTS = AppSettings().modeling

EXPECTED_ARTIFACTS = {(split, name) for split in SPLITS for name in ("dataset.parquet", "features.parquet")} | {
    (None, "preprocessor.joblib"),
    (None, "feature_schema.json"),
}


def race_bytes(session_key: int, year: int, day_offset: int, **overrides: Any) -> bytes:
    """One M5 session file, serialised once so every store gets identical bytes."""
    frame = pd.DataFrame(feature_rows(session_key, year, day_offset, **overrides))
    for column in ("compound", "team_name", "driver_acronym"):
        frame[column] = frame[column].astype("category")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


class Env:
    """A store, plus direct access to the medium behind it."""

    def __init__(self, kind: str, tmp_path: Path, backend: Optional[LocalStorageBackend] = None) -> None:
        self.kind = kind
        if kind == "local":
            self.root = tmp_path / "store"
            self.store = DataStore(backend=backend or LocalStorageBackend(self.root), layout=LegacyLayout())
        else:
            self.s3 = FakeS3Client()
            self.store = DataStore(backend=S3StorageBackend(BUCKET, PREFIX, client=self.s3), layout=LakeLayout())

    def input_key(self, year: int, session_key: int) -> str:
        root = self.store.root(Layer.FEATURES, DEFAULTS.input_path)
        return self.store.layout.session_dataset_key(root, year, session_key)

    def publish(self, races=RACES, **overrides: Any) -> Dict[int, bytes]:
        published = {}
        for session_key, year, offset in races:
            data = race_bytes(session_key, year, offset, **overrides)
            self.store.backend.write_bytes(self.input_key(year, session_key), data)
            published[session_key] = data
        return published

    def run(self, **overrides: Any):
        return ModelingRunner(AppSettings(modeling={**MODELING, **overrides}), store=self.store).run()

    def stored(self) -> Dict[str, bytes]:
        """Every object, keyed by the URI the backend reports for it, read from the medium."""
        if self.kind == "local":
            return {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        return {f"s3://{BUCKET}/{key}": data for key, data in self.s3.objects.get(BUCKET, {}).items()}

    def _uri(self, layer: Layer, configured: str) -> str:
        return self.store.backend.uri(self.store.root(layer, configured))

    def artifact_objects(self) -> Dict[str, bytes]:
        """What sits in the modelling layer, execution reports excluded."""
        base = self._uri(Layer.MODELING, DEFAULTS.output_path)
        reports = self._uri(Layer.MODELING_REPORTS, DEFAULTS.report_path)
        return {
            uri: data
            for uri, data in self.stored().items()
            if _below(uri, base) and not _below(uri, reports)
        }

    def reports(self) -> List[Dict[str, Any]]:
        base = self._uri(Layer.MODELING_REPORTS, DEFAULTS.report_path)
        return [json.loads(data) for uri, data in self.stored().items() if _below(uri, base)]


def _below(uri: str, base: str) -> bool:
    if uri.startswith("s3://"):
        return uri.startswith(base.rstrip("/") + "/")
    return Path(uri).is_relative_to(Path(base))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(params=["local", "s3"])
def env(request, tmp_path: Path) -> Env:
    return Env(request.param, tmp_path)


# -- input fingerprints ---------------------------------------------------


def test_each_input_hash_is_the_sha256_of_the_stored_bytes(env: Env) -> None:
    published = env.publish()

    report = env.run()

    assert report.aborted is None
    stored = env.stored()
    inputs = report.source["inputs"]
    assert sorted(item["session_key"] for item in inputs) == [1, 2, 3]
    for item in inputs:
        on_medium = stored[env.store.backend.uri(env.input_key(item["year"], item["session_key"]))]
        assert on_medium == published[item["session_key"]]
        assert item["sha256"] == sha256(on_medium)
        assert item["size_bytes"] == len(on_medium)
        assert item["input"] == f"session_{item['session_key']}.parquet"
    assert {item["year"] for item in inputs} == {2023, 2024, 2025}


def test_input_entries_carry_no_location(env: Env) -> None:
    env.publish()

    report = env.run()

    location = env.store.backend.uri("")
    for item in report.source["inputs"]:
        assert set(item) == {"input", "year", "session_key", "sha256", "size_bytes"}
        assert not any(location in str(value) or "s3://" in str(value) for value in item.values())


def test_the_fingerprint_follows_the_documented_canonical_form() -> None:
    """Pinned so a change to the algorithm cannot happen without a new scheme."""
    inputs = [
        InputFile("session_9.parquet", 2024, 9, "b" * 64, 10),
        InputFile("session_3.parquet", None, 3, "c" * 64, 10),
        InputFile("session_2.parquet", 2024, 2, "a" * 64, 10),
    ]
    document = {
        "inputs": [
            {"input": "session_3.parquet", "year": None, "session_key": 3, "sha256": "c" * 64},
            {"input": "session_2.parquet", "year": 2024, "session_key": 2, "sha256": "a" * 64},
            {"input": "session_9.parquet", "year": 2024, "session_key": 9, "sha256": "b" * 64},
        ],
        "scheme": INPUT_FINGERPRINT_SCHEME,
    }
    expected = sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))

    assert input_fingerprint(inputs) == f"sha256:{expected}"


def test_the_same_inputs_give_the_same_fingerprint(env: Env) -> None:
    env.publish()

    first = env.run()
    second = env.run()

    fingerprint = first.source["input_fingerprint"]
    assert fingerprint.startswith("sha256:") and len(fingerprint) == len("sha256:") + 64
    assert second.source["input_fingerprint"] == fingerprint
    assert second.source["inputs"] == first.source["inputs"]
    saved = env.reports()
    assert len(saved) == 2, "each run still keeps its own report"
    assert {item["source"]["input_fingerprint"] for item in saved} == {fingerprint}
    assert {item["summary"]["input_fingerprint"] for item in saved} == {fingerprint}


def test_changing_one_input_changes_the_fingerprint(env: Env) -> None:
    env.publish()
    before = env.run()

    session_key, year, offset = RACES[1]
    env.store.backend.write_bytes(env.input_key(year, session_key), race_bytes(session_key, year, offset, compound="HARD"))
    after = env.run()

    assert after.source["input_fingerprint"] != before.source["input_fingerprint"]
    changed = {
        item["session_key"]
        for item, previous in zip(after.source["inputs"], before.source["inputs"])
        if item["sha256"] != previous["sha256"]
    }
    assert changed == {session_key}


def test_a_different_session_assignment_changes_the_fingerprint() -> None:
    base = [InputFile("session_1.parquet", 2023, 1, "a" * 64, 1)]
    moved = [InputFile("session_1.parquet", 2024, 1, "a" * 64, 1)]

    assert input_fingerprint(base) != input_fingerprint(moved)


def test_the_order_inputs_are_given_in_does_not_change_the_fingerprint() -> None:
    inputs = [InputFile(f"session_{key}.parquet", 2020 + key % 3, key, f"{key:064x}", key) for key in range(12)]
    shuffled = list(inputs)
    random.Random(7).shuffle(shuffled)

    assert shuffled != inputs
    assert input_fingerprint(shuffled) == input_fingerprint(inputs)
    assert input_fingerprint(reversed(inputs)) == input_fingerprint(inputs)


class ReversedListing(LocalStorageBackend):
    """A backend that enumerates objects in the opposite order."""

    def list(self, prefix: str, *, recursive: bool = True) -> List[str]:
        return list(reversed(super().list(prefix, recursive=recursive)))


def test_the_order_storage_lists_files_in_does_not_change_the_fingerprint(tmp_path: Path) -> None:
    forward = Env("local", tmp_path)
    backward = Env("local", tmp_path, backend=ReversedListing(tmp_path / "store"))
    forward.publish()

    first = forward.run()
    second = backward.run()

    assert second.source["files"] == list(reversed(first.source["files"])), "the listing really was reversed"
    assert second.source["input_fingerprint"] == first.source["input_fingerprint"]


def test_local_disk_and_s3_give_the_same_fingerprint(tmp_path: Path) -> None:
    local, lake = Env("local", tmp_path), Env("s3", tmp_path)
    assert local.publish() == lake.publish()

    on_disk = local.run()
    on_s3 = lake.run()

    assert on_disk.source["files"] != on_s3.source["files"], "the locations really differ"
    assert on_s3.source["inputs"] == on_disk.source["inputs"]
    assert on_s3.source["input_fingerprint"] == on_disk.source["input_fingerprint"]


# -- artifact fingerprints ------------------------------------------------


def test_each_artifact_hash_is_the_sha256_of_the_stored_bytes(env: Env) -> None:
    env.publish()

    report = env.run()

    assert report.aborted is None
    declared = {item.path: item for item in report.artifact_files}
    stored = env.artifact_objects()
    assert set(declared) == set(stored), "every written object is declared, and nothing else"
    for path, item in declared.items():
        assert item.sha256 == sha256(stored[path])
        assert item.size_bytes == len(stored[path])
    assert {(item.split, item.artifact) for item in report.artifact_files} == EXPECTED_ARTIFACTS
    for split in report.artifacts:
        assert {split.dataset_path, split.matrix_path} <= set(declared)


def test_the_saved_report_carries_the_artifact_hashes(env: Env) -> None:
    env.publish()

    report = env.run()

    (saved,) = env.reports()
    assert saved["artifact_files"] == [item.to_dict() for item in report.artifact_files]
    assert saved["source"]["input_fingerprint_scheme"] == INPUT_FINGERPRINT_SCHEME


# -- runs that stop early -------------------------------------------------


def assert_inputs_match_medium(env: Env, report, expected_sessions: List[int]) -> None:
    stored = env.stored()
    inputs = report.source["inputs"]
    assert sorted(item["session_key"] for item in inputs) == expected_sessions
    for item in inputs:
        assert item["sha256"] == sha256(stored[env.store.backend.uri(env.input_key(item["year"], item["session_key"]))])


def assert_aborted_without_artifacts(env: Env, report) -> Dict[str, Any]:
    assert report.aborted is not None
    assert report.artifacts == [] and report.artifact_files == []
    assert env.artifact_objects() == {}, "nothing was written, so nothing may be declared"
    (saved,) = env.reports()
    assert saved["aborted"] == report.aborted
    assert saved["artifact_files"] == [] and saved["artifacts"] == []
    return saved


def test_a_run_stopped_by_a_guard_records_its_inputs_and_no_artifacts(env: Env, monkeypatch) -> None:
    env.publish()
    monkeypatch.setattr(
        checks,
        "check_chronological_split",
        lambda assignments: checks.CheckResult("chronological_split", False, "forced failure"),
    )

    report = env.run()

    saved = assert_aborted_without_artifacts(env, report)
    assert report.blocking_failures == ["chronological_split"]
    assert saved["summary"]["checks_passed"] is False
    assert_inputs_match_medium(env, report, [1, 2, 3])
    assert saved["source"]["input_fingerprint"] == report.source["input_fingerprint"]


def test_an_empty_training_split_records_its_inputs_and_no_artifacts(env: Env) -> None:
    env.publish()

    report = env.run(split_strategy="years", train_years=[], validation_years=[2023, 2024], test_years=[2025])

    assert_aborted_without_artifacts(env, report)
    assert "training split is empty" in report.aborted
    assert_inputs_match_medium(env, report, [1, 2, 3])


def test_files_read_but_filtered_out_are_still_listed(env: Env) -> None:
    """They were read; the filter discarded their rows, not the reading."""
    env.publish()

    report = env.run(years=[2019])

    assert_aborted_without_artifacts(env, report)
    assert report.source["rows_in"] == 0
    assert_inputs_match_medium(env, report, [1, 2, 3])


def test_no_input_files_means_no_inputs_listed(env: Env) -> None:
    report = env.run()

    assert_aborted_without_artifacts(env, report)
    assert report.source["inputs"] == []
    assert report.source["input_fingerprint"] == input_fingerprint([])


class FailingStorage(LocalStorageBackend):
    """Local storage that fails on one chosen key fragment."""

    def __init__(self, root: Path, *, read: str = "", write: str = "") -> None:
        super().__init__(root)
        self._read, self._write = read, write

    def read_bytes(self, key: str) -> bytes:
        if self._read and self._read in key:
            raise StorageError(f"forced read failure for {key}")
        return super().read_bytes(key)

    def write_parquet(self, key: str, frame) -> int:
        if self._write and self._write in key:
            raise StorageError(f"forced write failure for {key}")
        return super().write_parquet(key, frame)


def test_a_failed_read_records_no_inputs_at_all(tmp_path: Path) -> None:
    Env("local", tmp_path).publish()
    failing = Env("local", tmp_path, backend=FailingStorage(tmp_path / "store", read="session_2"))

    with pytest.raises(StorageError):
        failing.run()

    assert failing.reports() == [], "a run that could not read its inputs claims nothing"


def test_a_failed_write_declares_no_artifacts(tmp_path: Path) -> None:
    Env("local", tmp_path).publish()
    failing = Env("local", tmp_path, backend=FailingStorage(tmp_path / "store", write="validation/features"))

    with pytest.raises(StorageError):
        failing.run()

    assert failing.artifact_objects(), "earlier splits did reach storage"
    assert failing.reports() == [], "but no report declares a half-written output"


# -- what must not change -------------------------------------------------


def test_hashing_does_not_change_what_is_loaded(env: Env) -> None:
    """The bytes are parsed once for hashing and modelling; the rows must be the old ones."""
    env.publish()
    loader = FeatureDatasetLoader(DEFAULTS.input_path, store=env.store)
    root = env.store.root(Layer.FEATURES, DEFAULTS.input_path)

    loaded = loader.load()

    keys = env.store.backend.list(env.store.layout.session_dataset_listing_prefix(root))
    reference = pd.concat([env.store.backend.read_parquet(key) for key in keys], ignore_index=True)
    reference = reference[reference["has_target"]].sort_values(
        ["session_key", "driver_number", "lap_number"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded.frame, reference)


def test_the_guards_and_the_split_are_unchanged(env: Env) -> None:
    env.publish()

    report = env.run()

    assert report.blocking_failures == []
    assert all(item["status"] == "PASS" for item in report.checks)
    assert [item["split"] for item in report.split["assignments"]] == list(SPLITS)
    for item in report.artifact_files:
        if item.artifact == "dataset.parquet":
            frame = pd.read_parquet(io.BytesIO(env.artifact_objects()[item.path]))
            assert sorted(frame["session_key"].unique()) == report.split[item.split]["session_keys"]


def test_a_disabled_leakage_stop_still_writes_and_declares_everything(env: Env, monkeypatch) -> None:
    """fail_on_leakage=False keeps its meaning: the failure is reported, the output written."""
    env.publish()
    monkeypatch.setattr(
        checks,
        "check_chronological_split",
        lambda assignments: checks.CheckResult("chronological_split", False, "forced failure"),
    )

    report = env.run(fail_on_leakage=False)

    assert report.aborted is None
    assert report.blocking_failures == ["chronological_split"]
    assert {item.path for item in report.artifact_files} == set(env.artifact_objects())
