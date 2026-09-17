"""M2-M6 against the storage abstraction, on the S3 data lake layout.

S3 is an in-memory stub: no boto3, no network, no AWS account. The central
claim tested here is that changing where data lives changes nothing about
what the stages compute — so the S3 runs are compared against the same
stages run on local disk in the historical layout.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.consolidation.runner import ConsolidationRunner
from f1_race_intelligence.features.runner import FeatureRunner
from f1_race_intelligence.modeling.runner import ModelingRunner
from f1_race_intelligence.modeling.split import TEST, TRAIN, VALIDATION
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.pipelines.manifest import ExtractionStatus
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends import LocalStorageBackend, S3StorageBackend
from f1_race_intelligence.storage.layout import LakeLayout
from f1_race_intelligence.storage.migration import copy_raw
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.validation.manifest_index import ManifestIndex
from f1_race_intelligence.validation.runner import ValidationRunner
from factories import MEETING_KEY, SESSION_KEY, YEAR, car_data_record
from s3_fakes import FakeS3Client
from test_consolidation import build_raw  # noqa: F401 - fixture
from test_historical_extraction import make_client
from test_modeling import feature_rows

BUCKET = "f1-test-bucket"
PREFIX = "dev"


@pytest.fixture
def s3() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def lake(s3: FakeS3Client) -> DataStore:
    return DataStore(backend=S3StorageBackend(BUCKET, PREFIX, client=s3), layout=LakeLayout())


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch) -> Path:
    """Run from an empty directory, so any stray local write is visible."""
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


def objects(s3: FakeS3Client, prefix: str = "") -> list:
    return [key for key in s3.keys(BUCKET) if key.startswith(f"{PREFIX}/{prefix}")]


def extraction_settings(**overrides: Any) -> AppSettings:
    config: Dict[str, Any] = {"years": [2023], "session_types": ["Race"], "endpoints": ["laps", "car_data"]}
    config.update(overrides)
    return AppSettings(historical_extraction=config)


# -- M2 -----------------------------------------------------------------------


def test_m2_writes_the_lake_contract_to_s3_and_nothing_locally(lake, s3, isolated_cwd) -> None:
    manifest = HistoricalExtractionPipeline(extraction_settings(), client=make_client(), store=lake).run()

    session = "dev/raw/year=2023/meeting=1219/session=9158"
    assert f"{session}/endpoint=laps/session_9158.json" in s3.keys(BUCKET)
    assert f"{session}/endpoint=car_data/driver_1.json" in s3.keys(BUCKET)
    assert f"{session}/endpoint=car_data/driver_44.json" in s3.keys(BUCKET)
    assert "dev/raw/year=2023/endpoint=meetings/meetings.json" in s3.keys(BUCKET)
    assert "dev/raw/year=2023/meeting=1219/endpoint=sessions/sessions.json" in s3.keys(BUCKET)
    assert len(objects(s3, "manifests/extraction/manifest_")) == 1
    assert all(entry.file_path.startswith("s3://f1-test-bucket/dev/raw/") for entry in manifest.entries)
    assert list(isolated_cwd.iterdir()) == []


def test_m2_is_idempotent_on_s3(lake, s3) -> None:
    client = make_client()
    HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run()

    second = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run()

    assert {entry.status for entry in second.entries} == {ExtractionStatus.SKIPPED}
    assert client.get_laps.call_count == 1
    assert client.get_car_data.call_count == 2
    assert client.get_meetings.call_count == 1


def test_m2_overwrite_refetches_on_s3(lake) -> None:
    client = make_client()
    first = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run()

    rerun = HistoricalExtractionPipeline(extraction_settings(overwrite=True), client=client, store=lake).run()

    assert first.summary()["downloaded"] > 0
    assert rerun.summary()["downloaded"] == first.summary()["downloaded"]
    assert rerun.summary()["skipped"] == 0
    assert client.get_laps.call_count == 2


def test_m2_plan_then_one_session_job_needs_no_discovery_calls(lake, s3) -> None:
    client = make_client()
    pipeline = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake)

    units = pipeline.plan([2023])
    manifest = pipeline.run_session(2023, 9158)

    assert units == [{"year": 2023, "meeting_key": 1219, "session_key": 9158, "session_name": "Race"}]
    assert client.get_sessions.call_count == 1, "the session's meeting came from the stored catalogue"
    assert manifest.sessions_processed == [9158]
    assert manifest.config["scope"] == {"mode": "session", "year": 2023, "meeting_key": 1219, "session_key": 9158}
    assert len(objects(s3, "manifests/extraction/year=2023/session=9158/manifest_")) == 1
    assert "dev/raw/year=2023/meeting=1219/session=9158/endpoint=laps/session_9158.json" in s3.keys(BUCKET)


def test_m2_session_job_without_a_catalogue_looks_the_session_up_once(lake) -> None:
    client = make_client()
    client.get_sessions.side_effect = lambda **kwargs: (
        [{"session_key": 9158, "meeting_key": 1219, "year": 2023, "session_name": "Race"}]
        if kwargs.get("session_key") == 9158
        else [{"session_key": 9158, "session_name": "Race", "session_type": "Race"}]
    )

    manifest = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run_session(2023, 9158)

    assert manifest.summary()["failed"] == 0
    assert client.get_meetings.call_count == 0
    assert manifest.meetings_processed == [1219]


def test_m2_session_job_for_an_unknown_session_fails_without_extracting(lake) -> None:
    client = make_client(get_sessions=[])

    manifest = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run_session(2023, 1)

    assert manifest.summary()["failed"] == 1
    assert client.get_laps.call_count == 0


def test_m2_session_job_refuses_a_meeting_that_does_not_hold_the_session(lake) -> None:
    client = make_client()  # the catalogue lists 9158, 9157 and 9156 only

    manifest = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run_session(
        2023, 4242, meeting_key=1219
    )

    assert manifest.summary()["failed"] == 1
    assert client.get_laps.call_count == 0


def test_m2_session_job_refuses_a_session_from_another_year(lake) -> None:
    client = make_client(get_sessions=[{"session_key": 9158, "meeting_key": 1219, "year": 2024}])

    manifest = HistoricalExtractionPipeline(extraction_settings(), client=client, store=lake).run_session(2023, 9158)

    assert manifest.summary()["failed"] == 1
    assert client.get_laps.call_count == 0


# -- moving existing raw data without re-downloading -------------------------------


def test_copy_raw_moves_the_legacy_tree_into_the_lake_idempotently(build_raw, raw_root, lake, s3, tmp_path) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "manifest_20230305T120000000000Z.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    local = DataStore.local()

    first = copy_raw(local, lake, raw_path=str(raw_root), manifest_path=str(manifests))
    second = copy_raw(local, lake, raw_path=str(raw_root), manifest_path=str(manifests))

    session = f"dev/raw/year={YEAR}/meeting={MEETING_KEY}/session={SESSION_KEY}"
    assert f"{session}/endpoint=car_data/driver_1.json" in s3.keys(BUCKET)
    assert "dev/manifests/extraction/manifest_20230305T120000000000Z.json" in s3.keys(BUCKET)
    assert first.copied == len(list(raw_root.rglob("*.json"))) and first.manifests_copied == 1
    assert (second.copied, second.skipped, second.manifests_copied) == (0, first.copied, 0)
    # Byte for byte: the envelope, including retrieved_at, is unchanged.
    original = (raw_root / "laps" / f"year={YEAR}" / f"meeting_key={MEETING_KEY}" / f"session_key={SESSION_KEY}" / f"session_{SESSION_KEY}.json")
    assert s3.objects[BUCKET][f"{session}/endpoint=laps/session_{SESSION_KEY}.json"] == original.read_bytes()


# -- M3 -------------------------------------------------------------------------


def lake_with_raw(build_raw, raw_root, lake) -> DataStore:
    build_raw(car_data={1: [car_data_record(1, 10)], 44: [car_data_record(44, 110)]})
    copy_raw(DataStore.local(), lake, raw_path=str(raw_root), include_manifests=False)
    return lake


def test_m3_on_s3_reaches_the_same_verdict_as_on_disk(build_raw, raw_root, lake, s3, tmp_path) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    local_settings = AppSettings(
        validation={"raw_path": str(raw_root), "manifest_path": str(tmp_path / "m"), "report_path": str(tmp_path / "v")}
    )

    on_disk = ValidationRunner(local_settings).run()
    on_s3 = ValidationRunner(AppSettings(), store=lake).run()

    assert on_s3.global_status == on_disk.global_status
    assert on_s3.summary() == on_disk.summary()
    assert len(objects(s3, "validated/validation_report_")) == 1


def test_m3_session_scope_validates_one_session_and_records_the_filter(build_raw, raw_root, lake, s3) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    # Another session's data must not be validated by this job.
    lake.backend.write_json(
        "raw/year=2023/meeting=1141/session=1/endpoint=laps/session_1.json",
        {"endpoint": "laps", "parameters": {}, "data": "not a list"},
    )
    scope = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=SESSION_KEY)

    report = ValidationRunner(AppSettings(), store=lake, scope=scope).run()

    assert report.source["filter"] == scope.to_dict()
    assert all("session=1/" not in (result.file or "") for result in report.results)
    assert "meetings" in report.scope["endpoints"] and "sessions" in report.scope["endpoints"]
    written = objects(s3, f"validated/year={YEAR}/session={SESSION_KEY}/validation_report_")
    assert len(written) == 1
    assert json.loads(s3.objects[BUCKET][written[0]])["source"]["filter"]["session_key"] == SESSION_KEY


def test_manifest_index_combines_session_jobs_when_no_full_run_exists(lake) -> None:
    for session_key, endpoint_failed in ((1, None), (2, "pit")):
        entries = [{"endpoint": endpoint_failed, "session_key": session_key, "status": "failed", "status_code": 404}] if endpoint_failed else []
        lake.backend.write_json(
            f"manifests/extraction/year=2023/session={session_key}/manifest_2023030{session_key}T000000000000Z.json",
            {"config": {"endpoints": ["laps", "pit"], "scope": {"mode": "session", "session_key": session_key}},
             "sessions_processed": [session_key], "entries": entries},
        )

    combined = ManifestIndex.load_latest("manifests/extraction", backend=lake.backend)
    one = ManifestIndex.load_latest("manifests/extraction", backend=lake.backend, session_key=1)

    assert combined.sessions_processed == [1, 2]
    assert combined.expected_endpoints == ["laps", "pit"]
    assert combined.failure_for("pit", 2)["status_code"] == 404
    assert one.sessions_processed == [1]


# -- M4 -------------------------------------------------------------------------


def consolidation_on_disk(tmp_path: Path, raw_root: Path) -> pd.DataFrame:
    settings = AppSettings(
        consolidation={
            "raw_path": str(raw_root),
            "output_path": str(tmp_path / "local_out"),
            "report_path": str(tmp_path / "local_reports"),
            "validation_path": str(tmp_path / "no_validation"),
        }
    )
    ConsolidationRunner(settings).run()
    return pd.read_parquet(tmp_path / "local_out" / str(YEAR) / f"session_{SESSION_KEY}.parquet")


def test_m4_session_job_on_s3_produces_exactly_the_local_dataset(build_raw, raw_root, lake, s3, tmp_path) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    expected = consolidation_on_disk(tmp_path, raw_root)
    scope = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=SESSION_KEY)

    report = ConsolidationRunner(AppSettings(), store=lake, scope=scope).run()

    key = f"consolidated/year={YEAR}/session={SESSION_KEY}/session_{SESSION_KEY}.parquet"
    assert report.summary()["sessions_processed"] == 1
    assert report.sessions[0].output_path == f"s3://{BUCKET}/{PREFIX}/{key}"
    pd.testing.assert_frame_equal(lake.backend.read_parquet(key), expected)
    assert len(objects(s3, "manifests/consolidation/consolidation_report_")) == 1


def test_m4_session_job_is_idempotent_and_lists_only_its_session(build_raw, raw_root, lake, s3) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    scope = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=SESSION_KEY)
    ConsolidationRunner(AppSettings(), store=lake, scope=scope).run()
    s3.calls.clear()

    again = ConsolidationRunner(AppSettings(), store=lake, scope=scope).run()

    assert again.summary()["sessions_skipped"] == 1
    assert "GetObject" not in s3.calls, "a skipped session reads no raw data"


def test_m4_can_read_s3_and_write_local_disk(build_raw, raw_root, lake, tmp_path) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    output = DataStore(backend=LocalStorageBackend(tmp_path / "out"), layout=LakeLayout())

    ConsolidationRunner(AppSettings(), store=lake, output_store=output, scope=SessionScope(session_key=SESSION_KEY)).run()

    assert (tmp_path / "out" / "consolidated" / f"year={YEAR}" / f"session={SESSION_KEY}" / f"session_{SESSION_KEY}.parquet").exists()


def test_m4_session_job_only_trusts_validation_that_covered_its_session(build_raw, raw_root, lake) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    lake.backend.write_json(
        "validated/year=2023/session=1/validation_report_20990101T000000000000Z.json",
        {"global_status": "PASS", "source": {"filter": {"session_key": 1}}},
    )
    lake.backend.write_json(
        f"validated/year={YEAR}/session={SESSION_KEY}/validation_report_20230101T000000000000Z.json",
        {"global_status": "FAIL", "source": {"filter": {"session_key": SESSION_KEY}}},
    )
    settings = AppSettings(consolidation={"require_validation_pass": True})

    report = ConsolidationRunner(
        settings, store=lake, scope=SessionScope(year=YEAR, session_key=SESSION_KEY)
    ).run()

    assert report.source_validation["status"] == "FAIL"
    assert report.aborted is not None


# -- M5 -------------------------------------------------------------------------


def test_m5_session_job_on_s3_produces_exactly_the_local_features(build_raw, raw_root, lake, tmp_path) -> None:
    lake_with_raw(build_raw, raw_root, lake)
    consolidation_on_disk(tmp_path, raw_root)
    FeatureRunner(
        AppSettings(features={"input_path": str(tmp_path / "local_out"), "output_path": str(tmp_path / "feat"), "report_path": str(tmp_path / "fr")})
    ).run()
    expected = pd.read_parquet(tmp_path / "feat" / str(YEAR) / f"session_{SESSION_KEY}.parquet")
    scope = SessionScope(year=YEAR, session_key=SESSION_KEY)
    ConsolidationRunner(AppSettings(), store=lake, scope=scope).run()

    report = FeatureRunner(AppSettings(), store=lake, scope=scope).run()

    key = f"features/year={YEAR}/session={SESSION_KEY}/session_{SESSION_KEY}.parquet"
    assert report.summary()["sessions_processed"] == 1
    assert report.summary()["sessions_with_blocking_failures"] == []
    pd.testing.assert_frame_equal(lake.backend.read_parquet(key), expected)


# -- M6 -------------------------------------------------------------------------


MODELING = {"train_fraction": 1 / 3, "validation_fraction": 1 / 3, "min_sessions_per_split": 1}


def test_m6_on_s3_writes_the_same_artifacts_as_on_disk(lake, s3, tmp_path) -> None:
    local = tmp_path / "lap_features"
    for session_key, year, offset in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        frame = pd.DataFrame(feature_rows(session_key, year, offset))
        for column in ("compound", "team_name", "driver_acronym"):
            frame[column] = frame[column].astype("category")
        (local / str(year)).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(local / str(year) / f"session_{session_key}.parquet", index=False)
        lake.backend.write_parquet(f"features/year={year}/session={session_key}/session_{session_key}.parquet", frame)

    on_disk = ModelingRunner(
        AppSettings(modeling={**MODELING, "input_path": str(local), "output_path": str(tmp_path / "modeling"), "report_path": str(tmp_path / "mr")})
    ).run()
    on_s3 = ModelingRunner(AppSettings(modeling=MODELING), store=lake).run()

    assert on_s3.aborted is None and on_disk.aborted is None
    assert on_s3.split["assignments"] == on_disk.split["assignments"]
    assert on_s3.checks == on_disk.checks
    for split in (TRAIN, VALIDATION, TEST):
        for name in ("dataset.parquet", "features.parquet"):
            pd.testing.assert_frame_equal(
                lake.backend.read_parquet(f"modeling/split={split}/{name}"),
                pd.read_parquet(tmp_path / "modeling" / split / name),
            )
    preprocessor = lake.backend.read_joblib("modeling/preprocessing/preprocessor.joblib")
    assert list(preprocessor.get_feature_names_out()) == on_disk.preprocessing["output_feature_names"]
    assert objects(s3, "modeling/preprocessing/feature_schema")
    assert len(objects(s3, "manifests/modeling/modeling_report_")) == 1
