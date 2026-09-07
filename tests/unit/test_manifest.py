import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from f1_race_intelligence.pipelines.manifest import ExecutionManifest, ExtractionStatus, ManifestEntry


def make_manifest(**overrides) -> ExecutionManifest:
    kwargs = {
        "pipeline_name": "historical_extraction",
        "pipeline_version": "1.0.0",
        "config": {"years": [2023]},
    }
    kwargs.update(overrides)
    return ExecutionManifest(**kwargs)


def test_entry_serialization_omits_fields_that_do_not_apply() -> None:
    entry = ManifestEntry(endpoint="laps", status=ExtractionStatus.SKIPPED, session_key=9158)

    payload = entry.to_dict()

    assert payload == {"endpoint": "laps", "status": "skipped", "session_key": 9158}


def test_summary_counts_every_status_even_when_unused() -> None:
    manifest = make_manifest()
    manifest.add(ManifestEntry(endpoint="laps", status=ExtractionStatus.DOWNLOADED))
    manifest.add(ManifestEntry(endpoint="weather", status=ExtractionStatus.FAILED))
    manifest.record_session(9158)

    assert manifest.summary() == {
        "downloaded": 1,
        "skipped": 0,
        "failed": 1,
        "meetings": 0,
        "sessions": 1,
    }


def test_processed_keys_are_recorded_once() -> None:
    manifest = make_manifest()
    manifest.record_meeting(1219)
    manifest.record_meeting(1219)
    manifest.record_session(None)

    assert manifest.meetings_processed == [1219]
    assert manifest.sessions_processed == []


def test_duration_is_none_until_the_run_finishes() -> None:
    started = datetime(2023, 3, 5, 12, 0, tzinfo=timezone.utc)
    manifest = make_manifest(started_at=started)

    assert manifest.duration_seconds is None

    manifest.finished_at = started + timedelta(seconds=90)
    assert manifest.duration_seconds == 90.0


def test_save_writes_a_timestamped_file_per_execution(tmp_path: Path) -> None:
    first = make_manifest(started_at=datetime(2023, 3, 5, 12, 0, tzinfo=timezone.utc))
    second = make_manifest(started_at=datetime(2023, 3, 5, 12, 30, tzinfo=timezone.utc))

    first_path = first.save(tmp_path / "manifests")
    second_path = second.save(tmp_path / "manifests")

    assert first_path != second_path
    assert first_path.name.startswith("manifest_20230305T120000")


def test_two_runs_starting_on_the_same_clock_tick_keep_both_manifests(tmp_path: Path) -> None:
    """An execution record must never quietly replace an earlier one."""
    started = datetime(2023, 3, 5, 12, 0, 0, 123456, tzinfo=timezone.utc)
    first = make_manifest(started_at=started, config={"years": [2023]})
    second = make_manifest(started_at=started, config={"years": [2024]})

    first_path = first.save(tmp_path / "manifests")
    second_path = second.save(tmp_path / "manifests")

    assert first_path != second_path
    assert len(list((tmp_path / "manifests").glob("manifest_*.json"))) == 2
    assert json.loads(first_path.read_text(encoding="utf-8"))["config"] == {"years": [2023]}
    assert json.loads(second_path.read_text(encoding="utf-8"))["config"] == {"years": [2024]}
