"""The batch entrypoint: arguments, storage selection, exit codes."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.modeling import checks
from factories import MEETING_KEY, SESSION_KEY, YEAR, car_data_record
from test_consolidation import build_raw  # noqa: F401 - fixture
from test_historical_extraction import make_client
from test_modeling import feature_rows


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch) -> None:
    for name in ("F1_ENVIRONMENT", "F1_STORAGE_BACKEND", "F1_STORAGE_BUCKET", "F1_STORAGE_LAYOUT", "F1_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_file(tmp_path: Path, raw_root: Path) -> Path:
    """A local configuration whose every path points into tmp_path."""
    config: Dict[str, Any] = {
        "historical_extraction": {
            "years": [2023],
            "session_types": ["Race"],
            "endpoints": ["laps"],
            "output_path": str(raw_root),
            "manifest_path": str(tmp_path / "manifests"),
        },
        "validation": {"raw_path": str(raw_root), "manifest_path": str(tmp_path / "manifests"), "report_path": str(tmp_path / "validation")},
        "consolidation": {
            "raw_path": str(raw_root),
            "output_path": str(tmp_path / "processed"),
            "report_path": str(tmp_path / "consolidation"),
            "validation_path": str(tmp_path / "validation"),
        },
        "features": {"input_path": str(tmp_path / "processed"), "output_path": str(tmp_path / "features"), "report_path": str(tmp_path / "feature_reports")},
        "modeling": {
            "input_path": str(tmp_path / "features"),
            "output_path": str(tmp_path / "modeling"),
            "report_path": str(tmp_path / "modeling_reports"),
            "split_strategy": "fraction",
            "train_fraction": 1 / 3,
            "validation_fraction": 1 / 3,
            "min_sessions_per_split": 1,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def fake_client_factory(client):
    return lambda settings: client


# -- arguments ----------------------------------------------------------------


def test_stages_are_parsed_in_order() -> None:
    assert cli.parse_stages("m3, M4,m5") == ["m3", "m4", "m5"]
    with pytest.raises(ValueError):
        cli.parse_stages("m7")


def test_unknown_stage_is_a_usage_error(config_file) -> None:
    assert cli.main(["--stage", "m8", "--config", str(config_file)]) == cli.EXIT_USAGE


def test_s3_without_a_bucket_is_a_usage_error(config_file) -> None:
    assert cli.main(["--stage", "m4", "--config", str(config_file), "--storage-backend", "s3"]) == cli.EXIT_USAGE


def test_m2_for_a_session_needs_the_year(config_file) -> None:
    assert cli.main(["--stage", "m2", "--session-key", "9158", "--config", str(config_file)]) == cli.EXIT_USAGE


def test_flags_become_configuration_overrides() -> None:
    args = cli.build_parser().parse_args(
        ["--stage", "m4", "--storage-backend", "s3", "--bucket", "b", "--prefix", "dev", "--overwrite", "--environment", "aws"]
    )

    overrides = cli.settings_overrides(args)

    assert overrides["storage"] == {"backend": "s3", "bucket": "b", "prefix": "dev"}
    assert overrides["environment"] == "aws"
    assert overrides["consolidation"] == {"overwrite": True}


# -- running stages ---------------------------------------------------------------


def test_m3_m4_m5_chain_for_one_session_locally(build_raw, config_file, tmp_path, capsys) -> None:
    build_raw(car_data={1: [car_data_record(1, 10)]})

    code = cli.main(
        ["--stage", "m3,m4,m5", "--year", str(YEAR), "--meeting-key", str(MEETING_KEY), "--session-key", str(SESSION_KEY), "--config", str(config_file)]
    )

    assert code == cli.EXIT_OK
    assert (tmp_path / "processed" / str(YEAR) / f"session_{SESSION_KEY}.parquet").exists()
    assert (tmp_path / "features" / str(YEAR) / f"session_{SESSION_KEY}.parquet").exists()
    report = json.loads(next((tmp_path / "validation").glob("validation_report_*.json")).read_text(encoding="utf-8"))
    assert report["source"]["filter"]["session_key"] == SESSION_KEY
    status_lines = capsys.readouterr().err
    assert "m3: OK" in status_lines and "m5: OK" in status_lines


def test_a_session_with_no_data_fails_the_job(config_file) -> None:
    assert cli.main(["--stage", "m4", "--session-key", "123456", "--config", str(config_file)]) == cli.EXIT_FAILED


def test_m2_session_job_writes_the_lake_layout_to_the_output_storage(config_file, tmp_path) -> None:
    client = make_client()
    lake_dir = tmp_path / "lake"

    code = cli.main(
        ["--stage", "m2", "--year", "2023", "--meeting-key", "1219", "--session-key", "9158", "--config", str(config_file),
         "--output-storage", str(lake_dir), "--output-layout", "lake"],
        client_factory=fake_client_factory(client),
    )

    assert code == cli.EXIT_OK
    assert (lake_dir / "raw" / "year=2023" / "meeting=1219" / "session=9158" / "endpoint=laps" / "session_9158.json").exists()
    assert list((lake_dir / "manifests" / "extraction" / "year=2023" / "session=9158").glob("manifest_*.json"))


def test_m2_retryable_failures_fail_the_job_but_404s_do_not(config_file) -> None:
    from f1_race_intelligence.ingestion.exceptions import OpenF1HTTPError

    missing = make_client()
    missing.get_laps.side_effect = OpenF1HTTPError(status_code=404, message="No results found.", endpoint="laps")
    broken = make_client()
    broken.get_laps.side_effect = OpenF1HTTPError(status_code=503, message="unavailable", endpoint="laps")
    args = ["--stage", "m2", "--year", "2023", "--config", str(config_file)]

    assert cli.main(args, client_factory=fake_client_factory(missing)) == cli.EXIT_OK
    assert cli.main([*args, "--overwrite"], client_factory=fake_client_factory(broken)) == cli.EXIT_FAILED


def test_plan_lists_work_units(config_file, tmp_path, capsys) -> None:
    output = tmp_path / "plan" / "units.json"

    code = cli.main(
        ["--stage", "plan", "--year", "2023", "--config", str(config_file), "--plan-output", str(output)],
        client_factory=fake_client_factory(make_client()),
    )

    assert code == cli.EXIT_OK
    units = json.loads(output.read_text(encoding="utf-8"))
    assert units == [{"year": 2023, "meeting_key": 1219, "session_key": 9158, "session_name": "Race"}]


def test_m6_blocked_by_a_guard_exits_non_zero(config_file, tmp_path, monkeypatch) -> None:
    for session_key, year, offset in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        frame = pd.DataFrame(feature_rows(session_key, year, offset))
        (tmp_path / "features" / str(year)).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_path / "features" / str(year) / f"session_{session_key}.parquet", index=False)

    assert cli.main(["--stage", "m6", "--config", str(config_file)]) == cli.EXIT_OK

    monkeypatch.setattr(
        checks, "check_chronological_split", lambda assignments: checks.CheckResult("chronological_split", False, "forced")
    )
    assert cli.main(["--stage", "m6", "--config", str(config_file)]) == cli.EXIT_FAILED


def test_an_unexpected_exception_is_reported_as_a_failed_stage(config_file, monkeypatch) -> None:
    def explode(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.STAGE_RUNNERS, "m5", explode)

    assert cli.main(["--stage", "m5", "--config", str(config_file)]) == cli.EXIT_FAILED


def test_copy_raw_between_layouts(build_raw, config_file, raw_root, tmp_path) -> None:
    build_raw()

    code = cli.main(
        ["--stage", "copy-raw", "--config", str(config_file), "--input-layout", "legacy",
         "--output-storage", str(tmp_path / "lake"), "--output-layout", "lake"]
    )

    assert code == cli.EXIT_OK
    assert (tmp_path / "lake" / "raw" / f"year={YEAR}" / "endpoint=meetings" / "meetings.json").exists()
