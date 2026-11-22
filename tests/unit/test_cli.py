"""The batch entrypoint: arguments, storage selection, exit codes."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.modeling import checks
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
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
    assert cli.parse_stages("m7") == ["m7"]
    with pytest.raises(ValueError):
        cli.parse_stages("m7,unknown")


def test_m7_requires_an_m6_report(config_file) -> None:
    assert cli.main(["--stage", "m7", "--config", str(config_file)]) == cli.EXIT_USAGE


def test_m7_rejects_s3_storage(config_file) -> None:
    assert cli.main(
        [
            "--stage",
            "m7",
            "--m6-report",
            "m6-report.json",
            "--storage-backend",
            "s3",
            "--bucket",
            "not-used",
            "--config",
            str(config_file),
        ]
    ) == cli.EXIT_USAGE


def test_m7_must_run_against_an_existing_m6_report(config_file) -> None:
    assert cli.main(
        [
            "--stage",
            "m6,m7",
            "--m6-report",
            "m6-report.json",
            "--config",
            str(config_file),
        ]
    ) == cli.EXIT_USAGE


def test_unknown_stage_is_a_usage_error(config_file) -> None:
    assert cli.main(["--stage", "m8", "--config", str(config_file)]) == cli.EXIT_USAGE


def test_s3_without_a_bucket_is_a_usage_error(config_file) -> None:
    assert cli.main(["--stage", "m4", "--config", str(config_file), "--storage-backend", "s3"]) == cli.EXIT_USAGE


def test_m2_for_a_session_needs_the_year(config_file) -> None:
    assert cli.main(["--stage", "m2", "--session-key", "9158", "--config", str(config_file)]) == cli.EXIT_USAGE


# -- configuration that would make a job "succeed" doing nothing -------------------


def never_called(settings):
    raise AssertionError("no OpenF1 client may be built for a rejected run")


def test_a_missing_config_file_is_a_usage_error_naming_the_path(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.yaml"

    code = cli.main(["--stage", "m2", "--year", "2024", "--config", str(missing)], client_factory=never_called)

    assert code == cli.EXIT_USAGE
    error = capsys.readouterr().err
    assert "Configuration file not found" in error and str(missing) in error


def test_a_missing_f1_config_path_is_a_usage_error(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("F1_CONFIG_PATH", str(tmp_path / "missing.yaml"))

    assert cli.main(["--stage", "m3"]) == cli.EXIT_USAGE
    assert "F1_CONFIG_PATH" in capsys.readouterr().err


def test_an_installed_package_without_config_does_not_report_an_empty_m2_as_success(tmp_path, monkeypatch, capsys) -> None:
    from f1_race_intelligence.config import settings as settings_module

    monkeypatch.setattr(settings_module, "DEFAULT_CONFIG_PATH", tmp_path / "site-packages" / "configs" / "config.yaml")

    code = cli.main(["--stage", "m2", "--year", "2024", "--session-key", "9472"], client_factory=never_called)

    assert code == cli.EXIT_USAGE
    assert "F1_CONFIG_PATH is not set" in capsys.readouterr().err


def test_m2_without_endpoints_is_a_usage_error(tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("historical_extraction:\n  years: [2024]\n  endpoints: []\n", encoding="utf-8")

    code = cli.main(["--stage", "m2", "--year", "2024", "--session-key", "9472", "--config", str(config)], client_factory=never_called)

    assert code == cli.EXIT_USAGE
    assert "endpoints is empty" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["plan", "m2"])
def test_extraction_without_a_season_is_a_usage_error(tmp_path, capsys, stage) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("historical_extraction:\n  years: []\n  endpoints: [laps]\n", encoding="utf-8")

    assert cli.main(["--stage", stage, "--config", str(config)], client_factory=never_called) == cli.EXIT_USAGE
    assert "no season to process" in capsys.readouterr().err


def test_a_season_from_the_command_line_is_enough(config_file) -> None:
    settings = cli.load_settings(str(config_file), overrides={"historical_extraction": {"years": []}})

    cli.check_stage_inputs(["plan", "m2"], settings, year=2024)  # no exception
    cli.check_stage_inputs(["m3", "m4", "m5", "m6"], settings, year=None)  # only extraction needs one


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
    assert "m3: PARTIAL" in status_lines and "m5: OK" in status_lines


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
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert len(plan["sessions"]) == 1
    assert plan["sessions"][0] == {
        "year": 2023,
        "meeting_key": 1219,
        "session_key": 9158,
        "session_name": "Race",
        "process_reason": "no verified completion provenance",
    }
    assert plan["decisions"] == [
        {
            "year": 2023,
            "meeting_key": 1219,
            "session_key": 9158,
            "session_name": "Race",
            "decision": "include",
            "reason": "no verified completion provenance",
        }
    ]


def test_plan_can_be_written_to_s3_for_the_orchestrator(config_file, monkeypatch) -> None:
    # Step Functions cannot read a container's stdout: on AWS the plan is read
    # back from S3 by the state machine (LoadPlan).
    from s3_fakes import FakeS3Client

    s3 = FakeS3Client()
    monkeypatch.setattr(cli, "S3StorageBackend", lambda bucket, **kwargs: S3StorageBackend(bucket, **{**kwargs, "client": s3}))

    code = cli.main(
        ["--stage", "plan", "--year", "2023", "--config", str(config_file), "--plan-output", "s3://plans-bucket/manifests/plan/exec-1.json"],
        client_factory=fake_client_factory(make_client()),
    )

    assert code == cli.EXIT_OK
    stored = json.loads(s3.get_object(Bucket="plans-bucket", Key="manifests/plan/exec-1.json")["Body"].read())
    assert stored["sessions"][0]["session_key"] == 9158
    assert stored["decisions"][0]["decision"] == "include"


@pytest.mark.parametrize("destination", ["s3://plans-bucket", "s3://plans-bucket/manifests/plan/", "gs://bucket/plan.json"])
def test_an_unusable_plan_output_is_rejected_before_calling_openf1(config_file, destination, capsys) -> None:
    code = cli.main(["--stage", "plan", "--year", "2023", "--config", str(config_file), "--plan-output", destination], client_factory=never_called)

    assert code == cli.EXIT_USAGE
    assert "--plan-output" in capsys.readouterr().err


def test_m6_blocked_by_a_guard_exits_non_zero(config_file, tmp_path, monkeypatch) -> None:
    for session_key, year, offset in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        frame = pd.DataFrame(feature_rows(session_key, year, offset))
        (tmp_path / "features" / str(year)).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_path / "features" / str(year) / f"session_{session_key}.parquet", index=False)

    assert cli.main(["--stage", "m6", "--config", str(config_file)]) == cli.EXIT_OK

    # M6 reuses a previous VALID result for the same inputs (possible on a clean checkout),
    # which would skip the guard below. An M6 output location of its own has no previous
    # report, so the forced guard is evaluated by a fresh run.
    settings = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    settings["modeling"]["output_path"] = str(tmp_path / "modeling_guarded")
    settings["modeling"]["report_path"] = str(tmp_path / "modeling_guarded_reports")
    guarded_config = tmp_path / "config_guarded.yaml"
    guarded_config.write_text(yaml.safe_dump(settings), encoding="utf-8")

    monkeypatch.setattr(
        checks, "check_chronological_split", lambda assignments: checks.CheckResult("chronological_split", False, "forced")
    )
    assert cli.main(["--stage", "m6", "--config", str(guarded_config)]) == cli.EXIT_FAILED


def test_cli_skips_m6_only_for_matching_valid_inputs(config_file, tmp_path, monkeypatch, capsys) -> None:
    import f1_race_intelligence.modeling.runner as modeling_module

    for session_key, year, offset in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        frame = pd.DataFrame(feature_rows(session_key, year, offset))
        path = tmp_path / "features" / str(year) / f"session_{session_key}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(modeling_module, "source_revision", lambda: ("a" * 40, False))
    monkeypatch.setattr(modeling_module, "dependency_lock_sha256", lambda: "b" * 64)
    original_run = modeling_module.ModelingRunner.run
    calls = []

    def count_runs(self):
        calls.append(True)
        return original_run(self)

    monkeypatch.setattr(modeling_module.ModelingRunner, "run", count_runs)
    args = ["--stage", "m6", "--config", str(config_file)]

    assert cli.main(args) == cli.EXIT_OK
    assert cli.main(args) == cli.EXIT_OK
    assert len(calls) == 1
    assert "reused_from_run_id" in capsys.readouterr().err

    path = tmp_path / "features" / "2023" / "session_1.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "speed_mean"] += 1
    changed.to_parquet(path, index=False)
    assert cli.main(args) == cli.EXIT_OK
    assert len(calls) == 2


def test_m6_blocking_guard_still_fails_pipeline_when_leakage_stop_is_disabled(
    config_file, tmp_path, monkeypatch
) -> None:
    for session_key, year, offset in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        frame = pd.DataFrame(feature_rows(session_key, year, offset))
        (tmp_path / "features" / str(year)).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_path / "features" / str(year) / f"session_{session_key}.parquet", index=False)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config["modeling"]["fail_on_leakage"] = False
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(
        checks, "check_chronological_split", lambda assignments: checks.CheckResult("chronological_split", False, "forced")
    )
    assert cli.main(["--stage", "m6", "--config", str(config_file)]) == cli.EXIT_FAILED


def test_verify_m6_requires_a_report_and_accepts_a_legacy_report(config_file, tmp_path) -> None:
    assert cli.main(["--stage", "verify-m6", "--config", str(config_file)]) == cli.EXIT_USAGE
    report = tmp_path / "m6-v1.json"
    report.write_text(json.dumps({"artifact_files": []}), encoding="utf-8")

    assert cli.main(["--stage", "verify-m6", "--report", str(report), "--config", str(config_file)]) == cli.EXIT_OK


def test_an_unexpected_exception_is_reported_as_a_failed_stage(config_file, monkeypatch) -> None:
    def explode(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.STAGE_RUNNERS, "m5", explode)

    assert cli.main(["--stage", "m5", "--config", str(config_file)]) == cli.EXIT_FAILED


@pytest.mark.parametrize(
    ("global_status", "expected_status", "expected_label"),
    [
        ("PASS", "succeeded", "m3: OK"),
        ("WARNING", "partial", "m3: PARTIAL"),
        ("FAIL", "failed", "m3: FAILED (non-strict; pipeline continues)"),
    ],
)
def test_m3_global_status_is_not_reported_as_success_unless_pass(
    config_file, monkeypatch, caplog, capsys, global_status, expected_status, expected_label
) -> None:
    monkeypatch.setitem(
        cli.STAGE_RUNNERS,
        "m3",
        lambda ctx: cli.StageOutcome(ok=True, summary={"global_status": global_status}),
    )

    with caplog.at_level("INFO", logger="f1_race_intelligence.cli"):
        assert cli.main(["--stage", "m3", "--config", str(config_file)]) == cli.EXIT_OK

    finished = next(record for record in caplog.records if record.getMessage() == "stage_finished")
    assert finished.status == expected_status
    assert expected_label in capsys.readouterr().err


def test_m6_warnings_are_logged_as_a_partial_stage(config_file, monkeypatch, caplog, capsys) -> None:
    monkeypatch.setitem(
        cli.STAGE_RUNNERS,
        "m6",
        lambda ctx: cli.StageOutcome(
            ok=True,
            summary={"verdict": "VALID_WITH_WARNINGS", "run_id": "m6-run-1"},
        ),
    )

    with caplog.at_level("INFO", logger="f1_race_intelligence.cli"):
        assert cli.main(["--stage", "m6", "--config", str(config_file)]) == cli.EXIT_OK

    finished = next(record for record in caplog.records if record.getMessage() == "stage_finished")
    assert finished.status == "partial"
    assert finished.summary["run_id"] == "m6-run-1"
    assert "m6: PARTIAL" in capsys.readouterr().err


def test_copy_raw_between_layouts(build_raw, config_file, raw_root, tmp_path) -> None:
    build_raw()

    code = cli.main(
        ["--stage", "copy-raw", "--config", str(config_file), "--input-layout", "legacy",
         "--output-storage", str(tmp_path / "lake"), "--output-layout", "lake"]
    )

    assert code == cli.EXIT_OK
    assert (tmp_path / "lake" / "raw" / f"year={YEAR}" / "endpoint=meetings" / "meetings.json").exists()


# -- M7 model selection (--model-type / --registered-model-name) -----------------


@pytest.mark.parametrize(
    ("stage", "flags", "message"),
    [
        ("m6", ["--model-type", "mlp"], "--model-type is only valid with --stage m7"),
        ("m5", ["--model-type", "linear_regression"], "--model-type is only valid with --stage m7"),
        ("verify-m6", ["--report", "r.json", "--model-type", "mlp"], "--model-type is only valid with --stage m7"),
        ("m6", ["--registered-model-name", "custom"], "--registered-model-name is only valid with --stage m7"),
        ("m4", ["--registered-model-name", "custom"], "--registered-model-name is only valid with --stage m7"),
    ],
)
def test_m7_model_flags_are_rejected_outside_m7(config_file, capsys, stage, flags, message) -> None:
    assert cli.main(["--stage", stage, *flags, "--config", str(config_file)]) == cli.EXIT_USAGE
    assert message in capsys.readouterr().err


def test_an_unknown_model_type_is_rejected_by_the_parser(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--stage", "m7", "--model-type", "random_forest"])
    assert excinfo.value.code == 2
    assert "random_forest" in capsys.readouterr().err


def test_model_flags_default_to_none_and_add_no_training_overrides() -> None:
    args = cli.build_parser().parse_args(["--stage", "m7", "--m6-report", "report.json"])

    assert args.model_type is None
    assert args.registered_model_name is None
    assert "training" not in cli.settings_overrides(args)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--model-type", "mlp"], {"model_type": "mlp"}),
        (["--model-type", "linear_regression"], {"model_type": "linear_regression"}),
        (
            ["--registered-model-name", "custom"],
            {"registered_model_name": "custom", "mlp": {"registered_model_name": "custom"}},
        ),
        (
            ["--model-type", "mlp", "--registered-model-name", "custom-mlp"],
            {
                "model_type": "mlp",
                "registered_model_name": "custom-mlp",
                "mlp": {"registered_model_name": "custom-mlp"},
            },
        ),
    ],
)
def test_model_flags_become_training_overrides(flags, expected) -> None:
    args = cli.build_parser().parse_args(["--stage", "m7", "--m6-report", "report.json", *flags])

    assert cli.settings_overrides(args)["training"] == expected


@pytest.mark.parametrize(
    ("flags", "model_type", "registered_name"),
    [
        ([], "linear_regression", "f1-next-lap-linear-regression"),
        (["--model-type", "mlp"], "mlp", "f1-next-lap-mlp"),
        (["--model-type", "mlp", "--registered-model-name", "custom-mlp"], "mlp", "custom-mlp"),
        (["--model-type", "linear_regression", "--registered-model-name", "custom-lr"], "linear_regression", "custom-lr"),
    ],
)
def test_model_overrides_resolve_the_effective_settings(config_file, flags, model_type, registered_name) -> None:
    args = cli.build_parser().parse_args(["--stage", "m7", "--m6-report", "report.json", *flags])

    settings = cli.load_settings(str(config_file), overrides=cli.settings_overrides(args))

    assert settings.training.model_type == model_type
    assert settings.training.effective_registered_model_name() == registered_name
    # the overlay is merged deeply: the configured MLP hyperparameters survive
    assert settings.training.mlp.hidden_dims == [64, 32]


def test_registered_model_name_override_keeps_yaml_mlp_hyperparameters(config_file) -> None:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    data["training"] = {"model_type": "mlp", "mlp": {"hidden_dims": [8], "max_epochs": 3}}
    config_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    args = cli.build_parser().parse_args(
        ["--stage", "m7", "--m6-report", "report.json", "--registered-model-name", "from-cli"]
    )

    settings = cli.load_settings(str(config_file), overrides=cli.settings_overrides(args))

    assert settings.training.model_type == "mlp"
    assert settings.training.mlp.hidden_dims == [8]
    assert settings.training.mlp.max_epochs == 3
    assert settings.training.effective_registered_model_name() == "from-cli"


@pytest.mark.parametrize(
    ("flags", "expected_model_type", "expected_name"),
    [
        ([], None, "f1-next-lap-linear-regression"),
        (["--model-type", "mlp"], "mlp", "f1-next-lap-mlp"),
        (["--model-type", "mlp", "--registered-model-name", "cli-mlp"], "mlp", "cli-mlp"),
    ],
)
def test_m7_passes_model_selection_to_the_training_runner(
    config_file, tmp_path, monkeypatch, capsys, flags, expected_model_type, expected_name
) -> None:
    from f1_race_intelligence.modeling import training as training_module

    calls = []

    class FakeRunner:
        def __init__(self, settings, *, m6_report_path=None, model_type=None):
            calls.append(
                {
                    "m6_report_path": m6_report_path,
                    "model_type": model_type,
                    "effective_name": settings.training.effective_registered_model_name(model_type),
                }
            )

        def run(self):
            return training_module.TrainingResult(
                run_id="m7-run",
                m6_run_id="m6-run",
                input_fingerprint="sha256:" + "0" * 64,
                model_name=calls[-1]["effective_name"],
                model_version="1",
                model_uri=f"models:/{calls[-1]['effective_name']}/1",
                metrics={},
                report_path=str(tmp_path / "report.json"),
                source_commit=None,
                model_type=calls[-1]["model_type"] or "linear_regression",
            )

    monkeypatch.setattr(training_module, "TrainingRunner", FakeRunner)
    report = tmp_path / "m6-report.json"
    report.write_text("{}", encoding="utf-8")

    code = cli.main(["--stage", "m7", "--m6-report", str(report), *flags, "--config", str(config_file)])

    assert code == cli.EXIT_OK
    assert calls == [
        {"m6_report_path": str(report), "model_type": expected_model_type, "effective_name": expected_name}
    ]
    assert f'"model_name": "{expected_name}"' in capsys.readouterr().err


# -- M7 controlled MLP comparison (--stage m7-compare) ----------------------------


def test_m7_compare_is_a_known_stage() -> None:
    assert "m7-compare" in cli.STAGES
    assert cli.parse_stages("m7-compare") == ["m7-compare"]
    assert cli.parse_stages(" M7-Compare ") == ["m7-compare"]


def test_m7_compare_must_run_alone(config_file, capsys) -> None:
    code = cli.main(["--stage", "m6,m7-compare", "--m6-report", "m6-report.json", "--config", str(config_file)])

    assert code == cli.EXIT_USAGE
    assert "m7-compare must run alone" in capsys.readouterr().err


def test_m7_compare_requires_an_m6_report(config_file, capsys) -> None:
    assert cli.main(["--stage", "m7-compare", "--config", str(config_file)]) == cli.EXIT_USAGE
    assert "m7-compare requires --m6-report" in capsys.readouterr().err


def test_m7_compare_rejects_s3_storage(config_file, capsys) -> None:
    code = cli.main(
        [
            "--stage",
            "m7-compare",
            "--m6-report",
            "m6-report.json",
            "--storage-backend",
            "s3",
            "--bucket",
            "not-used",
            "--config",
            str(config_file),
        ]
    )

    assert code == cli.EXIT_USAGE
    assert "m7-compare supports local storage only" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--input-storage", "--output-storage"])
def test_m7_compare_rejects_storage_uri_overrides(config_file, tmp_path, capsys, flag) -> None:
    code = cli.main(
        [
            "--stage",
            "m7-compare",
            "--m6-report",
            "m6-report.json",
            flag,
            str(tmp_path / "elsewhere"),
            "--config",
            str(config_file),
        ]
    )

    assert code == cli.EXIT_USAGE
    assert "m7-compare does not accept storage URI overrides" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("stage", "extra"),
    [
        ("m7", ["--m6-report", "m6-report.json"]),
        ("m6", []),
        ("verify-m6", ["--report", "r.json"]),
    ],
)
def test_register_best_is_rejected_outside_m7_compare(config_file, capsys, stage, extra) -> None:
    code = cli.main(["--stage", stage, *extra, "--register-best", "--config", str(config_file)])

    assert code == cli.EXIT_USAGE
    assert "--register-best is only valid with --stage m7-compare" in capsys.readouterr().err


def test_register_best_defaults_to_false_and_adds_no_overrides() -> None:
    args = cli.build_parser().parse_args(["--stage", "m7-compare", "--m6-report", "report.json"])

    assert args.register_best is False
    assert "training" not in cli.settings_overrides(args)


def test_m7_compare_accepts_registered_model_name_and_register_best(
    config_file, tmp_path, monkeypatch, capsys
) -> None:
    from f1_race_intelligence.modeling import experiments as experiments_module

    calls = []

    class FakeComparisonRunner:
        def __init__(self, settings, *, m6_report_path=None, register_best=False):
            calls.append(
                {
                    "m6_report_path": m6_report_path,
                    "register_best": register_best,
                    "effective_name": settings.training.effective_registered_model_name("mlp"),
                }
            )

        def run(self):
            return experiments_module.MLPComparisonResult(
                comparison_id="cmp-test",
                m6_run_id="m6-run",
                input_fingerprint="sha256:" + "0" * 64,
                selection_metric="validation.mlp.mae",
                best_experiment="mlp_default",
                candidates=[],
                registered_winner=None,
                json_report_path=str(tmp_path / "cmp.json"),
                markdown_report_path=str(tmp_path / "cmp.md"),
            )

    monkeypatch.setattr(experiments_module, "MLPComparisonRunner", FakeComparisonRunner)
    report = tmp_path / "m6-report.json"
    report.write_text("{}", encoding="utf-8")

    code = cli.main(
        [
            "--stage",
            "m7-compare",
            "--m6-report",
            str(report),
            "--register-best",
            "--registered-model-name",
            "cli-compare-mlp",
            "--config",
            str(config_file),
        ]
    )

    assert code == cli.EXIT_OK
    assert calls == [{"m6_report_path": str(report), "register_best": True, "effective_name": "cli-compare-mlp"}]
    assert '"best_experiment": "mlp_default"' in capsys.readouterr().err


def test_m7_compare_without_register_best_passes_false(config_file, tmp_path, monkeypatch) -> None:
    from f1_race_intelligence.modeling import experiments as experiments_module

    seen = []

    class FakeComparisonRunner:
        def __init__(self, settings, *, m6_report_path=None, register_best=False):
            seen.append(register_best)

        def run(self):
            return experiments_module.MLPComparisonResult(
                comparison_id="cmp-test",
                m6_run_id="m6-run",
                input_fingerprint="sha256:" + "0" * 64,
                selection_metric="validation.mlp.mae",
                best_experiment="mlp_small",
                candidates=[],
                registered_winner=None,
                json_report_path=str(tmp_path / "cmp.json"),
                markdown_report_path=str(tmp_path / "cmp.md"),
            )

    monkeypatch.setattr(experiments_module, "MLPComparisonRunner", FakeComparisonRunner)
    report = tmp_path / "m6-report.json"
    report.write_text("{}", encoding="utf-8")

    assert cli.main(["--stage", "m7-compare", "--m6-report", str(report), "--config", str(config_file)]) == cli.EXIT_OK
    assert seen == [False]
