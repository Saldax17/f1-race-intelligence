"""M7 race_model contract parity tests: provenance, SHA-256 manifest/PUBLISHED,
generalization diagnosis, structural interpretability, Markdown report,
serving schema validation, verify-m7 CLI, and optional lazy MLflow logging.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.race_model.artifact import ModelRegistry
from f1_race_intelligence.race_model.contracts import (
    MANIFEST_FILENAME,
    PUBLISHED_MARKER,
    diagnose_race_model_generalization,
    extract_interpretability,
    race_input_fingerprint,
    render_evaluation_markdown,
    validate_forecast_request,
    validate_forecast_response,
)
from f1_race_intelligence.race_model.runner import RaceModelRunner
from f1_race_intelligence.storage.store import DataStore

from tests.unit.test_race_model_isolation import PoisonedCatalog


def _make_runner(tmp_path: Path, *, altered_validation: bool = False) -> RaceModelRunner:
    settings = AppSettings.model_validate(
        {
            "training": {
                "tracking_db_path": str(tmp_path / "mlflow" / "tracking.db"),
                "artifact_path": str(tmp_path / "mlflow" / "artifacts"),
                "experiment_name": "f1-race-model-contract-test",
            },
            "race_model": {
                "years": [2023, 2024, 2025],
                "train_years": [2023, 2024],
                "validation_until": "2025-08-01",
                "test_until": "2026-01-01",
                "burn_in_races": 1,
                "scenarios": 200,
                "noise_scales": [1.0, 2.0],
                "calibration_races": 2,
                "min_clean_laps": 3,
                "output_path": "models",
            },
        }
    )
    store = DataStore.local(str(tmp_path))
    run = RaceModelRunner(settings, store, store)
    run.catalog = PoisonedCatalog(altered_validation=altered_validation)
    return run


# -- 1. Provenance & Immutable Package Integrity ------------------------------


def test_provenance_and_manifest_published_roundtrip(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    assert trained.ok, trained.reason
    version = trained.summary["version"]

    artifact = run.registry.load(version)
    prov = artifact.provenance
    assert prov["schema_version"] == 2
    assert prov["model_type"] == "hierarchical_state_space"
    assert prov["input_fingerprint"].startswith("sha256:")
    assert len(prov["config_fingerprint"]) == 64
    assert prov["train_races_count"] == 6
    assert prov["validation_races_count"] == 2
    assert prov["test_sealed"] is True

    # Immutable manifest.json and PUBLISHED marker exist and verify cleanly.
    version_dir = tmp_path / "models" / f"version={version}"
    assert (version_dir / "model.json").is_file()
    assert (version_dir / MANIFEST_FILENAME).is_file()
    assert (version_dir / PUBLISHED_MARKER).is_file()

    verified = run.verify(version)
    assert verified.ok, verified.reason
    assert verified.summary["files_checked"] == 1
    assert verified.summary["errors"] == []


def test_train_input_fingerprint_is_invariant_to_validation_changes(tmp_path: Path) -> None:
    r_orig = _make_runner(tmp_path / "orig", altered_validation=False)
    r_alt = _make_runner(tmp_path / "alt", altered_validation=True)
    t_orig = r_orig.train()
    t_alt = r_alt.train()
    assert t_orig.ok and t_alt.ok
    assert t_orig.summary["input_fingerprint"] == t_alt.summary["input_fingerprint"]


def test_verify_detects_tampering_and_missing_marker(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    version = trained.summary["version"]
    version_dir = tmp_path / "models" / f"version={version}"

    # Tamper with model.json
    model_file = version_dir / "model.json"
    original_bytes = model_file.read_bytes()
    model_file.write_bytes(original_bytes + b" ")
    tampered = run.verify(version)
    assert not tampered.ok
    assert any("sha256 mismatch" in err for err in tampered.summary["errors"])

    # Restore model.json and remove PUBLISHED marker
    model_file.write_bytes(original_bytes)
    marker = version_dir / PUBLISHED_MARKER
    marker_bytes = marker.read_bytes()
    marker.unlink()
    missing = run.verify(version)
    assert not missing.ok
    assert any(PUBLISHED_MARKER in err for err in missing.summary["errors"])

    # Corrupt PUBLISHED marker hash
    marker.write_bytes(b"sha256:" + b"0" * 64 + b"\n")
    bad_marker = run.verify(version)
    assert not bad_marker.ok
    assert any("mismatch" in err for err in bad_marker.summary["errors"])

    # Restore marker -> passes again
    marker.write_bytes(marker_bytes)
    assert run.verify(version).ok


# -- 2. Generalization Diagnosis & Structural Interpretability ----------------


def test_diagnose_generalization_all_states() -> None:
    balanced = diagnose_race_model_generalization(
        prior_check={"train": {"rmse": 0.42, "z2": 1.05}, "validation": {"rmse": 0.41, "z2": 0.98}},
        validation_summary={"by_bucket": {"all": {"model_rps": 0.080, "persist_rps": 0.116, "transition_rps": 0.082}}},
        noise_scale_rps={"overall": {"1.0": 0.070, "2.0": 0.068}},
    )
    assert balanced["status"] == "balanced"
    assert balanced["evaluated_splits"] == ["train", "validation"]

    underfit = diagnose_race_model_generalization(
        prior_check={"train": {"rmse": 0.40, "z2": 1.0}, "validation": {"rmse": 0.41, "z2": 1.0}},
        validation_summary={"by_bucket": {"all": {"model_rps": 0.120, "persist_rps": 0.116, "transition_rps": 0.082}}},
    )
    assert underfit["status"] == "underfitting"

    overfit = diagnose_race_model_generalization(
        prior_check={"train": {"rmse": 0.30, "z2": 0.9}, "validation": {"rmse": 0.50, "z2": 1.8}},
        validation_summary={"by_bucket": {"all": {"model_rps": 0.095, "persist_rps": 0.116, "transition_rps": 0.082}}},
    )
    assert overfit["status"] == "overfitting_risk"

    missing = diagnose_race_model_generalization(prior_check={}, validation_summary={})
    assert missing["status"] == "insufficient_metrics"


def test_interpretability_and_markdown_evaluation_report(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    assert trained.ok
    version = trained.summary["version"]
    artifact = run.registry.load(version)

    interp = extract_interpretability(artifact)
    assert interp["status"] == "computed"
    assert interp["method"] == "structural_parameter_decomposition"
    assert interp["fit_split"] == "train"
    shares = [c["variance_share"] for c in interp["level2_season_variance"]["components"]]
    assert pytest.approx(sum(shares), abs=1e-6) == 1.0
    assert interp["note"]

    evaluated = run.evaluate(version=version, split="validation")
    assert evaluated.ok, evaluated.reason
    assert evaluated.summary["package_verified"] is True
    assert evaluated.summary["generalization_status"] in {"balanced", "underfitting", "overfitting_risk"}

    md_path = tmp_path / "models" / "reports" / "evaluate_validation.md"
    assert md_path.is_file()
    md = md_path.read_text(encoding="utf-8")
    for heading in (
        "## 1. Provenance & Experimental Protocol",
        "## 2. Comparative Metrics",
        "## 3. Overfitting / Underfitting Diagnosis",
        "## 4. Interpretability",
        "## 5. Model Registry & Artifact Integrity Status",
    ):
        assert heading in md

    # Evaluation JSON contains the contract sections too.
    eval_jsons = list((tmp_path / "models" / "reports").glob("evaluate_validation*.json"))
    assert eval_jsons
    eval_doc = json.loads(eval_jsons[-1].read_text(encoding="utf-8"))
    assert eval_doc["provenance"]["input_fingerprint"] == artifact.provenance["input_fingerprint"]
    assert eval_doc["generalization_diagnosis"]["evaluated_splits"] == ["train", "validation"]
    assert eval_doc["interpretability"]["method"] == "structural_parameter_decomposition"
    assert eval_doc["package_verification"]["ok"] is True


# -- 3. Serving Input & Output Contract Validation ----------------------------


def test_serving_request_and_response_contracts() -> None:
    assert validate_forecast_request({"mode": "live"}) == {"mode": "live"}
    assert validate_forecast_request({"session_key": 9998, "lap": 30, "source": "raw"})["session_key"] == 9998

    with pytest.raises(ValueError, match="unknown"):
        validate_forecast_request({"session_key": 9998, "unexpected_field": 1})
    with pytest.raises(ValueError, match="positive integer"):
        validate_forecast_request({"session_key": -1})
    with pytest.raises(ValueError, match="lap"):
        validate_forecast_request({"session_key": 9998, "lap": 0})
    with pytest.raises(ValueError, match="source"):
        validate_forecast_request({"session_key": 9998, "source": "ftp"})

    good_resp = {
        "model_version": "v1",
        "lap": 10,
        "total_laps": 50,
        "drivers": [{"driver": "VER", "p_dnf": 0.05}, {"driver": "NOR", "p_dnf": 0.04}],
        "matrix": [[0.7, 0.3], [0.3, 0.7]],
    }
    validate_forecast_response(good_resp)

    bad_matrix = {**good_resp, "matrix": [[0.9, 0.9], [0.1, 0.1]]}
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_forecast_response(bad_matrix)

    bad_dnf = {**good_resp, "drivers": [{"driver": "VER", "p_dnf": 1.5}, {"driver": "NOR", "p_dnf": 0.0}]}
    with pytest.raises(ValueError, match="p_dnf"):
        validate_forecast_response(bad_dnf)


# -- 4. CLI verify-m7 & --log-mlflow flags ------------------------------------


def test_cli_verify_m7_and_log_mlflow_flag_validation(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    version = trained.summary["version"]

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"race_model:\n  output_path: models\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "--stage",
                "verify-m7",
                "--config",
                str(cfg_path),
                "--model-version",
                version,
                "--local-root",
                str(tmp_path),
            ]
        )
        == cli.EXIT_OK
    )
    assert (
        cli.main(["--stage", "m7-train", "--log-mlflow", "--config", str(cfg_path)])
        == cli.EXIT_USAGE
    )


# -- 5. Optional Lazy MLflow Tracking & Pyfunc Reload -------------------------


def test_optional_mlflow_logging_and_pyfunc_reload(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    import mlflow.pyfunc

    run = _make_runner(tmp_path)
    trained = run.train()
    assert trained.ok
    version = trained.summary["version"]

    evaluated = run.evaluate(version=version, split="validation", log_mlflow=True)
    assert evaluated.ok, evaluated.reason
    info = evaluated.summary["mlflow"]
    assert info["registered_model_name"] == "f1-race-position-model"
    assert info["registered_version"] == "1"
    assert info["model_uri"] == "models:/f1-race-position-model/1"

    mlflow.set_tracking_uri(info["tracking_uri"])
    loaded = mlflow.pyfunc.load_model(info["model_uri"])
    out = loaded.predict(pd.DataFrame([{"session_key": 7, "lap": 5}]))
    assert out.loc[0, "model_version"] == version
    assert out.loc[0, "races_seen"] == 8

    client = mlflow.tracking.MlflowClient(tracking_uri=info["tracking_uri"])
    artifacts = {a.path for a in client.list_artifacts(info["run_id"])}
    assert {
        "data_provenance.json",
        "training_config.json",
        "metrics.json",
        "evaluate_validation.json",
        "evaluate_validation.md",
        "interpretability.json",
        "package_verification.json",
    } <= artifacts
