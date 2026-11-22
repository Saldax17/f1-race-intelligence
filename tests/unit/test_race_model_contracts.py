"""M7 race_model contract parity tests: deterministic v2 training fingerprint,
atomic/idempotent SHA-256 package publication & canonical JSON, mandatory
verification before load/promote, generalization diagnosis, structural
interpretability, Markdown report, strict serving contracts, verify-m7 CLI,
lazy MLflow import isolation, and predictive MLflow Pyfunc parity.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.race_model.artifact import ModelArtifact, ModelRegistry, new_version
from f1_race_intelligence.race_model.contracts import (
    INPUT_FINGERPRINT_SCHEME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    PUBLISHED_MARKER,
    ModelPackageIntegrityError,
    canonical_json_bytes,
    diagnose_race_model_generalization,
    extract_interpretability,
    race_input_fingerprint,
    validate_forecast_request,
    validate_forecast_response,
)
from f1_race_intelligence.race_model.racedata import race_to_forecast_payload
from f1_race_intelligence.race_model.runner import RaceModelRunner
from f1_race_intelligence.storage.store import DataStore

from tests.unit.test_race_model import synthetic_race
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


def _rich_train_race(seed: int = 1, session_key: int = 101):
    race = synthetic_race(seed=seed, session_key=session_key, year=2024, location="Monza")
    race.pit = [
        {"driver_number": 1, "lap_number": 8, "date": "2024-03-02T15:12:00+00:00", "pit_duration": 22.1},
        {"driver_number": 2, "lap_number": 9, "date": "2024-03-02T15:13:30+00:00", "pit_duration": 22.4},
    ]
    race.race_control = [
        {"date": "2024-03-02T15:15:00+00:00", "lap_number": 10, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"date": "2024-03-02T15:18:00+00:00", "lap_number": 12, "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
    ]
    race.grid = [
        {"driver_number": i + 1, "position": i + 1, "lap_duration": 80.0 + 0.2 * i, "session_key": 999}
        for i in range(6)
    ]
    return race


# -- 1. Provenance, Complete v2 Fingerprint & Canonical Package Integrity -----


def test_provenance_canonical_json_and_manifest_published_roundtrip(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    assert trained.ok, trained.reason
    version = trained.summary["version"]

    artifact = run.registry.load(version)
    prov = artifact.provenance
    assert prov["schema_version"] == 2
    assert prov["model_type"] == "hierarchical_state_space"
    assert prov["input_fingerprint"].startswith("sha256:")
    assert prov["input_fingerprint_scheme"] == INPUT_FINGERPRINT_SCHEME
    assert len(prov["config_fingerprint"]) == 64
    assert prov["train_races_count"] == 6
    assert prov["validation_races_count"] == 2
    assert prov["test_sealed"] is True

    # Both model.json and manifest.json are written in canonical JSON form.
    version_dir = tmp_path / "models" / f"version={version}"
    model_bytes = (version_dir / MODEL_FILENAME).read_bytes()
    manifest_bytes = (version_dir / MANIFEST_FILENAME).read_bytes()
    assert model_bytes == canonical_json_bytes(json.loads(model_bytes.decode("utf-8")))
    assert manifest_bytes == canonical_json_bytes(json.loads(manifest_bytes.decode("utf-8")))
    assert (version_dir / PUBLISHED_MARKER).is_file()

    verified = run.verify(version)
    assert verified.ok, verified.reason
    assert verified.summary["files_checked"] == 1
    assert verified.summary["errors"] == []


def test_train_input_fingerprint_is_complete_and_order_independent() -> None:
    base = _rich_train_race()
    base_fp = race_input_fingerprint([base])
    assert base_fp["input_fingerprint_scheme"] == "m7-race-model-input-fingerprint/v2"

    # Reordering rows in lists and DataFrames does not alter the fingerprint.
    reordered = copy.deepcopy(base)
    reordered.laps = reordered.laps.sample(frac=1.0, random_state=7).reset_index(drop=True)
    reordered.intervals = reordered.intervals.sample(frac=1.0, random_state=9).reset_index(drop=True)
    reordered.stints = list(reversed(reordered.stints))
    reordered.pit = list(reversed(reordered.pit))
    reordered.race_control = list(reversed(reordered.race_control))
    reordered.grid = list(reversed(reordered.grid))
    reordered.result = list(reversed(reordered.result))
    assert race_input_fingerprint([reordered])["input_fingerprint"] == base_fp["input_fingerprint"]

    # Modifying any concrete training observation (even keeping counts and sums identical)
    # MUST change the fingerprint.
    def clone():
        return copy.deepcopy(base)

    # 1. laps: swap delta across two laps so laps_count and lap_duration sum stay identical
    r_laps = clone()
    r_laps.laps.loc[0, "lap_duration"] += 0.25
    r_laps.laps.loc[1, "lap_duration"] -= 0.25
    assert race_input_fingerprint([r_laps])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 2. laps flag: flip is_pit_out_lap
    r_pitout = clone()
    r_pitout.laps.loc[0, "is_pit_out_lap"] = not bool(r_pitout.laps.loc[0, "is_pit_out_lap"])
    assert race_input_fingerprint([r_pitout])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 3. stints: change compound without changing stint count
    r_stints = clone()
    r_stints.stints[0]["compound"] = "SOFT"
    assert race_input_fingerprint([r_stints])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 4. pit: change pit_duration without changing pit count
    r_pit = clone()
    r_pit.pit[0]["pit_duration"] = 25.9
    assert race_input_fingerprint([r_pit])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 5. race_control: change lap_number without changing message count
    r_rc = clone()
    r_rc.race_control[0]["lap_number"] = 11
    assert race_input_fingerprint([r_rc])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 6. intervals: change a single interval value
    r_iv = clone()
    r_iv.intervals.loc[1, "interval"] = 0.42
    assert race_input_fingerprint([r_iv])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 7. grid: change qualifying lap_duration without changing grid count
    r_grid = clone()
    r_grid.grid[0]["lap_duration"] += 0.15
    assert race_input_fingerprint([r_grid])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 8. result: swap two finishing positions without changing result count
    r_res = clone()
    r_res.result[0]["position"], r_res.result[1]["position"] = (
        r_res.result[1]["position"],
        r_res.result[0]["position"],
    )
    assert race_input_fingerprint([r_res])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 9. drivers: change team_name
    r_drv = clone()
    r_drv.drivers[1]["team_name"] = "OtherTeam"
    assert race_input_fingerprint([r_drv])["input_fingerprint"] != base_fp["input_fingerprint"]

    # 10. ref metadata: change location
    r_ref = replace(clone(), ref=replace(base.ref, location="Imola"))
    assert race_input_fingerprint([r_ref])["input_fingerprint"] != base_fp["input_fingerprint"]


def test_train_input_fingerprint_is_invariant_to_validation_changes(tmp_path: Path) -> None:
    r_orig = _make_runner(tmp_path / "orig", altered_validation=False)
    r_alt = _make_runner(tmp_path / "alt", altered_validation=True)
    t_orig = r_orig.train()
    t_alt = r_alt.train()
    assert t_orig.ok and t_alt.ok
    assert t_orig.summary["input_fingerprint"] == t_alt.summary["input_fingerprint"]


def test_atomic_publication_failure_recovery_and_idempotent_retry(tmp_path: Path, monkeypatch) -> None:
    store = DataStore.local(str(tmp_path))
    registry = ModelRegistry(store, "models")
    artifact = ModelArtifact(
        version=new_version(),
        created_at="2026-11-22T00:00:00+00:00",
        settings={"random_seed": 42},
        level1={"phi": 0.4, "sigma_eps": 0.3},
        season={},
        events={},
        forecast={},
        transition={},
        races=[],
        provenance={"schema_version": 2},
    )

    real_write = store.backend.write_bytes_if_absent

    # Fail while writing manifest.json (leaving an orphan model.json without PUBLISHED).
    def fail_manifest(key, data, *, content_type=None):
        if key.endswith(f"/{MANIFEST_FILENAME}"):
            raise OSError("simulated storage outage during manifest write")
        return real_write(key, data, content_type=content_type)

    monkeypatch.setattr(store.backend, "write_bytes_if_absent", fail_manifest)
    with pytest.raises(OSError, match="manifest write"):
        registry.save(artifact)

    # Orphan model.json is invisible to versions() and rejected by verify(), load(), promote().
    assert artifact.version not in registry.versions()
    assert not registry.verify(artifact.version)["ok"]
    with pytest.raises(ModelPackageIntegrityError):
        registry.load(artifact.version)
    with pytest.raises(ModelPackageIntegrityError):
        registry.promote(artifact.version, {"reason": "must_fail"})
    assert registry.latest_version() is None

    # Next attempt fails while writing PUBLISHED (both model.json and manifest.json exist).
    def fail_published(key, data, *, content_type=None):
        if key.endswith(f"/{PUBLISHED_MARKER}"):
            raise OSError("simulated storage outage during PUBLISHED commit")
        return real_write(key, data, content_type=content_type)

    monkeypatch.setattr(store.backend, "write_bytes_if_absent", fail_published)
    with pytest.raises(OSError, match="PUBLISHED commit"):
        registry.save(artifact)

    assert artifact.version not in registry.versions()
    with pytest.raises(ModelPackageIntegrityError):
        registry.load(artifact.version)
    with pytest.raises(ModelPackageIntegrityError):
        registry.promote(artifact.version, {"reason": "still_unpublished"})

    # Restore storage -> retrying save() recovers cleanly and commits PUBLISHED.
    monkeypatch.setattr(store.backend, "write_bytes_if_absent", real_write)
    uri = registry.save(artifact)
    assert artifact.version in registry.versions()
    assert registry.verify(artifact.version)["ok"]

    # Retrying save() with identical artifact is safe and idempotent.
    assert registry.save(artifact) == uri

    # Attempting to overwrite the same version with different content fails with FileExistsError.
    mutated = ModelArtifact(
        version=artifact.version,
        created_at="2026-11-22T01:00:00+00:00",
        settings={"random_seed": 99},
        level1={"phi": 0.4, "sigma_eps": 0.3},
        season={},
        events={},
        forecast={},
        transition={},
        races=[],
        provenance={"schema_version": 2},
    )
    with pytest.raises(FileExistsError):
        registry.save(mutated)

    # Promotion and load now succeed.
    registry.promote(artifact.version, {"reason": "recovered"})
    assert registry.latest_version() == artifact.version
    assert registry.load(artifact.version).version == artifact.version


def test_verify_load_and_promote_reject_tampering_missing_marker_and_extra_files(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    version = trained.summary["version"]
    version_dir = tmp_path / "models" / f"version={version}"

    # Nonexistent version raises LookupError.
    with pytest.raises(LookupError):
        run.registry.promote("nonexistent-version", {"reason": "test"})
    with pytest.raises(LookupError):
        run.registry.load("nonexistent-version")

    # 1. Tamper with model.json -> verify fails AND load/promote/read_model_bytes all refuse.
    model_file = version_dir / MODEL_FILENAME
    original_bytes = model_file.read_bytes()
    model_file.write_bytes(original_bytes + b" ")
    tampered = run.verify(version)
    assert not tampered.ok
    assert any("sha256 mismatch" in err for err in tampered.summary["errors"])
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.load(version)
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.read_model_bytes(version)
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.promote(version, {"reason": "tampered"})
    assert run.registry.latest_version() is None

    # 2. Restore model.json and remove PUBLISHED marker -> verify/load/promote all refuse.
    model_file.write_bytes(original_bytes)
    marker = version_dir / PUBLISHED_MARKER
    marker_bytes = marker.read_bytes()
    marker.unlink()
    missing = run.verify(version)
    assert not missing.ok
    assert any(PUBLISHED_MARKER in err for err in missing.summary["errors"])
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.load(version)
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.promote(version, {"reason": "missing_marker"})

    # 3. Corrupt PUBLISHED marker hash -> rejected.
    marker.write_bytes(b"sha256:" + b"0" * 64 + b"\n")
    bad_marker = run.verify(version)
    assert not bad_marker.ok
    assert any("mismatch" in err for err in bad_marker.summary["errors"])
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.load(version)
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.promote(version, {"reason": "bad_marker"})

    # 4. Restore marker, add unexpected extra file in version directory -> rejected.
    marker.write_bytes(marker_bytes)
    stray = version_dir / "stray.txt"
    stray.write_text("unexpected", encoding="utf-8")
    extra_check = run.verify(version)
    assert not extra_check.ok
    assert any("unexpected extra" in err for err in extra_check.summary["errors"])
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.load(version)
    with pytest.raises(ModelPackageIntegrityError):
        run.registry.promote(version, {"reason": "extra_file"})
    stray.unlink()

    # 5. Clean state passes verification, load, and promotion.
    assert run.verify(version).ok
    run.registry.promote(version, {"reason": "verified"})
    assert run.registry.latest_version() == version
    assert run.registry.load().version == version


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
    assert interp["events"]["pit_loss_green_s"] is not None
    assert {"SC", "VSC", "RED", "DNF"} <= set(interp["events"]["global_hazards"])
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

    eval_jsons = list((tmp_path / "models" / "reports").glob("evaluate_validation*.json"))
    assert eval_jsons
    eval_doc = json.loads(eval_jsons[-1].read_text(encoding="utf-8"))
    assert eval_doc["provenance"]["input_fingerprint"] == artifact.provenance["input_fingerprint"]
    assert eval_doc["generalization_diagnosis"]["evaluated_splits"] == ["train", "validation"]
    assert eval_doc["interpretability"]["method"] == "structural_parameter_decomposition"
    assert eval_doc["package_verification"]["ok"] is True


# -- 3. Strict Serving Input & Output Contract Validation ---------------------


def test_serving_request_contract_strict_validation() -> None:
    assert validate_forecast_request({"mode": "live"}) == {"mode": "live"}
    assert validate_forecast_request({"mode": "live", "source": "live"})["mode"] == "live"
    assert validate_forecast_request({"mode": "live", "model_version": "v1"})["model_version"] == "v1"
    assert validate_forecast_request({"session_key": 9998, "lap": 30, "total_laps": 50, "source": "raw"})["session_key"] == 9998

    # Live mode must reject conflicting on-demand fields and bad model_version / source.
    for bad_live in (
        {"mode": "live", "session_key": 9998},
        {"mode": "live", "lap": 10},
        {"mode": "live", "total_laps": 50},
        {"mode": "live", "year": 2025},
        {"mode": "live", "source": "raw"},
        {"mode": "live", "source": None},
        {"mode": "live", "model_version": ""},
        {"mode": "live", "model_version": "../bad"},
        {"mode": None, "session_key": 9998},
        {"mode": "batch"},
    ):
        with pytest.raises(ValueError):
            validate_forecast_request(bad_live)

    # On-demand validation: types, ranges, lap <= total_laps, unknown keys, source.
    for bad_req in (
        {"session_key": 9998, "unexpected_field": 1},
        {"session_key": -1},
        {"session_key": True},
        {"session_key": 9998.5},
        {"session_key": 9998, "lap": 0},
        {"session_key": 9998, "lap": True},
        {"session_key": 9998, "lap": 10.5},
        {"session_key": 9998, "total_laps": 0},
        {"session_key": 9998, "total_laps": False},
        {"session_key": 9998, "lap": 51, "total_laps": 50},
        {"session_key": 9998, "year": 2022},
        {"session_key": 9998, "source": "ftp"},
        {"session_key": 9998, "source": None},
        {"session_key": 9998, "model_version": "  "},
        {"session_key": 9998, "model_version": "a/b"},
    ):
        with pytest.raises(ValueError):
            validate_forecast_request(bad_req)


def test_serving_response_contract_strict_validation() -> None:
    good_resp = {
        "model_version": "v1",
        "session_key": 9998,
        "race": "2025 Spa",
        "lap": 10,
        "total_laps": 50,
        "regime_at_k": "GREEN",
        "scenarios": 200,
        "drivers": [
            {
                "driver_number": 1,
                "driver": "VER",
                "p_dnf": 0.05,
                "p_position": [0.7, 0.3],
                "p_win": 0.7,
                "p_podium": 1.0,
                "p_points": 1.0,
                "expected_position": 1.3,
                "retired_at_k": False,
            },
            {
                "driver_number": 4,
                "driver": "NOR",
                "p_dnf": 0.04,
                "p_position": [0.3, 0.7],
                "p_win": 0.3,
                "p_podium": 1.0,
                "p_points": 1.0,
                "expected_position": 1.7,
                "retired_at_k": False,
            },
        ],
        "matrix": [[0.7, 0.3], [0.3, 0.7]],
        "ahead": [[0.0, 0.7], [0.3, 0.0]],
    }
    validate_forecast_response(good_resp)

    invalid_cases = [
        # Missing/invalid mandatory metadata & lap ranges
        {**good_resp, "model_version": ""},
        {**good_resp, "model_version": None},
        {**good_resp, "lap": 0},
        {**good_resp, "lap": True},
        {**good_resp, "lap": 10.5},
        {**good_resp, "total_laps": 0},
        {**good_resp, "lap": 51, "total_laps": 50},
        {**good_resp, "regime_at_k": "UNKNOWN"},
        # Duplicate driver IDs or driver numbers
        {
            **good_resp,
            "drivers": [
                {**good_resp["drivers"][0], "driver": "VER"},
                {**good_resp["drivers"][1], "driver": "VER"},
            ],
        },
        {
            **good_resp,
            "drivers": [
                {**good_resp["drivers"][0], "driver_number": 1},
                {**good_resp["drivers"][1], "driver_number": 1},
            ],
        },
        # Strictly reject out-of-[0, 1] p_dnf even within numerical atol
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "p_dnf": -1e-6}, good_resp["drivers"][1]],
        },
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "p_dnf": 1.0 + 1e-6}, good_resp["drivers"][1]],
        },
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "p_dnf": float("nan")}, good_resp["drivers"][1]],
        },
        # Strictly reject negative or >1 matrix entries even within numerical atol
        {**good_resp, "matrix": [[1.0 + 1e-6, -1e-6], [-1e-6, 1.0 + 1e-6]]},
        # Bad matrix sums and malformed structures (ragged, 1D, booleans, strings)
        {**good_resp, "matrix": [[0.9, 0.9], [0.1, 0.1]]},
        {**good_resp, "matrix": [[0.7, 0.3], [1.0]]},
        {**good_resp, "matrix": [0.5, 0.5]},
        {**good_resp, "matrix": [[True, False], [False, True]]},
        {**good_resp, "matrix": [["0.7", "0.3"], ["0.3", "0.7"]]},
        # Driver-to-matrix mismatch (p_position / p_win / expected_position)
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "p_position": [0.5, 0.5]}, good_resp["drivers"][1]],
        },
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "p_win": 0.2}, good_resp["drivers"][1]],
        },
        {
            **good_resp,
            "drivers": [{**good_resp["drivers"][0], "expected_position": 2.0}, good_resp["drivers"][1]],
        },
        # Malformed ahead matrix
        {**good_resp, "ahead": [[0.2, 0.7], [0.3, 0.0]]},
        {**good_resp, "ahead": [[0.0, 0.8], [0.8, 0.0]]},
    ]
    for payload in invalid_cases:
        with pytest.raises(ValueError):
            validate_forecast_response(payload)


# -- 4. CLI verify-m7, Lazy MLflow Isolation & Pyfunc Forecast Parity ---------


def test_cli_verify_m7_and_log_mlflow_flag_validation(tmp_path: Path) -> None:
    run = _make_runner(tmp_path)
    trained = run.train()
    version = trained.summary["version"]

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("race_model:\n  output_path: models\n", encoding="utf-8")

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


def test_base_race_model_imports_never_load_mlflow() -> None:
    """Importing core race_model modules must never eagerly import mlflow, even when installed."""
    code = (
        "import sys\n"
        "import f1_race_intelligence.race_model.contracts\n"
        "import f1_race_intelligence.race_model.artifact\n"
        "import f1_race_intelligence.race_model.runner\n"
        "import f1_race_intelligence.race_model.lambda_handler\n"
        "import f1_race_intelligence.race_model.forecast\n"
        "assert 'mlflow' not in sys.modules, 'mlflow was imported eagerly!'\n"
        "assert 'f1_race_intelligence.race_model.mlflow_pyfunc' not in sys.modules\n"
    )
    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src_dir}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"lazy import check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"


def test_optional_mlflow_logging_and_pyfunc_forecast_parity(tmp_path: Path) -> None:
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

    # Run real forecast via RaceModelRunner.forecast (also validating end-to-end serving contract).
    runner_fc = run.forecast(session_key=7, lap=10, version=version, year=2025)
    assert runner_fc.ok, runner_fc.reason
    runner_payload = run.output_store.backend.read_json(runner_fc.summary["forecast_uri"])
    validate_forecast_response(runner_payload)

    # Load from MLflow Registry and run real prediction on the exact same race state.
    mlflow.set_tracking_uri(info["tracking_uri"])
    loaded = mlflow.pyfunc.load_model(info["model_uri"])
    val_ref = next(r for r in run.catalog.races([2025]) if r.session_key == 7)
    val_race = run.catalog.load(val_ref)
    race_json = canonical_json_bytes(race_to_forecast_payload(val_race)).decode("utf-8")

    out = loaded.predict(
        pd.DataFrame(
            [
                {
                    "race_json": race_json,
                    "lap": 10,
                    "total_laps": runner_payload["total_laps"],
                    "seed": int(run.cfg.random_seed),
                }
            ]
        )
    )
    assert out.loc[0, "model_version"] == version
    assert int(out.loc[0, "session_key"]) == 7
    assert int(out.loc[0, "lap"]) == 10
    assert int(out.loc[0, "total_laps"]) == runner_payload["total_laps"]

    pyfunc_payload = json.loads(out.loc[0, "forecast_json"])
    validate_forecast_response(pyfunc_payload)

    # Exact parity between MLflow Pyfunc prediction and RaceModelRunner.forecast.
    for key in ("model_version", "session_key", "race", "lap", "total_laps", "regime_at_k", "scenarios"):
        assert pyfunc_payload[key] == runner_payload[key], f"mismatch on {key}"
    assert pyfunc_payload["drivers"] == runner_payload["drivers"]
    assert pyfunc_payload["matrix"] == runner_payload["matrix"]
    assert pyfunc_payload["ahead"] == runner_payload["ahead"]

    # Verify all expected tracking artifacts were logged.
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
