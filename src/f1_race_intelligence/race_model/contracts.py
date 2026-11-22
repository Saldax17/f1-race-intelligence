"""Shared M7 contracts for the traditional race model: provenance, manifests,
generalization diagnosis, structural interpretability, Markdown reporting,
serving schema validation, and optional lazy MLflow logging.

Every function here is pure Python / NumPy / stdlib and never imports MLflow at
module import time, keeping the M1-M6 and AWS Fargate runtime image isolated
from training-only extras (``requirements/training-lock.txt``).
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from f1_race_intelligence.modeling.traceability import (
    config_fingerprint,
    dependency_lock_sha256,
    source_revision,
)
from f1_race_intelligence.storage.backends.base import join_key

SCHEMA_VERSION = 2
PIPELINE_NAME = "race_model"
PIPELINE_VERSION = "1.0.0"
MODEL_TYPE = "hierarchical_state_space"
ESTIMATOR_NAME = "f1_race_intelligence.race_model.RaceModelRunner"
INPUT_FINGERPRINT_SCHEME = "m7-race-model-input-fingerprint/v1"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PUBLISHED_MARKER = "PUBLISHED"
MODEL_FILENAME = "model.json"

# Generalization diagnosis thresholds (mirroring modeling/dl.py discipline;
# evaluated strictly on train and validation, never on the sealed test split).
OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO = 1.25
OVERFITTING_VAL_TO_TRAIN_RPS_RATIO = 1.35
CALIBRATION_MAX_Z2 = 1.50
_MIN_DENOM = 1e-9

STRUCTURAL_INTERPRETABILITY_NOTE = (
    "Structural parameters (AR(1) persistence phi, clean-lap noise sigma_eps, tyre "
    "degradation slopes, Kalman variance components and event hazards) are fitted "
    "strictly on training races (2023-24). Initial team/driver prior variances "
    "(p0_gamma, p0_delta) are fixed priors rather than data-identified parameters; "
    "variance shares reflect the innovation and race-level decomposition across "
    "fitted Level-2 components."
)

MLFLOW_INSTALL_HINT = (
    "python -m pip install -r requirements/training-lock.txt && "
    "python -m pip install -e '.[training]' --no-deps"
)

ALLOWED_FORECAST_REQUEST_KEYS = frozenset(
    {"session_key", "lap", "source", "model_version", "total_laps", "year", "mode"}
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 JSON (sorted keys, compact separators) for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=float,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


# -- 1. Provenance & Input Fingerprinting -------------------------------------


def race_input_fingerprint(train_races: Sequence[Any]) -> Dict[str, Any]:
    """Deterministic SHA-256 fingerprint over training races only.

    Hashing strictly ``train_races`` preserves the anti-leakage invariance of
    ``test_race_model_isolation.py``: replacing validation races changes neither
    fitted parameters nor the training input fingerprint, and sealed test races
    are never touched.
    """
    entries: List[Dict[str, Any]] = []
    for race in sorted(train_races, key=lambda r: (r.ref.year, r.ref.session_key)):
        laps_count = int(len(race.laps)) if race.laps is not None else 0
        valid_laps = (
            float(np.nansum(race.laps["lap_duration"].to_numpy(dtype=float)))
            if laps_count and "lap_duration" in race.laps.columns
            else 0.0
        )
        race_digest_payload = {
            "year": int(race.ref.year),
            "meeting_key": int(race.ref.meeting_key),
            "session_key": int(race.ref.session_key),
            "location": str(race.ref.location),
            "date_start": str(race.ref.date_start),
            "drivers": sorted(int(d) for d in race.driver_numbers),
            "laps_count": laps_count,
            "laps_duration_sum": round(valid_laps, 6),
            "stints_count": len(race.stints),
            "pit_count": len(race.pit),
            "race_control_count": len(race.race_control),
            "result_count": len(race.result),
            "grid_count": len(race.grid),
        }
        entries.append(
            {
                "year": int(race.ref.year),
                "session_key": int(race.ref.session_key),
                "tag": race.tag,
                "sha256": sha256_bytes(_canonical_json_bytes(race_digest_payload)),
            }
        )
    doc = {"scheme": INPUT_FINGERPRINT_SCHEME, "inputs": entries}
    return {
        "input_fingerprint": f"sha256:{sha256_bytes(_canonical_json_bytes(doc))}",
        "input_fingerprint_scheme": INPUT_FINGERPRINT_SCHEME,
        "train_races_count": len(entries),
        "train_inputs": entries,
    }


def build_provenance(
    settings_dump: Mapping[str, Any],
    train_races: Sequence[Any],
    *,
    validation_races_count: int,
    test_sealed: bool,
    random_seed: int,
) -> Dict[str, Any]:
    """Build M6/M7-aligned provenance metadata for a race-model version."""
    commit, dirty = source_revision()
    lock_sha = dependency_lock_sha256()
    cfg_safe = {
        k: v
        for k, v in settings_dump.items()
        if k not in {"raw_path", "output_path", "input_storage", "output_storage"}
    }
    fp = race_input_fingerprint(train_races)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "model_type": MODEL_TYPE,
        "estimator": ESTIMATOR_NAME,
        "source_commit": commit,
        "source_dirty": dirty,
        "dependency_lock_sha256": lock_sha,
        "config_fingerprint": config_fingerprint(cfg_safe),
        "input_fingerprint": fp["input_fingerprint"],
        "input_fingerprint_scheme": fp["input_fingerprint_scheme"],
        "train_races_count": fp["train_races_count"],
        "validation_races_count": int(validation_races_count),
        "test_sealed": bool(test_sealed),
        "random_seed": int(random_seed),
    }


# -- 2. Immutable Package Manifest & Verification -----------------------------


def write_version_manifest(
    backend: Any,
    version_prefix: str,
    *,
    version: str,
    created_at: str,
    model_format: str,
    model_bytes: bytes,
    provenance: Mapping[str, Any],
    parent_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``manifest.json`` and ``PUBLISHED`` marker alongside ``model.json``.

    Mirrors ``modeling/model_package.py``: conditional writes
    (``write_bytes_if_absent``), canonical JSON serialization, and ``PUBLISHED``
    written last containing ``sha256:<digest of manifest.json>``.
    """
    model_sha = sha256_bytes(model_bytes)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": model_format,
        "version": version,
        "created_at": created_at,
        "model_type": MODEL_TYPE,
        "estimator": ESTIMATOR_NAME,
        "parent_version": parent_version,
        "provenance": dict(provenance),
        "files": [
            {
                "path": MODEL_FILENAME,
                "sha256": model_sha,
                "size_bytes": len(model_bytes),
            }
        ],
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    manifest_key = join_key(version_prefix, MANIFEST_FILENAME)
    published_key = join_key(version_prefix, PUBLISHED_MARKER)

    if not backend.write_bytes_if_absent(manifest_key, manifest_bytes, content_type="application/json"):
        existing = backend.read_bytes(manifest_key)
        if existing != manifest_bytes:
            raise FileExistsError(f"conflicting manifest already exists at {backend.uri(manifest_key)}")

    marker_bytes = f"sha256:{manifest_sha}\n".encode("utf-8")
    if not backend.write_bytes_if_absent(published_key, marker_bytes, content_type="text/plain; charset=utf-8"):
        existing_marker = backend.read_bytes(published_key)
        if existing_marker != marker_bytes:
            raise FileExistsError(f"conflicting PUBLISHED marker already exists at {backend.uri(published_key)}")

    return {
        "manifest_uri": backend.uri(manifest_key),
        "published_uri": backend.uri(published_key),
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "model_size_bytes": len(model_bytes),
    }


def verify_version_package(
    backend: Any,
    version_prefix: str,
    version: str,
    *,
    expected_format: str = "f1-race-model/1",
) -> Dict[str, Any]:
    """Read-only SHA-256 verification of a saved race-model version package.

    Checks ``PUBLISHED`` marker -> ``manifest.json`` -> ``model.json`` bytes and
    format integrity, mirroring ``modeling/verification.py`` and
    ``modeling/model_package.py``.
    """
    errors: List[str] = []
    manifest_key = join_key(version_prefix, MANIFEST_FILENAME)
    published_key = join_key(version_prefix, PUBLISHED_MARKER)
    files_checked = 0
    manifest_sha: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None

    if not backend.exists(published_key):
        errors.append(f"missing {PUBLISHED_MARKER} marker at {backend.uri(published_key)}")
        declared_manifest_sha = None
    else:
        raw_marker = backend.read_bytes(published_key).decode("utf-8", errors="replace").strip()
        if not raw_marker.startswith("sha256:") or len(raw_marker) != 7 + 64:
            errors.append(f"invalid {PUBLISHED_MARKER} marker format: {raw_marker!r}")
            declared_manifest_sha = None
        else:
            declared_manifest_sha = raw_marker[7:]

    if not backend.exists(manifest_key):
        errors.append(f"missing {MANIFEST_FILENAME} at {backend.uri(manifest_key)}")
    else:
        manifest_bytes = backend.read_bytes(manifest_key)
        manifest_sha = sha256_bytes(manifest_bytes)
        if declared_manifest_sha is not None and manifest_sha != declared_manifest_sha:
            errors.append(
                f"{MANIFEST_FILENAME}: sha256 mismatch against {PUBLISHED_MARKER} "
                f"(expected {declared_manifest_sha}, got {manifest_sha})"
            )
        try:
            parsed = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse {MANIFEST_FILENAME}: {exc}")
        else:
            if not isinstance(parsed, dict):
                errors.append(f"{MANIFEST_FILENAME} root must be a JSON object")
            else:
                manifest = parsed
                if manifest.get("version") != version:
                    errors.append(
                        f"{MANIFEST_FILENAME} version mismatch: expected {version!r}, got {manifest.get('version')!r}"
                    )
                if manifest.get("format") != expected_format:
                    errors.append(
                        f"{MANIFEST_FILENAME} format mismatch: expected {expected_format!r}, got {manifest.get('format')!r}"
                    )
                declared_files = manifest.get("files")
                if not isinstance(declared_files, list) or not declared_files:
                    errors.append(f"{MANIFEST_FILENAME} has no declared files")
                else:
                    for entry in declared_files:
                        rel_path = entry.get("path")
                        expected_sha = entry.get("sha256")
                        expected_size = entry.get("size_bytes")
                        file_key = join_key(version_prefix, str(rel_path))
                        if not backend.exists(file_key):
                            errors.append(f"{rel_path}: missing artifact at {backend.uri(file_key)}")
                            continue
                        content = backend.read_bytes(file_key)
                        files_checked += 1
                        actual_sha = sha256_bytes(content)
                        if actual_sha != expected_sha:
                            errors.append(
                                f"{rel_path}: sha256 mismatch (expected {expected_sha}, got {actual_sha})"
                            )
                        if expected_size != len(content):
                            errors.append(
                                f"{rel_path}: size_bytes mismatch (expected {expected_size}, got {len(content)})"
                            )
                        if rel_path == MODEL_FILENAME:
                            try:
                                model_doc = json.loads(content.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                errors.append(f"{rel_path}: invalid JSON ({exc})")
                            else:
                                if model_doc.get("format") != expected_format:
                                    errors.append(f"{rel_path}: unexpected model format {model_doc.get('format')!r}")
                                if model_doc.get("version") != version:
                                    errors.append(f"{rel_path}: version mismatch {model_doc.get('version')!r}")

    return {
        "ok": not errors,
        "version": version,
        "manifest_uri": backend.uri(manifest_key),
        "manifest_sha256": manifest_sha,
        "files_checked": files_checked,
        "provenance": manifest.get("provenance") if isinstance(manifest, dict) else None,
        "errors": errors,
    }


# -- 3. Generalization Diagnosis (strictly train & validation) ----------------


def diagnose_race_model_generalization(
    *,
    prior_check: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    noise_scale_rps: Optional[Mapping[str, Any]] = None,
    baseline_key: str = "persist",
) -> Dict[str, Any]:
    """Transparent over/underfitting verdict strictly on ``train`` and ``validation``.

    Mirrors ``modeling.dl.diagnose_generalization`` with the exact same 4-state
    vocabulary (``balanced``, ``overfitting_risk``, ``underfitting``,
    ``insufficient_metrics``) and strict ``evaluated_splits = ["train", "validation"]``
    anti-snooping guarantee (sealed ``test`` never participates).
    """
    train_pc = prior_check.get("train") or {}
    val_pc = prior_check.get("validation") or {}
    train_rmse = float(train_pc["rmse"]) if _is_finite_real(train_pc.get("rmse")) else None
    val_rmse = float(val_pc["rmse"]) if _is_finite_real(val_pc.get("rmse")) else None
    train_z2 = float(train_pc["z2"]) if _is_finite_real(train_pc.get("z2")) else None
    val_z2 = float(val_pc["z2"]) if _is_finite_real(val_pc.get("z2")) else None

    overall = (validation_summary.get("by_bucket") or {}).get("all") or {}
    val_rps = float(overall["model_rps"]) if _is_finite_real(overall.get("model_rps")) else None
    base_rps = float(overall.get(f"{baseline_key}_rps")) if _is_finite_real(overall.get(f"{baseline_key}_rps")) else None
    trans_rps = float(overall["transition_rps"]) if _is_finite_real(overall.get("transition_rps")) else None

    # Best training calibration RPS (from noise_scale selection on training races), if recorded.
    train_cal_rps: Optional[float] = None
    if isinstance(noise_scale_rps, Mapping):
        overall_cal = noise_scale_rps.get("overall")
        if isinstance(overall_cal, Mapping):
            finite_vals = [float(v) for v in overall_cal.values() if _is_finite_real(v)]
            if finite_vals:
                train_cal_rps = min(finite_vals)

    val_to_train_prior_rmse_ratio = (
        val_rmse / train_rmse
        if train_rmse is not None and val_rmse is not None and train_rmse > _MIN_DENOM
        else None
    )
    val_to_train_rps_ratio = (
        val_rps / train_cal_rps
        if train_cal_rps is not None and val_rps is not None and train_cal_rps > _MIN_DENOM
        else None
    )
    val_vs_baseline_rps_delta = (
        val_rps - base_rps if val_rps is not None and base_rps is not None else None
    )
    val_vs_transition_rps_delta = (
        val_rps - trans_rps if val_rps is not None and trans_rps is not None else None
    )

    if val_rps is None or (train_rmse is None and val_rmse is None):
        status = "insufficient_metrics"
        explanation = "Cannot diagnose generalization: validation RPS or prior-check RMSE is missing."
    elif base_rps is not None and val_rps >= base_rps:
        status = "underfitting"
        explanation = (
            f"Underfitting: validation RPS {val_rps:.4f} does not beat the {baseline_key!r} "
            f"baseline ({base_rps:.4f})."
        )
    elif (
        (val_to_train_prior_rmse_ratio is not None and val_to_train_prior_rmse_ratio >= OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO)
        or (val_z2 is not None and val_z2 >= CALIBRATION_MAX_Z2)
        or (val_to_train_rps_ratio is not None and val_to_train_rps_ratio >= OVERFITTING_VAL_TO_TRAIN_RPS_RATIO)
    ):
        status = "overfitting_risk"
        reasons = []
        if val_to_train_prior_rmse_ratio is not None and val_to_train_prior_rmse_ratio >= OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:
            reasons.append(
                f"validation prior RMSE ratio {val_to_train_prior_rmse_ratio:.2f}x >= "
                f"{OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:.2f}x"
            )
        if val_z2 is not None and val_z2 >= CALIBRATION_MAX_Z2:
            reasons.append(f"validation calibration z² {val_z2:.2f} >= {CALIBRATION_MAX_Z2:.2f}")
        if val_to_train_rps_ratio is not None and val_to_train_rps_ratio >= OVERFITTING_VAL_TO_TRAIN_RPS_RATIO:
            reasons.append(
                f"validation/train calibration RPS ratio {val_to_train_rps_ratio:.2f}x >= "
                f"{OVERFITTING_VAL_TO_TRAIN_RPS_RATIO:.2f}x"
            )
        explanation = "Overfitting risk: " + "; ".join(reasons) + "."
    else:
        status = "balanced"
        parts = [f"validation RPS {val_rps:.4f}"]
        if val_vs_baseline_rps_delta is not None:
            parts.append(f"beats {baseline_key!r} by {-val_vs_baseline_rps_delta:.4f}")
        if val_to_train_prior_rmse_ratio is not None:
            parts.append(f"prior RMSE ratio {val_to_train_prior_rmse_ratio:.2f}x (< {OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO:.2f}x)")
        if val_z2 is not None:
            parts.append(f"calibration z² {val_z2:.2f} (< {CALIBRATION_MAX_Z2:.2f})")
        explanation = "Balanced: " + ", ".join(parts) + "."

    return {
        "status": status,
        "model_key": "race_model",
        "baseline_key": baseline_key,
        "train_prior_rmse": train_rmse,
        "validation_prior_rmse": val_rmse,
        "val_to_train_prior_rmse_ratio": val_to_train_prior_rmse_ratio,
        "train_prior_z2": train_z2,
        "validation_prior_z2": val_z2,
        "train_calibration_rps": train_cal_rps,
        "validation_rps": val_rps,
        "baseline_validation_rps": base_rps,
        "transition_validation_rps": trans_rps,
        "val_vs_baseline_rps_delta": val_vs_baseline_rps_delta,
        "val_vs_transition_rps_delta": val_vs_transition_rps_delta,
        "thresholds": {
            "overfitting_val_to_train_prior_rmse_ratio": OVERFITTING_VAL_TO_TRAIN_PRIOR_RMSE_RATIO,
            "overfitting_val_to_train_rps_ratio": OVERFITTING_VAL_TO_TRAIN_RPS_RATIO,
            "calibration_max_z2": CALIBRATION_MAX_Z2,
        },
        "explanation": explanation,
        "evaluated_splits": ["train", "validation"],
    }


# -- 4. Structural Interpretability -------------------------------------------


def extract_interpretability(artifact: Any) -> Dict[str, Any]:
    """Structured interpretability summary of the fitted hierarchical state-space model.

    Mirrors ``training_report.json["interpretability"]`` in the DL pipeline:
    instead of post-hoc PFI on a black-box MLP, the traditional model exposes
    its training-fitted structural parameters (AR(1) persistence, compound
    degradation curves, Kalman variance decomposition across car/driver/circuit/
    race-day noise, and event hazards).
    """
    l1 = artifact.level1 or {}
    phi = float(l1.get("phi", 0.0))
    sigma_eps = float(l1.get("sigma_eps", 0.0))
    ar1_mult = math.sqrt((1.0 + phi) / max(1e-6, 1.0 - phi)) if abs(phi) < 1.0 else float("nan")

    tyres_raw = l1.get("tyres") or {}
    offsets = tyres_raw.get("offset") or {}
    slopes = tyres_raw.get("slope") or {}
    compounds = sorted(set(offsets) | set(slopes))
    tyre_table = [
        {
            "compound": str(c),
            "offset_pct": float(offsets.get(c, 0.0)),
            "degradation_slope_pct_per_lap": float(slopes.get(c, 0.0)),
        }
        for c in compounds
    ]
    tyre_table.sort(key=lambda row: -row["degradation_slope_pct_per_lap"])

    season_params = (artifact.season or {}).get("state", {}).get("params") or {}
    var_components = {
        "team_in_season_q_gamma": float(season_params.get("q_gamma", 0.0)) ** 2,
        "driver_in_season_q_delta": float(season_params.get("q_delta", 0.0)) ** 2,
        "team_circuit_fit_s_c": float(season_params.get("s_c", 0.0)) ** 2,
        "race_day_transient_s_w": float(season_params.get("s_w", 0.0)) ** 2,
    }
    total_var = sum(var_components.values())
    variance_decomposition = [
        {
            "component": name,
            "sd_pct": math.sqrt(val),
            "variance": val,
            "variance_share": (val / total_var) if total_var > 0 else 0.0,
        }
        for name, val in sorted(var_components.items(), key=lambda kv: -kv[1])
    ]

    events = artifact.events or {}
    hazards = events.get("hazards") or {}
    sc_by_loc = (events.get("sc_by_location") or {}) if isinstance(events.get("sc_by_location"), Mapping) else {}
    top_sc_circuits = sorted(
        [{"location": k, "sc_per_lap": float(v)} for k, v in sc_by_loc.items() if _is_finite_real(v)],
        key=lambda r: -r["sc_per_lap"],
    )[:5]

    fc = artifact.forecast or {}
    return {
        "status": "computed",
        "method": "structural_parameter_decomposition",
        "fit_split": "train",
        "level1_pace": {
            "phi_ar1": phi,
            "sigma_eps_pct": sigma_eps,
            "ar1_long_run_sd_multiplier": ar1_mult,
            "tyre_compounds": tyre_table,
        },
        "level2_season_variance": {
            "total_in_race_innovation_variance": total_var,
            "components": variance_decomposition,
            "winter_team_reset_sd_pct": float(season_params.get("q_gamma_season", 0.0)),
            "qualifying_weight_lambda": float(season_params.get("lam", 0.0)),
            "qualifying_noise_sd_pct": float(season_params.get("sigma_q", 0.0)),
        },
        "events": {
            "global_hazards": {k: float(v) for k, v in hazards.items() if _is_finite_real(v)},
            "pit_loss_green_s": events.get("pit_green"),
            "pit_loss_sc_s": events.get("pit_sc"),
            "pit_loss_vsc_s": events.get("pit_vsc"),
            "top_sc_hazard_circuits": top_sc_circuits,
        },
        "phase_noise_multipliers": {
            "default_noise_scale": fc.get("noise_scale"),
            "by_fraction": fc.get("noise_scale_by_fraction"),
        },
        "overtaking_friction": {
            "pass_cost_seconds": fc.get("pass_cost_seconds", 0.0),
            "pooled_pass_rate": events.get("pass_rate"),
            "wet_noise_multiplier": events.get("wet_noise_multiplier", 1.0),
        },
        "phase_transition_shrinkage": fc.get("transition_shrinkage_by_fraction"),
        "note": STRUCTURAL_INTERPRETABILITY_NOTE,
    }


# -- 5. Academic Markdown Evaluation Report -----------------------------------


def render_evaluation_markdown(
    *,
    report: Mapping[str, Any],
    artifact: Any,
    promoted: bool,
    verification: Optional[Mapping[str, Any]] = None,
    mlflow_info: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render an academic Markdown evaluation report mirroring ``mlp_comparison_<id>.md``."""
    version = str(report.get("version", artifact.version))
    split = str(report.get("split", "validation"))
    prov = report.get("provenance") or getattr(artifact, "provenance", None) or {}
    by_bucket = report.get("by_bucket") or {}
    calib = report.get("calibration") or {}
    diag = report.get("generalization_diagnosis") or {}
    interp = report.get("interpretability") or {}

    def fmt(val: Any, digits: int = 4) -> str:
        return f"{float(val):.{digits}f}" if _is_finite_real(val) else "n/a"

    def fmt_ci(ci: Any) -> str:
        if not isinstance(ci, Mapping) or not _is_finite_real(ci.get("mean")):
            return "n/a"
        return f"{float(ci['mean']):+.4f} [{float(ci['lo']):+.4f}, {float(ci['hi']):+.4f}]"

    lines: List[str] = [
        f"# M7 Race Model Evaluation Report (`{version}` — `{split}`)",
        "",
        "## 1. Provenance & Experimental Protocol",
        "",
        f"- **Model Version:** `{version}`",
        f"- **Model Type:** `{prov.get('model_type', MODEL_TYPE)}` (`{prov.get('estimator', ESTIMATOR_NAME)}`)",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Evaluated Split:** `{split}` (`{report.get('races', 0)}` races, `{report.get('rows', 0)}` race-lap checkpoints)",
        f"- **Source Commit:** `{prov.get('source_commit') or 'unavailable'}` (dirty: `{prov.get('source_dirty')}`)",
        f"- **Train Input Fingerprint:** `{prov.get('input_fingerprint') or 'unavailable'}`",
        f"- **Config Fingerprint:** `{prov.get('config_fingerprint') or 'unavailable'}`",
        f"- **Dependency Lock SHA-256:** `{prov.get('dependency_lock_sha256') or 'unavailable'}`",
        f"- **Test Set Sealed:** `{prov.get('test_sealed')}`",
        f"- **Promotion Gate:** beat `{diag.get('baseline_key', 'persist')}` on validation mean RPS -> promoted: `{promoted}`",
        "",
        "> **Anti-Snooping & Leakage Policy:** Every pooled quantity (AR(1) `phi`, `sigma_eps`, tyres, season hyperparameters, event hazards, race distances, transition baseline and phase noise multipliers) is fitted strictly on training races (`2023-2024`). Promotion and generalization diagnosis use `train` and `validation` exclusively; the `test` split remains sealed unless explicitly unsealed once via `m7-evaluate --split test --unseal-test`.",
        "",
        "## 2. Comparative Metrics (`persist` vs `transition` vs `race_model`)",
        "",
        "| Bucket | Race Model RPS | Persist RPS | Transition RPS | Model Pairwise LL | Transition Pairwise LL | Δ vs Persist [95% CI] | Δ vs Transition [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    bucket_order = [b for b in by_bucket if b != "all"] + (["all"] if "all" in by_bucket else [])
    for bucket in bucket_order:
        row = by_bucket[bucket]
        label = "**all (overall)**" if bucket == "all" else bucket
        lines.append(
            f"| {label} | {fmt(row.get('model_rps'))} | {fmt(row.get('persist_rps'))} | "
            f"{fmt(row.get('transition_rps'))} | {fmt(row.get('model_pll'))} | "
            f"{fmt(row.get('transition_pll'))} | {fmt_ci(row.get('delta_rps_vs_persist'))} | "
            f"{fmt_ci(row.get('delta_rps_vs_transition'))} |"
        )

    lines.extend([
        "",
        "### Event Calibration (Brier Scores)",
        "",
        "| Event | Brier Score | Notes |",
        "|---|---:|---|",
    ])
    for event_name in ("win", "podium", "points", "dnf"):
        entry = calib.get(event_name) or {}
        extra = ""
        if event_name == "dnf" and _is_finite_real(entry.get("mean_forecast")) and _is_finite_real(entry.get("observed")):
            extra = f"mean forecast {float(entry['mean_forecast']):.3f} vs observed {float(entry['observed']):.3f}"
        lines.append(f"| `{event_name}` | {fmt(entry.get('brier'))} | {extra} |")

    ratio_val = diag.get("val_to_train_prior_rmse_ratio")
    ratio_str = f"{float(ratio_val):.2f}x" if _is_finite_real(ratio_val) else "n/a"
    lines.extend([
        "",
        "## 3. Overfitting / Underfitting Diagnosis (`train` vs `validation`)",
        "",
        "| Model | Diagnosis | Train Prior RMSE | Val Prior RMSE | Prior Ratio | Val z² | Val RPS | Persist RPS | Transition RPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `race_model` | `{diag.get('status', 'n/a')}` | {fmt(diag.get('train_prior_rmse'))} | "
        f"{fmt(diag.get('validation_prior_rmse'))} | {ratio_str} | "
        f"{fmt(diag.get('validation_prior_z2'), 2)} | {fmt(diag.get('validation_rps'))} | "
        f"{fmt(diag.get('baseline_validation_rps'))} | {fmt(diag.get('transition_validation_rps'))} |",
        "",
        f"- **{diag.get('status', 'n/a')}:** {diag.get('explanation', '')}",
        "",
        "## 4. Interpretability — Structural Parameter & Variance Decomposition (`train`)",
        "",
    ])

    l1_interp = interp.get("level1_pace") or {}
    lines.append(
        f"- **Level 1 Pace (clean laps):** `phi = {fmt(l1_interp.get('phi_ar1'))}`, "
        f"`sigma_eps = {fmt(l1_interp.get('sigma_eps_pct'))}%`, "
        f"AR(1) long-run SD multiplier `= {fmt(l1_interp.get('ar1_long_run_sd_multiplier'), 3)}x`"
    )
    tyres = l1_interp.get("tyre_compounds") or []
    if tyres:
        lines.extend([
            "",
            "| Compound | Offset (% lap) | Degradation Slope (% / lap) |",
            "|---|---:|---:|",
        ])
        for t in tyres:
            lines.append(
                f"| `{t['compound']}` | {fmt(t.get('offset_pct'))} | {fmt(t.get('degradation_slope_pct_per_lap'))} |"
            )

    l2_interp = interp.get("level2_season_variance") or {}
    comps = l2_interp.get("components") or []
    if comps:
        lines.extend([
            "",
            "| Level-2 Variance Component | SD (% lap) | Variance | Share |",
            "|---|---:|---:|---:|",
        ])
        for c in comps:
            share_val = c.get("variance_share")
            share_str = f"{float(share_val) * 100:.1f}%" if _is_finite_real(share_val) else "n/a"
            lines.append(
                f"| `{c['component']}` | {fmt(c.get('sd_pct'))} | {fmt(c.get('variance'))} | {share_str} |"
            )
        lines.append("")
        lines.append(
            f"- **Winter reset SD (`q_gamma_season`):** `{fmt(l2_interp.get('winter_team_reset_sd_pct'))}%` · "
            f"**Qualifying weight (`lambda`):** `{fmt(l2_interp.get('qualifying_weight_lambda'))}` · "
            f"**Qualifying noise (`sigma_q`):** `{fmt(l2_interp.get('qualifying_noise_sd_pct'))}%`"
        )

    lines.extend([
        "",
        f"> **Interpretability Note:** {interp.get('note', STRUCTURAL_INTERPRETABILITY_NOTE)}",
        "",
        "## 5. Model Registry & Artifact Integrity Status",
        "",
        f"- **Version:** `{version}`",
        f"- **Promoted to `latest.json`:** `{promoted}`",
    ])
    if verification:
        lines.append(
            f"- **Immutable Package Verification (`PUBLISHED` + `manifest.json`):** "
            f"`ok={verification.get('ok')}` (`files_checked={verification.get('files_checked')}`, "
            f"`manifest_sha256={verification.get('manifest_sha256')}`)"
        )
    if mlflow_info:
        lines.append(
            f"- **MLflow Registry:** `{mlflow_info.get('model_uri')}` (run `{mlflow_info.get('run_id')}`)"
        )
    else:
        lines.append("- **MLflow Registry:** not logged in this run (pass `--log-mlflow` to log & register locally)")
    lines.append("")
    return "\n".join(lines)


# -- 6. Serving Input / Output Contract Validation ----------------------------


def validate_forecast_request(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an on-demand or live forecast request against the serving contract.

    Rejects unknown keys, invalid types, and missing required routing fields
    (mirroring ``serving/contract.py`` for M8).
    """
    if not isinstance(event, Mapping):
        raise ValueError("forecast event must be a JSON object")
    unknown = sorted(set(event) - ALLOWED_FORECAST_REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unknown forecast event field(s): {unknown}")

    mode = event.get("mode")
    if mode is not None:
        if mode != "live":
            raise ValueError(f"unsupported mode {mode!r}; expected 'live'")
        return dict(event)

    if "session_key" not in event or not _is_int(event["session_key"]) or int(event["session_key"]) <= 0:
        raise ValueError("event needs a positive integer 'session_key' or 'mode': 'live'")
    if "lap" in event and event["lap"] is not None:
        if not _is_int(event["lap"]) or int(event["lap"]) < 1:
            raise ValueError("'lap' must be an integer >= 1 when provided")
    if "total_laps" in event and event["total_laps"] is not None:
        if not _is_int(event["total_laps"]) or int(event["total_laps"]) < 1:
            raise ValueError("'total_laps' must be an integer >= 1 when provided")
    if "year" in event and event["year"] is not None:
        if not _is_int(event["year"]) or int(event["year"]) < 2023:
            raise ValueError("'year' must be an integer >= 2023 when provided")
    source = event.get("source", "raw")
    if source not in ("raw", "live"):
        raise ValueError(f"'source' must be 'raw' or 'live', got {source!r}")
    if "model_version" in event and event["model_version"] is not None:
        if not isinstance(event["model_version"], str) or not event["model_version"].strip():
            raise ValueError("'model_version' must be a non-empty string when provided")
    return dict(event)


def validate_forecast_response(payload: Mapping[str, Any], *, atol: float = 1e-4) -> None:
    """Validate the forecast output contract (doubly stochastic N x N matrix + Pr(DNF))."""
    for required in ("model_version", "lap", "total_laps", "drivers", "matrix"):
        if required not in payload:
            raise ValueError(f"forecast response missing required field {required!r}")
    drivers = payload["drivers"]
    matrix = np.asarray(payload["matrix"], dtype=float)
    n = len(drivers)
    if n == 0 or matrix.shape != (n, n):
        raise ValueError(f"forecast matrix shape {matrix.shape} does not match {n} drivers")
    if not np.isfinite(matrix).all() or (matrix < -atol).any():
        raise ValueError("forecast matrix contains non-finite or negative probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=atol):
        raise ValueError("forecast matrix rows must sum to 1.0")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=atol):
        raise ValueError("forecast matrix columns must sum to 1.0")
    for driver in drivers:
        p_dnf = driver.get("p_dnf")
        if not _is_finite_real(p_dnf) or not (-atol <= float(p_dnf) <= 1.0 + atol):
            raise ValueError(f"driver {driver.get('driver')!r} has invalid p_dnf={p_dnf!r}")


# -- 7. Optional Lazy MLflow Tracking & Model Registry ------------------------


try:  # pragma: no cover - optional dependency; lazy stand-in keeps M1-M6 isolated
    from mlflow.pyfunc import PythonModel as _PyfuncBase
except ImportError:  # pragma: no cover
    class _PyfuncBase:  # type: ignore[no-redef]
        pass


class RaceModelPyfuncModel(_PyfuncBase):
    """Stateless MLflow pyfunc wrapper reconstructing ``LoadedModel`` from ``model.json``."""

    def load_context(self, context: Any) -> None:
        from f1_race_intelligence.race_model.artifact import ModelArtifact
        from f1_race_intelligence.race_model.runner import LoadedModel

        model_path = Path(context.artifacts["model_json"])
        artifact = ModelArtifact.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
        self._artifact = artifact
        self._model = LoadedModel.from_artifact(artifact)

    def predict(self, context: Any, model_input: Any) -> Any:
        """Return model metadata & version contract summary for requested session/lap rows."""
        import pandas as pd

        rows = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return pd.DataFrame(
            [
                {
                    "model_version": self._artifact.version,
                    "format": self._artifact.format,
                    "phi": float(self._model.phi),
                    "sigma_eps": float(self._model.sigma_eps),
                    "races_seen": len(self._artifact.races),
                }
                for _ in range(len(rows))
            ]
        )


def _sqlite_uri(path: Path) -> str:
    resolved = path.resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def log_race_model_to_mlflow(
    *,
    settings: Any,
    artifact: Any,
    evaluation_report: Mapping[str, Any],
    markdown_report: str,
    model_bytes: bytes,
    promoted: bool,
    verification: Optional[Mapping[str, Any]] = None,
    register_model: bool = True,
) -> Dict[str, Any]:
    """Optionally log a race-model evaluation run & versioned artifact to local MLflow.

    Uses the exact same local SQLite tracking/registry store as ``modeling/training.py``
    so LinearRegression, TabularMLP, and RaceModel appear side-by-side in ``mlflow ui``.
    """
    try:
        import mlflow
        import mlflow.pyfunc
        from mlflow.models import infer_signature
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            f"MLflow logging for race_model requires the optional `training` extra; install `{MLFLOW_INSTALL_HINT}` first"
        ) from exc

    training_cfg = settings.training
    race_cfg = settings.race_model
    db_path = Path(training_cfg.tracking_db_path)
    artifact_root = Path(training_cfg.artifact_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = _sqlite_uri(db_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(training_cfg.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            training_cfg.experiment_name,
            artifact_location=_file_uri(artifact_root),
        )
    else:
        experiment_id = experiment.experiment_id

    prov = evaluation_report.get("provenance") or getattr(artifact, "provenance", None) or {}
    split = str(evaluation_report.get("split", "validation"))
    reg_name = race_cfg.registered_model_name if register_model else None

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"m7-race-model-{artifact.version}",
    ) as active_run:
        run_id = active_run.info.run_id
        tags = {
            "project": "f1-race-intelligence",
            "stage": "m7-race-model",
            "model_type": MODEL_TYPE,
            "model_version": str(artifact.version),
            "evaluated_split": split,
            "promoted": str(bool(promoted)).lower(),
            "test_sealed": str(bool(prov.get("test_sealed", True))).lower(),
            "input_fingerprint": str(prov.get("input_fingerprint") or ""),
            "source_commit": str(prov.get("source_commit") or ""),
        }
        mlflow.set_tags(tags)

        season_params = (artifact.season or {}).get("state", {}).get("params") or {}
        params: Dict[str, Any] = {
            "model_type": MODEL_TYPE,
            "estimator": ESTIMATOR_NAME,
            "reference_baseline": race_cfg.promote_requires_beating,
            "random_seed": race_cfg.random_seed,
            "burn_in_races": race_cfg.burn_in_races,
            "scenarios": artifact.forecast.get("scenarios"),
            "noise_scale": artifact.forecast.get("noise_scale"),
            "noise_scale_per_phase": race_cfg.noise_scale_per_phase,
            "level1_phi": round(float(artifact.level1.get("phi", 0.0)), 6),
            "level1_sigma_eps": round(float(artifact.level1.get("sigma_eps", 0.0)), 6),
            "input_fingerprint": prov.get("input_fingerprint"),
            "config_fingerprint": prov.get("config_fingerprint"),
            "source_commit": prov.get("source_commit"),
        }
        for key in ("q_gamma", "q_gamma_season", "q_delta", "s_c", "s_w", "sigma_q", "lam"):
            if key in season_params:
                params[f"season_{key}"] = round(float(season_params[key]), 6)
        mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

        metrics_to_log: Dict[str, float] = {}
        by_bucket = evaluation_report.get("by_bucket") or {}
        for bucket, vals in by_bucket.items():
            tag = "overall" if bucket == "all" else bucket.replace("%", "pct")
            for src_key, dst_key in (
                ("model_rps", f"{split}.race_model.rps.{tag}"),
                ("persist_rps", f"{split}.persist.rps.{tag}"),
                ("transition_rps", f"{split}.transition.rps.{tag}"),
                ("model_pll", f"{split}.race_model.pairwise_log_loss.{tag}"),
                ("transition_pll", f"{split}.transition.pairwise_log_loss.{tag}"),
            ):
                if _is_finite_real(vals.get(src_key)):
                    metrics_to_log[dst_key] = float(vals[src_key])

        calib = evaluation_report.get("calibration") or {}
        for ev in ("win", "podium", "points", "dnf"):
            brier_val = (calib.get(ev) or {}).get("brier")
            if _is_finite_real(brier_val):
                metrics_to_log[f"{split}.race_model.brier.{ev}"] = float(brier_val)

        prior_check = (artifact.metrics or {}).get("prior_check") or {}
        for pc_split in ("train", "validation"):
            pc = prior_check.get(pc_split) or {}
            if _is_finite_real(pc.get("rmse")):
                metrics_to_log[f"{pc_split}.race_model.prior_rmse"] = float(pc["rmse"])
            if _is_finite_real(pc.get("z2")):
                metrics_to_log[f"{pc_split}.race_model.prior_z2"] = float(pc["z2"])

        if metrics_to_log:
            mlflow.log_metrics(metrics_to_log)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / MODEL_FILENAME
            model_path.write_bytes(model_bytes)

            def _write_json(name: str, obj: Any) -> Path:
                p = tmp / name
                p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
                return p

            mlflow.log_artifact(str(_write_json("data_provenance.json", prov)))
            mlflow.log_artifact(str(_write_json("training_config.json", artifact.settings)))
            mlflow.log_artifact(str(_write_json("metrics.json", metrics_to_log)))
            mlflow.log_artifact(str(_write_json(f"evaluate_{split}.json", evaluation_report)))
            if evaluation_report.get("interpretability"):
                mlflow.log_artifact(str(_write_json("interpretability.json", evaluation_report["interpretability"])))
            if verification:
                mlflow.log_artifact(str(_write_json("package_verification.json", verification)))
            md_path = tmp / f"evaluate_{split}.md"
            md_path.write_text(markdown_report, encoding="utf-8")
            mlflow.log_artifact(str(md_path))

            example_in = pd.DataFrame([{"session_key": 0, "lap": 1}])
            example_out = pd.DataFrame(
                [
                    {
                        "model_version": str(artifact.version),
                        "format": str(artifact.format),
                        "phi": float(artifact.level1.get("phi", 0.0)),
                        "sigma_eps": float(artifact.level1.get("sigma_eps", 0.0)),
                        "races_seen": len(artifact.races),
                    }
                ]
            )
            signature = infer_signature(example_in, example_out)
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RaceModelPyfuncModel(),
                artifacts={"model_json": str(model_path)},
                registered_model_name=reg_name,
                signature=signature,
                input_example=example_in,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"numpy=={np.__version__}",
                    f"pandas=={pd.__version__}",
                ],
            )

    registered_version: Optional[str] = None
    model_uri = f"runs:/{run_id}/model"
    if reg_name:
        versions = client.search_model_versions(f"name='{reg_name}' and run_id='{run_id}'")
        if versions:
            registered_version = str(versions[0].version)
            model_uri = f"models:/{reg_name}/{registered_version}"

    return {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": training_cfg.experiment_name,
        "registered_model_name": reg_name,
        "registered_version": registered_version,
        "model_uri": model_uri,
        "logged_model_uri": getattr(model_info, "model_uri", f"runs:/{run_id}/model"),
    }
