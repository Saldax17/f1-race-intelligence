"""Controlled experimental comparison of M7 PyTorch MLP configurations.

Runs a set of MLP architectures/regularisation settings on the exact same
verified M6 report (shared ``input_fingerprint``), evaluates each candidate
without polluting the MLflow Model Registry (``register_model=False``),
selects the winner strictly on the ``validation`` split (anti-snooping:
``test`` is never consulted during ranking or selection), optionally
registers only the winning configuration, and exports both JSON and academic
Markdown comparison reports.

PyTorch and MLflow are never imported at module level so the M1–M6 base
runtime can import this module without optional training or DL dependencies.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.features.selection import TARGET_COLUMN
from f1_race_intelligence.modeling.verification import verify_report

__all__ = [
    "DEFAULT_MLP_EXPERIMENTS",
    "MLPComparisonResult",
    "MLPComparisonRunner",
    "MLPExperimentSpec",
]

SELECTION_METRIC = "validation.mlp.mae"
TIE_BREAKER_METRIC = "validation.mlp.rmse"


@dataclass(frozen=True)
class MLPExperimentSpec:
    """One candidate MLP configuration evaluated in an experimental sweep."""

    name: str
    description: str
    mlp_overrides: Dict[str, Any]


@dataclass
class MLPComparisonResult:
    """Persisted outcome of comparing candidate MLP configurations on one M6 report."""

    comparison_id: str
    m6_run_id: str
    input_fingerprint: str
    selection_metric: str
    best_experiment: str
    candidates: List[Dict[str, Any]]
    registered_winner: Optional[Dict[str, Any]]
    json_report_path: str
    markdown_report_path: str

    def summary(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "m6_run_id": self.m6_run_id,
            "input_fingerprint": self.input_fingerprint,
            "selection_metric": self.selection_metric,
            "best_experiment": self.best_experiment,
            "candidates": [
                {
                    "rank": cand.get("rank"),
                    "name": cand["name"],
                    "run_id": cand["run_id"],
                    "validation_mae": _split_metric(cand["metrics"], "validation", "mlp", "mae"),
                    "validation_rmse": _split_metric(cand["metrics"], "validation", "mlp", "rmse"),
                    "test_mae": _split_metric(cand["metrics"], "test", "mlp", "mae"),
                    "test_rmse": _split_metric(cand["metrics"], "test", "mlp", "rmse"),
                    "generalization_status": (cand.get("generalization_diagnosis") or {}).get("status"),
                }
                for cand in self.candidates
            ],
            "registered_winner": (
                {
                    "experiment": self.registered_winner["experiment"],
                    "run_id": self.registered_winner["run_id"],
                    "model_name": self.registered_winner["model_name"],
                    "model_version": self.registered_winner["model_version"],
                    "model_uri": self.registered_winner["model_uri"],
                }
                if self.registered_winner is not None
                else None
            ),
            "json_report_path": self.json_report_path,
            "markdown_report_path": self.markdown_report_path,
        }


DEFAULT_MLP_EXPERIMENTS: Tuple[MLPExperimentSpec, ...] = (
    MLPExperimentSpec(
        name="mlp_small",
        description="Compact 1-layer MLP (lower capacity baseline to check underfitting)",
        mlp_overrides={"hidden_dims": [32], "dropout": 0.0, "weight_decay": 1e-4, "loss": "huber"},
    ),
    MLPExperimentSpec(
        name="mlp_default",
        description="Default 2-layer regularized MLP with Huber loss",
        mlp_overrides={"hidden_dims": [64, 32], "dropout": 0.1, "weight_decay": 1e-4, "loss": "huber"},
    ),
    MLPExperimentSpec(
        name="mlp_deeper_reg",
        description="Deeper 3-layer MLP with stronger dropout and L2 regularization",
        mlp_overrides={"hidden_dims": [128, 64, 32], "dropout": 0.2, "weight_decay": 5e-4, "loss": "huber"},
    ),
    MLPExperimentSpec(
        name="mlp_mse",
        description="Default 2-layer MLP with MSE loss (loss ablation vs Huber)",
        mlp_overrides={"hidden_dims": [64, 32], "dropout": 0.1, "weight_decay": 1e-4, "loss": "mse"},
    ),
)


class MLPComparisonRunner:
    """Compare multiple MLP configurations on one verified M6 report.

    Model selection uses ``validation.mlp.mae`` (tie-breaker
    ``validation.mlp.rmse``) exclusively. ``test`` metrics are reported only
    for post-hoc documentation and never influence ranking or selection.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        m6_report_path: Optional[str] = None,
        experiments: Optional[Sequence[MLPExperimentSpec]] = None,
        register_best: bool = False,
    ) -> None:
        self._settings = settings or load_settings()
        self._m6_report_path = m6_report_path or self._settings.training.m6_report_path
        specs = tuple(DEFAULT_MLP_EXPERIMENTS if experiments is None else experiments)
        if not specs:
            raise ValueError("experiments must contain at least one MLPExperimentSpec")
        seen_names = set()
        for spec in specs:
            if not isinstance(spec, MLPExperimentSpec):
                raise TypeError(f"each experiment must be an MLPExperimentSpec, got {type(spec).__name__}")
            if not isinstance(spec.name, str) or not spec.name.strip():
                raise ValueError("each MLPExperimentSpec must have a non-empty name")
            if spec.name in seen_names:
                raise ValueError(f"duplicate MLPExperimentSpec name: {spec.name!r}")
            seen_names.add(spec.name)
        self._experiments: Tuple[MLPExperimentSpec, ...] = specs
        self._register_best: bool = bool(register_best)

    def run(self) -> MLPComparisonResult:
        """Verify M6, train all candidate MLPs, rank on validation, and write reports."""
        if self._settings.storage.backend != "local":
            raise ValueError("M7 comparison supports local storage only; AWS and external stores are disabled")
        if not self._m6_report_path:
            raise ValueError("an M6 report path is required (use --m6-report)")
        if self._m6_report_path.startswith("s3://"):
            raise ValueError("M7 comparison accepts a local M6 report only")

        report_path = Path(self._m6_report_path).resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"M6 report not found: {report_path}")
        m6_report = json.loads(report_path.read_bytes().decode("utf-8"))
        integrity = verify_report(str(report_path), storage=self._settings.storage)
        if not integrity["ok"]:
            raise ValueError(f"M6 report or artifacts failed verification: {integrity['errors']}")
        if m6_report.get("verdict") not in {"VALID", "VALID_WITH_WARNINGS"}:
            raise ValueError(f"M6 report is not trainable (verdict={m6_report.get('verdict')!r})")
        if m6_report.get("aborted") or m6_report.get("blocking_failures"):
            raise ValueError("M6 report contains an abort or blocking leakage failure")
        if m6_report.get("target", {}).get("name") != TARGET_COLUMN:
            raise ValueError(f"M6 target must be {TARGET_COLUMN!r}")

        source = m6_report.get("source", {})
        input_fingerprint = source.get("input_fingerprint")
        m6_run_id = m6_report.get("run_id")
        if not isinstance(input_fingerprint, str) or not input_fingerprint.startswith("sha256:"):
            raise ValueError("M6 report has no valid input_fingerprint")
        if not isinstance(m6_run_id, str) or not m6_run_id:
            raise ValueError("M6 report has no run_id")

        from f1_race_intelligence.modeling.training import MLP_MODEL_TYPE, TrainingRunner

        comparison_id = f"cmp-{uuid.uuid4().hex[:8]}"
        raw_candidates: List[Dict[str, Any]] = []
        for spec in self._experiments:
            candidate_settings = _settings_with_mlp_overrides(self._settings, spec.mlp_overrides)
            result = TrainingRunner(
                candidate_settings,
                m6_report_path=str(report_path),
                model_type=MLP_MODEL_TYPE,
                register_model=False,
                extra_tags={
                    "comparison_id": comparison_id,
                    "experiment_label": spec.name,
                },
                run_name_prefix=f"m7-cmp-{spec.name}",
            ).run()
            saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
            raw_candidates.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "mlp_overrides": dict(spec.mlp_overrides),
                    "mlp_config": saved_report.get("config", {}).get("mlp", {}),
                    "run_id": result.run_id,
                    "model_uri": result.model_uri,
                    "model_version": result.model_version,
                    "report_path": result.report_path,
                    "metrics": result.metrics,
                    "mlp_training": saved_report.get("mlp_training", {}),
                    "generalization_diagnosis": saved_report.get("generalization_diagnosis"),
                    "interpretability": saved_report.get("interpretability"),
                }
            )

        # Anti-snooping: rank exclusively on the validation split (never test).
        ranked_indexed = sorted(enumerate(raw_candidates), key=_validation_ranking_key)
        ranked_candidates: List[Dict[str, Any]] = [
            {"rank": rank, **candidate}
            for rank, (_, candidate) in enumerate(ranked_indexed, start=1)
        ]
        winner_idx, _ = ranked_indexed[0]
        winner_spec = self._experiments[winner_idx]
        winner_candidate = ranked_candidates[0]
        best_experiment = winner_spec.name

        registered_winner: Optional[Dict[str, Any]] = None
        if self._register_best:
            winner_settings = _settings_with_mlp_overrides(self._settings, winner_spec.mlp_overrides)
            winner_result = TrainingRunner(
                winner_settings,
                m6_report_path=str(report_path),
                model_type=MLP_MODEL_TYPE,
                register_model=True,
                extra_tags={
                    "comparison_id": comparison_id,
                    "experiment_label": winner_spec.name,
                    "comparison_winner": "true",
                },
                run_name_prefix=f"m7-winner-{winner_spec.name}",
            ).run()
            registered_winner = {
                "experiment": winner_spec.name,
                "run_id": winner_result.run_id,
                "model_name": winner_result.model_name,
                "model_version": winner_result.model_version,
                "model_uri": winner_result.model_uri,
                "report_path": winner_result.report_path,
                "metrics": winner_result.metrics,
            }

        baselines = _extract_baselines(winner_candidate["metrics"])
        generated_at = datetime.now(timezone.utc).isoformat()
        m6_provenance = {
            "m6_report_path": str(report_path),
            "m6_run_id": m6_run_id,
            "input_fingerprint": input_fingerprint,
            "m6_verdict": m6_report.get("verdict"),
            "m6_warnings": list(m6_report.get("warnings") or []),
            "m6_source_commit": m6_report.get("source_commit"),
            "m6_pipeline_version": m6_report.get("pipeline_version"),
            "m6_splits": _extract_m6_split_summary(m6_report),
        }
        report_payload: Dict[str, Any] = {
            "schema_version": 1,
            "comparison_id": comparison_id,
            "generated_at": generated_at,
            "m6_provenance": m6_provenance,
            "selection_policy": {
                "selection_metric": SELECTION_METRIC,
                "tie_breaker": TIE_BREAKER_METRIC,
                "anti_snooping": (
                    "Candidates are ranked strictly on the validation split; "
                    "test metrics are reported post-hoc and never used for model selection."
                ),
            },
            "best_experiment": best_experiment,
            "baselines": baselines,
            "candidates": ranked_candidates,
            "registered_winner": registered_winner,
        }

        comparisons_dir = Path(self._settings.training.output_path).resolve() / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        json_report_path = comparisons_dir / f"mlp_comparison_{comparison_id}.json"
        markdown_report_path = comparisons_dir / f"mlp_comparison_{comparison_id}.md"

        json_report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        markdown_content = _render_markdown_report(
            report_payload=report_payload,
            winner_candidate=winner_candidate,
            min_sessions_per_split=int(self._settings.modeling.min_sessions_per_split),
        )
        markdown_report_path.write_text(markdown_content, encoding="utf-8")

        return MLPComparisonResult(
            comparison_id=comparison_id,
            m6_run_id=m6_run_id,
            input_fingerprint=input_fingerprint,
            selection_metric=SELECTION_METRIC,
            best_experiment=best_experiment,
            candidates=ranked_candidates,
            registered_winner=registered_winner,
            json_report_path=str(json_report_path),
            markdown_report_path=str(markdown_report_path),
        )


def _settings_with_mlp_overrides(base_settings: AppSettings, overrides: Mapping[str, Any]) -> AppSettings:
    """Return a validated copy of ``base_settings`` with ``training.mlp`` overrides applied."""
    payload = base_settings.model_dump()
    payload["training"]["model_type"] = "mlp"
    payload["training"]["mlp"].update(dict(overrides))
    return AppSettings.model_validate(payload)


def _split_metric(
    metrics: Mapping[str, Any], split: str, model_key: str, metric_name: str
) -> Optional[float]:
    split_map = metrics.get(split)
    if not isinstance(split_map, Mapping):
        return None
    model_map = split_map.get(model_key)
    if not isinstance(model_map, Mapping):
        return None
    value = model_map.get(metric_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validation_ranking_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[bool, float, bool, float, int]:
    """Strict validation-only ranking key: lowest validation MAE, then RMSE, then original order."""
    idx, candidate = item
    val_mae = _split_metric(candidate["metrics"], "validation", "mlp", "mae")
    val_rmse = _split_metric(candidate["metrics"], "validation", "mlp", "rmse")
    return (
        val_mae is None,
        math.inf if val_mae is None else val_mae,
        val_rmse is None,
        math.inf if val_rmse is None else val_rmse,
        idx,
    )


def _extract_baselines(metrics: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    baselines: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        split_map = metrics.get(split) or {}
        baselines[split] = {
            key: split_map[key]
            for key in ("reference_mean", "linear_regression")
            if key in split_map
        }
    return baselines


def _extract_m6_split_summary(m6_report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    splits = m6_report.get("split")
    if not isinstance(splits, Mapping):
        splits = m6_report.get("splits")
    if not isinstance(splits, Mapping):
        return {}
    summary: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        info = splits.get(split)
        if isinstance(info, Mapping):
            entry: Dict[str, Any] = {}
            for key in ("rows", "row_count", "n_rows", "sessions", "session_count", "n_sessions", "valid_target_rows"):
                if key in info:
                    entry[key] = info[key]
            summary[split] = entry
    return summary


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_lr(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2e}"


def _small_split_warnings(
    m6_provenance: Mapping[str, Any],
    baselines: Mapping[str, Any],
    min_sessions_per_split: int,
) -> List[str]:
    notices: List[str] = []
    if m6_provenance.get("m6_verdict") == "VALID_WITH_WARNINGS":
        notices.append("M6 report verdict is `VALID_WITH_WARNINGS`.")
    for warning in m6_provenance.get("m6_warnings") or []:
        notices.append(f"M6 warning: {warning}")
    splits_info = m6_provenance.get("m6_splits") or {}
    for split in ("train", "validation", "test"):
        info = splits_info.get(split) if isinstance(splits_info, Mapping) else None
        if isinstance(info, Mapping):
            session_count = info.get("session_count")
            if session_count is None and isinstance(info.get("sessions"), int):
                session_count = info.get("sessions")
            elif session_count is None and isinstance(info.get("sessions"), list):
                session_count = len(info["sessions"])
            if isinstance(session_count, int) and session_count < max(min_sessions_per_split, 3):
                notices.append(
                    f"Split `{split}` has only {session_count} session(s); treat metrics as a technical "
                    "check rather than a statistically robust benchmark."
                )
        ref = (baselines.get(split) or {}).get("reference_mean") or {}
        valid_rows = ref.get("valid_rows")
        if isinstance(valid_rows, int) and valid_rows < 30:
            notices.append(
                f"Split `{split}` has only {valid_rows} finite target row(s); variance across configurations may be high."
            )
    return notices


def _render_markdown_report(
    *,
    report_payload: Mapping[str, Any],
    winner_candidate: Mapping[str, Any],
    min_sessions_per_split: int,
) -> str:
    m6 = report_payload["m6_provenance"]
    baselines = report_payload["baselines"]
    candidates = report_payload["candidates"]
    best_name = report_payload["best_experiment"]
    registered_winner = report_payload.get("registered_winner")

    lines: List[str] = [
        f"# M7 MLP Experimental Comparison Report (`{report_payload['comparison_id']}`)",
        "",
        "## 1. Provenance & Experimental Protocol",
        "",
        f"- **Comparison ID:** `{report_payload['comparison_id']}`",
        f"- **Generated (UTC):** `{report_payload['generated_at']}`",
        f"- **M6 Run ID:** `{m6['m6_run_id']}`",
        f"- **M6 Input Fingerprint:** `{m6['input_fingerprint']}`",
        f"- **M6 Verdict:** `{m6['m6_verdict']}`",
        f"- **M6 Report Path:** `{m6['m6_report_path']}`",
        f"- **Selection Metric:** `{SELECTION_METRIC}` (tie-breaker: `{TIE_BREAKER_METRIC}`)",
        f"- **Selected Configuration (Validation Winner):** `{best_name}`",
        "",
        "> **Anti-Snooping Policy:** Candidate ranking and winner selection use the `validation` split exclusively. "
        "The `test` split is strictly held out and reported below only for post-hoc academic evaluation.",
        "",
    ]

    notices = _small_split_warnings(m6, baselines, min_sessions_per_split)
    if notices:
        lines.append("> **Data / Split Caution:**")
        for notice in notices:
            lines.append(f"> - {notice}")
        lines.append("")

    lines.extend(
        [
            "### Evaluated MLP Configurations",
            "",
            "| Rank | Config | Hidden Dims | Dropout | Weight Decay | Loss | LR Scheduler | Description |",
            "|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for cand in candidates:
        cfg = cand.get("mlp_config") or {}
        hidden_dims = cfg.get("hidden_dims", cand.get("mlp_overrides", {}).get("hidden_dims"))
        dropout = cfg.get("dropout", cand.get("mlp_overrides", {}).get("dropout"))
        weight_decay = cfg.get("weight_decay", cand.get("mlp_overrides", {}).get("weight_decay"))
        loss = cfg.get("loss", cand.get("mlp_overrides", {}).get("loss"))
        scheduler = cfg.get("lr_scheduler", "n/a")
        label = f"**{cand['name']}** (selected)" if cand["name"] == best_name else cand["name"]
        lines.append(
            f"| {cand['rank']} | {label} | `{hidden_dims}` | {_fmt_float(dropout, 2)} | "
            f"{_fmt_lr(weight_decay)} | `{loss}` | `{scheduler}` | {cand['description']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 2. Comparative Metrics (`DummyRegressor` vs `LinearRegression` vs MLP Configurations)",
            "",
            "| Model / Config | Train MAE | Train RMSE | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for baseline_key, baseline_label in (
        ("reference_mean", "Baseline: `DummyRegressor(mean)`"),
        ("linear_regression", "Baseline: `LinearRegression`"),
    ):
        lines.append(
            f"| {baseline_label} "
            f"| {_fmt_float(_split_metric(baselines, 'train', baseline_key, 'mae'))} "
            f"| {_fmt_float(_split_metric(baselines, 'train', baseline_key, 'rmse'))} "
            f"| {_fmt_float(_split_metric(baselines, 'validation', baseline_key, 'mae'))} "
            f"| {_fmt_float(_split_metric(baselines, 'validation', baseline_key, 'rmse'))} "
            f"| {_fmt_float(_split_metric(baselines, 'validation', baseline_key, 'r2'))} "
            f"| {_fmt_float(_split_metric(baselines, 'test', baseline_key, 'mae'))} "
            f"| {_fmt_float(_split_metric(baselines, 'test', baseline_key, 'rmse'))} "
            f"| {_fmt_float(_split_metric(baselines, 'test', baseline_key, 'r2'))} |"
        )

    for cand in candidates:
        metrics = cand["metrics"]
        label = f"MLP: **`{cand['name']}`** (winner)" if cand["name"] == best_name else f"MLP: `{cand['name']}`"
        lines.append(
            f"| {label} "
            f"| {_fmt_float(_split_metric(metrics, 'train', 'mlp', 'mae'))} "
            f"| {_fmt_float(_split_metric(metrics, 'train', 'mlp', 'rmse'))} "
            f"| {_fmt_float(_split_metric(metrics, 'validation', 'mlp', 'mae'))} "
            f"| {_fmt_float(_split_metric(metrics, 'validation', 'mlp', 'rmse'))} "
            f"| {_fmt_float(_split_metric(metrics, 'validation', 'mlp', 'r2'))} "
            f"| {_fmt_float(_split_metric(metrics, 'test', 'mlp', 'mae'))} "
            f"| {_fmt_float(_split_metric(metrics, 'test', 'mlp', 'rmse'))} "
            f"| {_fmt_float(_split_metric(metrics, 'test', 'mlp', 'r2'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 3. Overfitting / Underfitting Diagnosis (`train` vs `validation`)",
            "",
            "| Config | Diagnosis | Train MAE | Val MAE | Val/Train Ratio | Val vs Dummy Δ | Best / Trained / Max Epochs | Early Stopped | Final LR | Params |",
            "|---|---|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for cand in candidates:
        diag = cand.get("generalization_diagnosis") or {}
        training_info = cand.get("mlp_training") or {}
        best_ep = diag.get("best_epoch", training_info.get("best_epoch", "n/a"))
        trained_ep = diag.get("epochs_trained", training_info.get("epochs_trained", "n/a"))
        max_ep = diag.get("max_epochs", (cand.get("mlp_config") or {}).get("max_epochs", "n/a"))
        early_stopped = diag.get("early_stopped")
        early_str = "yes" if early_stopped is True else ("no" if early_stopped is False else "n/a")
        final_lr = training_info.get("final_learning_rate")
        total_params = training_info.get("total_params", "n/a")
        lines.append(
            f"| `{cand['name']}` | `{diag.get('status', 'n/a')}` "
            f"| {_fmt_float(diag.get('train_mae'))} "
            f"| {_fmt_float(diag.get('validation_mae'))} "
            f"| {_fmt_float(diag.get('val_to_train_mae_ratio'), 2)}x "
            f"| {_fmt_float(diag.get('val_vs_baseline_mae_delta'))} "
            f"| {best_ep} / {trained_ep} / {max_ep} "
            f"| {early_str} "
            f"| {_fmt_lr(final_lr)} "
            f"| {total_params} |"
        )
    lines.append("")
    for cand in candidates:
        diag = cand.get("generalization_diagnosis") or {}
        explanation = diag.get("explanation")
        if explanation:
            lines.append(f"- **`{cand['name']}`:** {explanation}")
    lines.append("")

    lines.extend(
        [
            f"## 4. Interpretability — Permutation Feature Importance of Winner (`{best_name}`, `validation` split)",
            "",
        ]
    )
    interp = winner_candidate.get("interpretability") or {}
    importances = interp.get("importances")
    if isinstance(importances, list) and importances:
        lines.extend(
            [
                f"- **Split:** `{interp.get('split', 'validation')}` "
                f"(`n_rows={interp.get('n_rows')}`, `n_repeats={interp.get('n_repeats')}`, "
                f"`random_seed={interp.get('random_seed')}`)",
                f"- **Unshuffled Baseline:** MAE = `{_fmt_float(interp.get('baseline_mae'))}` s, "
                f"RMSE = `{_fmt_float(interp.get('baseline_rmse'))}` s",
                "",
                "| Rank | Encoded Feature | MAE Increase (mean ± std, s) | RMSE Increase (mean ± std, s) |",
                "|---:|---|---:|---:|",
            ]
        )
        for idx, entry in enumerate(importances, start=1):
            lines.append(
                f"| {idx} | `{entry['feature']}` "
                f"| {_fmt_float(entry.get('mae_increase_mean'))} ± {_fmt_float(entry.get('mae_increase_std'))} "
                f"| {_fmt_float(entry.get('rmse_increase_mean'))} ± {_fmt_float(entry.get('rmse_increase_std'))} |"
            )
        lines.append("")
        collinearity_note = interp.get("collinearity_note")
        if collinearity_note:
            lines.append(f"> **Collinearity Note:** {collinearity_note}")
            lines.append("")
    else:
        status = interp.get("status", "unavailable")
        reason = interp.get("reason")
        detail = f" ({reason})" if reason else ""
        lines.append(f"Permutation feature importance status: `{status}`{detail}.")
        lines.append("")

    lines.extend(["## 5. Model Registry Status", ""])
    if registered_winner is not None:
        lines.extend(
            [
                "Only the validation-selected winner was registered in the MLflow Model Registry "
                "(discarded candidates were logged as unregistered runs only):",
                "",
                f"- **Winning Experiment:** `{registered_winner['experiment']}`",
                f"- **Registered Model Name:** `{registered_winner['model_name']}`",
                f"- **Registered Version:** `{registered_winner['model_version']}`",
                f"- **Model URI:** `{registered_winner['model_uri']}`",
                f"- **Winner MLflow Run ID:** `{registered_winner['run_id']}`",
                f"- **Winner Training Report:** `{registered_winner['report_path']}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No model was registered in the MLflow Model Registry (`register_best=False`). "
                "Re-run with `--register-best` to register the validation winner for publication and M8 serving.",
                "",
            ]
        )

    return "\n".join(lines).rstrip("\n") + "\n"
