"""M7 training and evaluation over verified M6 artifacts.

M7 never rebuilds features or splits. Its model inputs are the ordered
``output_feature_names`` written by M6, and every run starts by verifying the
M6 report and its content hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.features.selection import TARGET_COLUMN
from f1_race_intelligence.modeling.traceability import source_revision
from f1_race_intelligence.modeling.verification import verify_report

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)
FEATURE_ARTIFACT = "features.parquet"
PREPROCESSOR_ARTIFACT = "preprocessor.joblib"
SCHEMA_ARTIFACT = "feature_schema.json"
LINEAR_MODEL_TYPE = "linear_regression"
MLP_MODEL_TYPE = "mlp"
SUPPORTED_MODEL_TYPES = (LINEAR_MODEL_TYPE, MLP_MODEL_TYPE)


@dataclass
class TrainingResult:
    """The persisted identity, provenance, and scores of one M7 run."""

    run_id: str
    m6_run_id: str
    input_fingerprint: str
    model_name: str
    model_version: Optional[str]
    model_uri: str
    metrics: Dict[str, Dict[str, Dict[str, Any]]]
    report_path: str
    source_commit: Optional[str]
    model_type: str = LINEAR_MODEL_TYPE

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "m6_run_id": self.m6_run_id,
            "input_fingerprint": self.input_fingerprint,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_uri": self.model_uri,
            "metrics": self.metrics,
            "report_path": self.report_path,
            "source_commit": self.source_commit,
        }


def regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Calculate regression scores while reporting excluded and zero targets.

    MAPE is calculated only for finite, non-zero targets. It is returned as a
    percentage; when no such target exists its value is ``None``.
    """
    actual = np.asarray(y_true, dtype="float64").reshape(-1)
    predicted = np.asarray(y_pred, dtype="float64").reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same number of rows")

    finite = np.isfinite(actual) & np.isfinite(predicted)
    valid_actual = actual[finite]
    valid_predicted = predicted[finite]
    nonzero = valid_actual != 0

    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    if len(valid_actual):
        errors = valid_predicted - valid_actual
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if len(valid_actual) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                score = 1.0 - (
                    np.sum(np.square(errors))
                    / np.sum(np.square(valid_actual - np.mean(valid_actual)))
                )
            if math.isfinite(float(score)):
                r2 = float(score)

    mape: Optional[float] = None
    if np.any(nonzero):
        mape = float(
            np.mean(
                np.abs((valid_predicted[nonzero] - valid_actual[nonzero]) / valid_actual[nonzero])
            )
            * 100.0
        )

    return {
        "rows": int(len(actual)),
        "valid_rows": int(np.count_nonzero(finite)),
        "excluded_rows": int(len(actual) - np.count_nonzero(finite)),
        "zero_target_rows": int(np.count_nonzero(finite & (actual == 0))),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


class TrainingRunner:
    """Fit an M7 model and reference baselines on the M6 train matrix and score held-out splits."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        m6_report_path: Optional[str] = None,
        model_type: Optional[str] = None,
        register_model: bool = True,
        extra_tags: Optional[Dict[str, str]] = None,
        run_name_prefix: Optional[str] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._config = self._settings.training
        self._m6_report_path = m6_report_path or self._config.m6_report_path
        resolved_model_type = self._config.model_type if model_type is None else model_type
        if resolved_model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported M7 model_type {resolved_model_type!r}; "
                f"expected one of {', '.join(SUPPORTED_MODEL_TYPES)}"
            )
        self._model_type: str = resolved_model_type
        self._register_model: bool = bool(register_model)
        self._extra_tags: Dict[str, str] = (
            {str(key): str(value) for key, value in extra_tags.items()} if extra_tags else {}
        )
        self._run_name_prefix: Optional[str] = (
            run_name_prefix.strip()
            if isinstance(run_name_prefix, str) and run_name_prefix.strip()
            else None
        )

    def run(self) -> TrainingResult:
        """Verify M6, train the configured model with baselines, and register it in MLflow."""
        if self._settings.storage.backend != "local":
            raise ValueError("M7 supports local storage only; AWS and external stores are disabled")
        if not self._m6_report_path:
            raise ValueError("an M6 report path is required (use --m6-report)")
        if self._m6_report_path.startswith("s3://"):
            raise ValueError("M7 accepts a local M6 report only")

        report_path = Path(self._m6_report_path).resolve()
        report_bytes = report_path.read_bytes()
        m6_report = json.loads(report_bytes.decode("utf-8"))
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

        feature_names = m6_report.get("preprocessing", {}).get("output_feature_names")
        if (
            not isinstance(feature_names, list)
            or not feature_names
            or not all(isinstance(name, str) and name for name in feature_names)
            or len(feature_names) != len(set(feature_names))
            or TARGET_COLUMN in feature_names
        ):
            raise ValueError("M6 report has an invalid encoded feature schema")

        artifacts = self._artifact_map(m6_report)
        matrices = {
            split: self._load_matrix(artifacts[(split, FEATURE_ARTIFACT)], feature_names)
            for split in SPLITS
        }
        train_x, train_y = matrices[TRAIN]
        train_mask = np.isfinite(train_y)
        if not np.any(train_mask):
            raise ValueError("M6 train split contains no finite targets")
        train_x = train_x.loc[train_mask].reset_index(drop=True)
        train_y = train_y[train_mask]

        seed = int(self._settings.modeling.random_seed)
        linear = LinearRegression()
        reference = DummyRegressor(strategy="mean")
        linear.fit(train_x, train_y)
        reference.fit(train_x, train_y)

        fitted_mlp = None
        log_mlp_pyfunc_model = None
        diagnose_generalization_fn = None
        pfi_result = None
        interpretability_report: Optional[Dict[str, Any]] = None
        final_learning_rate: Optional[float] = None
        if self._model_type == MLP_MODEL_TYPE:
            from f1_race_intelligence.modeling.dl import (
                MLPConfig,
                compute_permutation_importance,
                diagnose_generalization as _diagnose_generalization,
                log_mlp_pyfunc_model as _log_mlp_pyfunc_model,
                train_tabular_mlp,
            )

            mlp_settings = self._config.mlp
            mlp_config = MLPConfig(
                hidden_dims=tuple(mlp_settings.hidden_dims),
                dropout=mlp_settings.dropout,
                learning_rate=mlp_settings.learning_rate,
                weight_decay=mlp_settings.weight_decay,
                batch_size=mlp_settings.batch_size,
                max_epochs=mlp_settings.max_epochs,
                early_stopping_patience=mlp_settings.early_stopping_patience,
                min_delta=mlp_settings.min_delta,
                loss=mlp_settings.loss,
                scale_target=mlp_settings.scale_target,
                random_seed=seed,
                lr_scheduler=mlp_settings.lr_scheduler,
                lr_scheduler_factor=mlp_settings.lr_scheduler_factor,
                lr_scheduler_patience=mlp_settings.lr_scheduler_patience,
                lr_scheduler_min_lr=mlp_settings.lr_scheduler_min_lr,
            )
            val_x, val_y = matrices[VALIDATION]
            fitted_mlp = train_tabular_mlp(
                train_x,
                train_y,
                val_x,
                val_y,
                feature_names=feature_names,
                config=mlp_config,
            )
            log_mlp_pyfunc_model = _log_mlp_pyfunc_model
            diagnose_generalization_fn = _diagnose_generalization
            recorded_lrs = [entry.learning_rate for entry in fitted_mlp.history if entry.learning_rate is not None]
            final_learning_rate = float(recorded_lrs[-1]) if recorded_lrs else float(fitted_mlp.config.learning_rate)

            if mlp_settings.compute_permutation_importance:
                if np.any(np.isfinite(val_y)):
                    pfi_result = compute_permutation_importance(
                        fitted_mlp,
                        val_x,
                        val_y,
                        split=VALIDATION,
                        n_repeats=mlp_settings.permutation_importance_repeats,
                        random_seed=seed,
                    )
                    interpretability_report = {
                        "method": "permutation_feature_importance",
                        **pfi_result.to_dict(top_k=mlp_settings.permutation_importance_top_k),
                    }
                else:
                    interpretability_report = {
                        "method": "permutation_feature_importance",
                        "status": "skipped",
                        "reason": "validation split has no finite targets",
                    }
            else:
                interpretability_report = {
                    "method": "permutation_feature_importance",
                    "status": "disabled",
                }

        metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        predictions: Dict[str, Dict[str, np.ndarray]] = {}
        for split, (features, targets) in matrices.items():
            split_predictions: Dict[str, np.ndarray] = {}
            if fitted_mlp is not None:
                split_predictions["mlp"] = fitted_mlp.predict(features)
            split_predictions["linear_regression"] = linear.predict(features)
            split_predictions["reference_mean"] = reference.predict(features)
            predictions[split] = split_predictions
            metrics[split] = {
                model_name: regression_metrics(targets, predicted)
                for model_name, predicted in split_predictions.items()
            }

        generalization_diagnosis: Optional[Dict[str, Any]] = None
        if fitted_mlp is not None and diagnose_generalization_fn is not None:
            generalization_diagnosis = diagnose_generalization_fn(
                metrics,
                model_key="mlp",
                baseline_key="reference_mean",
                epochs_trained=fitted_mlp.epochs_trained,
                best_epoch=fitted_mlp.best_epoch,
                max_epochs=fitted_mlp.config.max_epochs,
            )

        mlflow, client, tracking_uri = self._configure_tracking()
        source_commit, source_dirty = source_revision()
        registered_name = self._config.effective_registered_model_name(self._model_type)
        estimator_name = (
            "f1_race_intelligence.modeling.dl.TabularMLP"
            if self._model_type == MLP_MODEL_TYPE
            else "sklearn.linear_model.LinearRegression"
        )
        config_snapshot: Dict[str, Any] = {
            "m6_report_path": str(report_path),
            "tracking_db_path": str(Path(self._config.tracking_db_path).resolve()),
            "artifact_path": str(Path(self._config.artifact_path).resolve()),
            "output_path": str(Path(self._config.output_path).resolve()),
            "experiment_name": self._config.experiment_name,
            "model_type": self._model_type,
            "registered_model_name": registered_name,
            "random_seed": seed,
            "estimator": estimator_name,
            "reference_baseline": "sklearn.dummy.DummyRegressor(strategy='mean')",
            "mape_zero_policy": "exclude zero targets and report zero_target_rows",
        }
        if fitted_mlp is not None:
            mlp_settings = self._config.mlp
            config_snapshot["mlp"] = {
                "hidden_dims": list(fitted_mlp.config.hidden_dims),
                "dropout": fitted_mlp.config.dropout,
                "learning_rate": fitted_mlp.config.learning_rate,
                "weight_decay": fitted_mlp.config.weight_decay,
                "batch_size": fitted_mlp.config.batch_size,
                "max_epochs": fitted_mlp.config.max_epochs,
                "early_stopping_patience": fitted_mlp.config.early_stopping_patience,
                "min_delta": fitted_mlp.config.min_delta,
                "loss": fitted_mlp.config.loss,
                "scale_target": fitted_mlp.config.scale_target,
                "random_seed": fitted_mlp.config.random_seed,
                "lr_scheduler": fitted_mlp.config.lr_scheduler,
                "lr_scheduler_factor": fitted_mlp.config.lr_scheduler_factor,
                "lr_scheduler_patience": fitted_mlp.config.lr_scheduler_patience,
                "lr_scheduler_min_lr": fitted_mlp.config.lr_scheduler_min_lr,
                "compute_permutation_importance": mlp_settings.compute_permutation_importance,
                "permutation_importance_repeats": mlp_settings.permutation_importance_repeats,
                "permutation_importance_top_k": mlp_settings.permutation_importance_top_k,
            }
        provenance = {
            "m6_run_id": m6_run_id,
            "m6_pipeline_version": m6_report.get("pipeline_version"),
            "m6_source_commit": m6_report.get("source_commit"),
            "m7_source_commit": source_commit,
            "m7_source_dirty": source_dirty,
            "input_fingerprint": input_fingerprint,
            "input_fingerprint_scheme": source.get("input_fingerprint_scheme"),
            "data_version": f"m6:{m6_run_id}:{input_fingerprint}",
            "m6_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "verified_artifact_hashes": [
                {
                    "artifact": item.get("artifact"),
                    "split": item.get("split"),
                    "sha256": item.get("sha256"),
                }
                for item in m6_report.get("artifact_files", [])
            ],
        }

        mlflow.set_experiment(self._config.experiment_name)
        default_prefix = "m7-mlp" if self._model_type == MLP_MODEL_TYPE else "m7-linear-regression"
        run_prefix = self._run_name_prefix or default_prefix
        run_name = f"{run_prefix}-{uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            run_params: Dict[str, Any] = {
                "model_type": self._model_type,
                "estimator": config_snapshot["estimator"],
                "reference_baseline": config_snapshot["reference_baseline"],
                "random_seed": seed,
                "m6_run_id": m6_run_id,
                "m6_pipeline_version": str(m6_report.get("pipeline_version", "")),
                "input_fingerprint": input_fingerprint,
                "source_commit": source_commit or "unavailable",
            }
            if fitted_mlp is not None:
                run_params.update(
                    {
                        "mlp_hidden_dims": json.dumps(list(fitted_mlp.config.hidden_dims)),
                        "mlp_dropout": fitted_mlp.config.dropout,
                        "mlp_learning_rate": fitted_mlp.config.learning_rate,
                        "mlp_weight_decay": fitted_mlp.config.weight_decay,
                        "mlp_batch_size": fitted_mlp.config.batch_size,
                        "mlp_max_epochs": fitted_mlp.config.max_epochs,
                        "mlp_early_stopping_patience": fitted_mlp.config.early_stopping_patience,
                        "mlp_min_delta": fitted_mlp.config.min_delta,
                        "mlp_loss": fitted_mlp.config.loss,
                        "mlp_scale_target": fitted_mlp.config.scale_target,
                        "mlp_lr_scheduler": fitted_mlp.config.lr_scheduler,
                        "mlp_lr_scheduler_factor": fitted_mlp.config.lr_scheduler_factor,
                        "mlp_lr_scheduler_patience": fitted_mlp.config.lr_scheduler_patience,
                        "mlp_lr_scheduler_min_lr": fitted_mlp.config.lr_scheduler_min_lr,
                        "mlp_final_learning_rate": final_learning_rate,
                        "mlp_total_params": fitted_mlp.total_params,
                        "mlp_best_epoch": fitted_mlp.best_epoch,
                        "mlp_epochs_trained": fitted_mlp.epochs_trained,
                    }
                )
            mlflow.log_params(run_params)
            core_tags = {
                "project": "f1-race-intelligence",
                "stage": "m7-training",
                "model_type": self._model_type,
                "data_version": provenance["data_version"],
                "input_fingerprint": input_fingerprint,
                "m6_run_id": m6_run_id,
                "m6_source_commit": str(m6_report.get("source_commit") or "unavailable"),
                "m7_source_commit": str(source_commit or "unavailable"),
            }
            mlflow.set_tags({**self._extra_tags, **core_tags})
            if fitted_mlp is not None:
                for entry in fitted_mlp.history:
                    mlflow.log_metric("train.mlp.loss", float(entry.train_loss), step=entry.epoch)
                    if entry.val_loss is not None:
                        mlflow.log_metric("validation.mlp.loss", float(entry.val_loss), step=entry.epoch)
                    if entry.learning_rate is not None:
                        mlflow.log_metric("train.mlp.lr", float(entry.learning_rate), step=entry.epoch)
            for split in SPLITS:
                for model_name, values in metrics[split].items():
                    mlflow.log_metrics(
                        {
                            f"{split}.{model_name}.{key}": float(value)
                            for key, value in values.items()
                            if value is not None
                        }
                    )

            mlflow.log_dict(config_snapshot, "training_config.json")
            mlflow.log_dict(provenance, "data_provenance.json")
            mlflow.log_dict(metrics, "metrics.json")
            if fitted_mlp is not None:
                history_payload = [
                    {
                        "epoch": entry.epoch,
                        "train_loss": entry.train_loss,
                        "val_loss": entry.val_loss,
                        "learning_rate": entry.learning_rate,
                    }
                    for entry in fitted_mlp.history
                ]
                mlflow.log_dict(history_payload, "training_history.json")
                if pfi_result is not None:
                    mlflow.log_dict(pfi_result.to_dict(), "permutation_importance.json")
            mlflow.log_artifact(str(report_path), artifact_path="m6")
            mlflow.log_artifact(str(artifacts[(None, PREPROCESSOR_ARTIFACT)]), artifact_path="m6")
            schema_path = artifacts.get((None, SCHEMA_ARTIFACT))
            if schema_path is not None:
                mlflow.log_artifact(str(schema_path), artifact_path="m6")

            example = train_x.head(min(5, len(train_x)))
            target_registered_name = registered_name if self._register_model else None
            if self._model_type == LINEAR_MODEL_TYPE:
                from mlflow.models import infer_signature
                import mlflow.sklearn

                signature = infer_signature(example, linear.predict(example))
                mlflow.sklearn.log_model(
                    linear,
                    artifact_path="model",
                    registered_model_name=target_registered_name,
                    signature=signature,
                    input_example=example,
                )
                primary_predict = linear.predict
                recovery_tol = 1e-10
            else:
                assert fitted_mlp is not None and log_mlp_pyfunc_model is not None
                log_mlp_pyfunc_model(
                    fitted_mlp,
                    example=example,
                    registered_model_name=target_registered_name,
                    artifact_path="model",
                )
                primary_predict = fitted_mlp.predict
                recovery_tol = 1e-5

            if self._register_model:
                version: Optional[str] = self._registered_version(client, registered_name, run_id)
                model_uri = f"models:/{registered_name}/{version}"
            else:
                version = None
                model_uri = f"runs:/{run_id}/model"
            import mlflow.pyfunc

            loaded_model = mlflow.pyfunc.load_model(model_uri)
            check_features = matrices[TEST][0].head(1)
            expected = np.asarray(primary_predict(check_features), dtype="float64").reshape(-1)
            recovered = np.asarray(loaded_model.predict(check_features), dtype="float64").reshape(-1)
            if expected.shape != recovered.shape or not np.allclose(
                expected, recovered, rtol=recovery_tol, atol=recovery_tol
            ):
                raise RuntimeError("registered model inference did not reproduce the fitted model prediction")

            finished_at = datetime.now(timezone.utc).isoformat()
            inference_input_desc = "M6 encoded feature matrix columns from preprocessing.output_feature_names"
            if self._register_model:
                registered_model_report: Dict[str, Any] = {
                    "name": registered_name,
                    "version": version,
                    "uri": model_uri,
                    "inference_input": inference_input_desc,
                    "recovery_check": "passed",
                }
            else:
                registered_model_report = {
                    "name": None,
                    "version": None,
                    "uri": model_uri,
                    "registered": False,
                    "inference_input": inference_input_desc,
                    "recovery_check": "passed",
                }
            report: Dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": active_run.info.start_time,
                "finished_at": finished_at,
                "tracking_uri": tracking_uri,
                "experiment_name": self._config.experiment_name,
                "model_type": self._model_type,
                "m6_run_id": m6_run_id,
                "data_version": provenance["data_version"],
                "input_fingerprint": input_fingerprint,
                "source_commit": source_commit,
                "source_dirty": source_dirty,
                "config": config_snapshot,
                "features": feature_names,
                "target": TARGET_COLUMN,
                "metrics": metrics,
                "registered_model": registered_model_report,
                "m6_integrity": integrity,
            }
            if fitted_mlp is not None:
                report["mlp_training"] = {
                    "best_epoch": fitted_mlp.best_epoch,
                    "epochs_trained": fitted_mlp.epochs_trained,
                    "best_val_loss": fitted_mlp.best_val_loss,
                    "total_params": fitted_mlp.total_params,
                    "y_mean": fitted_mlp.y_mean,
                    "y_std": fitted_mlp.y_std,
                    "final_learning_rate": final_learning_rate,
                }
                report["generalization_diagnosis"] = generalization_diagnosis
                report["interpretability"] = interpretability_report
            mlflow.log_dict(report, "training_report.json")

        output_dir = Path(self._config.output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_report = output_dir / f"training_report_{run_id}.json"
        saved_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return TrainingResult(
            run_id=run_id,
            m6_run_id=m6_run_id,
            input_fingerprint=input_fingerprint,
            model_name=registered_name,
            model_version=version,
            model_uri=model_uri,
            metrics=metrics,
            report_path=str(saved_report),
            source_commit=source_commit,
            model_type=self._model_type,
        )

    def _configure_tracking(self) -> Tuple[Any, Any, str]:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "M7 requires the optional training dependencies; install "
                '`python -m pip install -e ".[training]"` first'
            ) from exc

        db_path = Path(self._config.tracking_db_path).resolve()
        artifact_root = Path(self._config.artifact_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path.as_posix()}"
        artifact_uri = artifact_root.as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)

        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = client.get_experiment_by_name(self._config.experiment_name)
        if experiment is None:
            client.create_experiment(self._config.experiment_name, artifact_location=artifact_uri)
        elif experiment.artifact_location != artifact_uri:
            raise ValueError(
                "the existing MLflow experiment uses a different artifact location; "
                "use a new tracking database or restore its configured artifact_path"
            )
        return mlflow, client, tracking_uri

    @staticmethod
    def _artifact_map(report: Dict[str, Any]) -> Dict[Tuple[Optional[str], str], Path]:
        result: Dict[Tuple[Optional[str], str], Path] = {}
        for item in report.get("artifact_files", []):
            if not isinstance(item, dict):
                continue
            name, split, location = item.get("artifact"), item.get("split"), item.get("path")
            if not isinstance(name, str) or not isinstance(location, str):
                continue
            if location.startswith("s3://"):
                raise ValueError("M7 accepts local M6 artifacts only")
            path = Path(location)
            if not path.is_file():
                raise ValueError(f"M6 artifact is missing: {location}")
            result[(split, name)] = path
        for split in SPLITS:
            if (split, FEATURE_ARTIFACT) not in result:
                raise ValueError(f"M6 report does not declare {split}/{FEATURE_ARTIFACT}")
        if (None, PREPROCESSOR_ARTIFACT) not in result:
            raise ValueError("M6 report does not declare its fitted preprocessor")
        return result

    @staticmethod
    def _load_matrix(path: Path, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        frame = pd.read_parquet(path, engine="pyarrow")
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"M6 matrix {path} has no {TARGET_COLUMN!r} target column")
        missing_features = [name for name in feature_names if name not in frame.columns]
        if missing_features:
            raise ValueError(f"M6 matrix {path} is missing declared features: {missing_features}")
        features = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        values = features.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"M6 matrix {path} has missing or non-finite model features")
        targets = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype="float64")
        return features, targets

    @staticmethod
    def _registered_version(client: Any, model_name: str, run_id: str) -> str:
        versions = [
            item for item in client.search_model_versions(f"name='{model_name}'")
            if item.run_id == run_id
        ]
        if not versions:
            raise RuntimeError(f"MLflow did not register a version of {model_name!r} for run {run_id}")
        version = max(versions, key=lambda item: int(item.version))
        if version.status != "READY":
            raise RuntimeError(
                f"registered model {model_name!r} version {version.version} is {version.status}, not READY"
            )
        return str(version.version)
