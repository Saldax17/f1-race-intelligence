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


@dataclass
class TrainingResult:
    """The persisted identity, provenance, and scores of one M7 run."""

    run_id: str
    m6_run_id: str
    input_fingerprint: str
    model_name: str
    model_version: str
    model_uri: str
    metrics: Dict[str, Dict[str, Dict[str, Any]]]
    report_path: str
    source_commit: Optional[str]

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "m6_run_id": self.m6_run_id,
            "input_fingerprint": self.input_fingerprint,
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
    """Fit the M7 baselines on the M6 train matrix and score held-out splits."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        m6_report_path: Optional[str] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._config = self._settings.training
        self._m6_report_path = m6_report_path or self._config.m6_report_path

    def run(self) -> TrainingResult:
        """Verify M6, train the two baselines, and register LinearRegression."""
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

        metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        predictions: Dict[str, Dict[str, np.ndarray]] = {}
        for split, (features, targets) in matrices.items():
            predictions[split] = {
                "linear_regression": linear.predict(features),
                "reference_mean": reference.predict(features),
            }
            metrics[split] = {
                model_name: regression_metrics(targets, predicted)
                for model_name, predicted in predictions[split].items()
            }

        mlflow, client, tracking_uri = self._configure_tracking()
        source_commit, source_dirty = source_revision()
        config_snapshot = {
            "m6_report_path": str(report_path),
            "tracking_db_path": str(Path(self._config.tracking_db_path).resolve()),
            "artifact_path": str(Path(self._config.artifact_path).resolve()),
            "output_path": str(Path(self._config.output_path).resolve()),
            "experiment_name": self._config.experiment_name,
            "registered_model_name": self._config.registered_model_name,
            "random_seed": seed,
            "estimator": "sklearn.linear_model.LinearRegression",
            "reference_baseline": "sklearn.dummy.DummyRegressor(strategy='mean')",
            "mape_zero_policy": "exclude zero targets and report zero_target_rows",
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
        run_name = f"m7-linear-regression-{uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params(
                {
                    "estimator": config_snapshot["estimator"],
                    "reference_baseline": config_snapshot["reference_baseline"],
                    "random_seed": seed,
                    "m6_run_id": m6_run_id,
                    "m6_pipeline_version": str(m6_report.get("pipeline_version", "")),
                    "input_fingerprint": input_fingerprint,
                    "source_commit": source_commit or "unavailable",
                }
            )
            mlflow.set_tags(
                {
                    "project": "f1-race-intelligence",
                    "stage": "m7-training",
                    "data_version": provenance["data_version"],
                    "input_fingerprint": input_fingerprint,
                    "m6_run_id": m6_run_id,
                    "m6_source_commit": str(m6_report.get("source_commit") or "unavailable"),
                    "m7_source_commit": str(source_commit or "unavailable"),
                }
            )
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
            mlflow.log_artifact(str(report_path), artifact_path="m6")
            mlflow.log_artifact(str(artifacts[(None, PREPROCESSOR_ARTIFACT)]), artifact_path="m6")
            schema_path = artifacts.get((None, SCHEMA_ARTIFACT))
            if schema_path is not None:
                mlflow.log_artifact(str(schema_path), artifact_path="m6")

            from mlflow.models import infer_signature
            import mlflow.sklearn

            example = train_x.head(min(5, len(train_x)))
            signature = infer_signature(example, linear.predict(example))
            mlflow.sklearn.log_model(
                linear,
                artifact_path="model",
                registered_model_name=self._config.registered_model_name,
                signature=signature,
                input_example=example,
            )

            version = self._registered_version(client, self._config.registered_model_name, run_id)
            model_uri = f"models:/{self._config.registered_model_name}/{version}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            check_features = matrices[TEST][0].head(1)
            expected = linear.predict(check_features)
            recovered = loaded_model.predict(check_features)
            if not np.allclose(expected, recovered, rtol=1e-10, atol=1e-10):
                raise RuntimeError("registered model inference did not reproduce the fitted model prediction")

            finished_at = datetime.now(timezone.utc).isoformat()
            report = {
                "schema_version": 1,
                "run_id": run_id,
                "started_at": active_run.info.start_time,
                "finished_at": finished_at,
                "tracking_uri": tracking_uri,
                "experiment_name": self._config.experiment_name,
                "m6_run_id": m6_run_id,
                "data_version": provenance["data_version"],
                "input_fingerprint": input_fingerprint,
                "source_commit": source_commit,
                "source_dirty": source_dirty,
                "config": config_snapshot,
                "features": feature_names,
                "target": TARGET_COLUMN,
                "metrics": metrics,
                "registered_model": {
                    "name": self._config.registered_model_name,
                    "version": version,
                    "uri": model_uri,
                    "inference_input": "M6 encoded feature matrix columns from preprocessing.output_feature_names",
                    "recovery_check": "passed",
                },
                "m6_integrity": integrity,
            }
            mlflow.log_dict(report, "training_report.json")

        output_dir = Path(self._config.output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_report = output_dir / f"training_report_{run_id}.json"
        saved_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return TrainingResult(
            run_id=run_id,
            m6_run_id=m6_run_id,
            input_fingerprint=input_fingerprint,
            model_name=self._config.registered_model_name,
            model_version=version,
            model_uri=model_uri,
            metrics=metrics,
            report_path=str(saved_report),
            source_commit=source_commit,
        )

    def _configure_tracking(self) -> Tuple[Any, Any, str]:
        import mlflow
        from mlflow.tracking import MlflowClient

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
