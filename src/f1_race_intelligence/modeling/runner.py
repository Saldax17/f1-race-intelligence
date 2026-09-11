"""Orchestration of the modelling-dataset stage.

Run it with::

    python -m f1_race_intelligence.modeling.runner

The order is deliberate and is the whole point of the stage:

    load M5 -> drop rows with no target -> split races in time ->
    fit preprocessing on train alone -> transform every split -> check -> write

Fitting happens after the split and only on the training races. Doing it
before, on everything, is the most common way a lap-time model ends up
looking better than it is.

Nothing is written when a blocking guard fails: a modelling dataset that
is known to leak is worse than none, because the score it produces is
believable.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.features.selection import TARGET_COLUMN, TRACEABILITY
from f1_race_intelligence.modeling import checks, preprocessing, split as split_module
from f1_race_intelligence.modeling.loaders import FeatureDatasetLoader, feature_columns
from f1_race_intelligence.modeling.models import ModelingReport, SplitArtifacts
from f1_race_intelligence.modeling.report import save_report
from f1_race_intelligence.modeling.split import SPLITS, TEST, TRAIN, VALIDATION, SplitStrategy
from f1_race_intelligence.utils.files import write_json_unique
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PIPELINE_NAME = "modeling_dataset"
PIPELINE_VERSION = "1.0.0"

PREPROCESSOR_FILE = "preprocessor.joblib"

GRAIN_COLUMNS = ("session_key", "driver_number", "lap_number")

# The prefix lives with the guard that enforces it; re-exported here because
# this is the module that writes the columns.
KEY_PREFIX = checks.KEY_PREFIX


class ModelingRunner:
    """Turns the M5 feature dataset into train/validation/test artifacts."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        loader: Optional[FeatureDatasetLoader] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._config = self._settings.modeling
        self._loader = loader or FeatureDatasetLoader(input_path=self._config.input_path)

    def run(self) -> ModelingReport:
        """Build the modelling datasets and return the run's report."""
        started = time.monotonic()
        report = ModelingReport(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config=self._config_snapshot(),
            target=self._target_definition(),
        )

        dataset = self._loader.load(
            years=list(self._config.years) or None,
            session_keys=list(self._config.session_keys) or None,
        )
        report.source = {**dataset.to_dict(), "input_path": str(self._loader.input_path)}

        logger.info(
            "modeling_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "rows_in": dataset.rows_in,
                "rows_modelled": dataset.rows_out,
                "sessions": dataset.sessions,
            },
        )

        if dataset.frame.empty:
            report.aborted = "no rows with a valid target were found in the feature dataset"
            return self._finish(report, started)

        columns = feature_columns(dataset.frame)
        report.features = {
            "numeric": columns["numeric"],
            "categorical": columns["categorical"],
            "source": "M5 catalogue (features/selection.py)",
        }

        assignments = split_module.assign_sessions(
            dataset.frame,
            strategy=SplitStrategy(self._config.split_strategy),
            train_fraction=self._config.train_fraction,
            validation_fraction=self._config.validation_fraction,
            train_years=self._config.train_years,
            validation_years=self._config.validation_years,
            test_years=self._config.test_years,
            validation_share_of_remainder=self._config.validation_share_of_remainder,
        )
        splits = split_module.apply_assignments(dataset.frame, assignments)
        report.split = {
            "strategy": self._config.split_strategy,
            "by": "whole sessions, ordered by session_date",
            **split_module.describe(assignments, splits),
            "assignments": [item.to_dict() for item in assignments],
        }
        self._warn_about_small_splits(report, splits)

        matrices = {
            name: preprocessing.prepare_matrix(frame, columns["numeric"], columns["categorical"])
            for name, frame in splits.items()
        }
        targets = {name: frame[TARGET_COLUMN].astype("float64") for name, frame in splits.items()}

        if matrices[TRAIN].empty:
            report.aborted = "the training split is empty; nothing can be fitted"
            return self._finish(report, started)

        preprocessor = preprocessing.build_preprocessor(
            columns["numeric"],
            columns["categorical"],
            numeric_imputation=self._config.numeric_imputation,
            scale_numeric=self._config.scale_numeric,
            add_missing_indicators=self._config.add_missing_indicators,
        )
        preprocessor.fit(matrices[TRAIN])

        transformed = {name: preprocessing.transform_to_frame(preprocessor, matrix) for name, matrix in matrices.items()}
        report.preprocessing = preprocessing.describe(
            preprocessor,
            columns["numeric"],
            columns["categorical"],
            scale_numeric=self._config.scale_numeric,
            add_missing_indicators=self._config.add_missing_indicators,
            numeric_imputation=self._config.numeric_imputation,
        )

        outcomes = self._run_checks(preprocessor, matrices, transformed, splits, targets, assignments, columns, dataset.frame)
        report.checks = [item.to_dict() for item in outcomes]
        report.blocking_failures = checks.blocking_failures(outcomes)

        if report.blocking_failures and self._config.fail_on_leakage:
            report.aborted = f"leakage checks failed: {', '.join(report.blocking_failures)}"
            logger.error("modeling_aborted", extra={"failed": report.blocking_failures})
            return self._finish(report, started)

        report.artifacts = self._write_artifacts(splits, transformed, targets, preprocessor, report)
        return self._finish(report, started)

    # -- pieces -----------------------------------------------------------

    def _run_checks(
        self,
        preprocessor,
        matrices: Dict[str, pd.DataFrame],
        transformed: Dict[str, pd.DataFrame],
        splits: Dict[str, pd.DataFrame],
        targets: Dict[str, pd.Series],
        assignments,
        columns: Dict[str, List[str]],
        source: pd.DataFrame,
    ) -> List[checks.CheckResult]:
        encoded_names = list(transformed[TRAIN].columns)
        matrix_columns = encoded_names + [TARGET_COLUMN] + [f"{KEY_PREFIX}{c}" for c in GRAIN_COLUMNS]
        return [
            checks.check_target_not_in_features(encoded_names),
            checks.check_forbidden_columns_not_in_features(encoded_names),
            checks.check_traceability_columns_not_in_model_features(encoded_names, matrix_columns),
            checks.check_no_session_overlap_between_splits(assignments),
            checks.check_split_rows_do_not_overlap(splits),
            checks.check_chronological_split(assignments),
            checks.check_preprocessing_fitted_on_train_only(
                preprocessor, matrices[TRAIN], columns["numeric"], columns["categorical"]
            ),
            checks.check_no_validation_statistics_used_in_training(preprocessor, matrices[TRAIN], matrices[VALIDATION]),
            checks.check_no_test_statistics_used_in_training(preprocessor, matrices[TRAIN], matrices[TEST]),
            checks.check_consistent_feature_schema(transformed),
            checks.check_categorical_unknown_handling(preprocessor, matrices[TRAIN], columns["categorical"]),
            checks.check_target_alignment(splits, targets, source),
            checks.check_target_rows_are_valid(splits, targets),
        ]

    def _write_artifacts(
        self,
        splits: Dict[str, pd.DataFrame],
        transformed: Dict[str, pd.DataFrame],
        targets: Dict[str, pd.Series],
        preprocessor,
        report: ModelingReport,
    ) -> List[SplitArtifacts]:
        """Write each split twice: as it is, and as a model consumes it.

        The untransformed copy keeps the traceability columns so any row can
        be followed back to M5 and from there to the raw files. The encoded
        matrix is what a model reads. Keeping both means an odd prediction
        can be traced to a lap without re-running anything.
        """
        base = Path(self._config.output_path)
        artifacts: List[SplitArtifacts] = []

        for name in SPLITS:
            frame = splits[name]
            directory = base / name
            directory.mkdir(parents=True, exist_ok=True)

            dataset_path = directory / "dataset.parquet"
            matrix_path = directory / "features.parquet"

            frame.to_parquet(dataset_path, engine="pyarrow", index=False)

            matrix = transformed[name].copy()
            matrix[TARGET_COLUMN] = targets[name].to_numpy() if not frame.empty else pd.Series(dtype="float64")
            # The keys are prefixed because a column can be both a key and a
            # feature: lap_number is one, and writing it unprefixed would
            # overwrite its transformed value with the raw lap count.
            for column in GRAIN_COLUMNS:
                if column in frame.columns:
                    matrix[f"{KEY_PREFIX}{column}"] = (
                        frame[column].to_numpy() if not frame.empty else pd.Series(dtype="Int64")
                    )
            matrix.to_parquet(matrix_path, engine="pyarrow", index=False)

            artifacts.append(
                SplitArtifacts(
                    name=name,
                    rows=len(frame),
                    sessions=int(frame["session_key"].nunique()) if not frame.empty else 0,
                    years=sorted({int(y) for y in frame["year"].dropna().unique()}) if not frame.empty else [],
                    dataset_path=str(dataset_path),
                    matrix_path=str(matrix_path),
                    bytes_written=dataset_path.stat().st_size + matrix_path.stat().st_size,
                    missing_before=self._missing_share(frame, report.features["numeric"] + report.features["categorical"]),
                    missing_after=self._missing_share(transformed[name], list(transformed[name].columns)),
                )
            )

        preprocessing_dir = base / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessing_dir / PREPROCESSOR_FILE)
        write_json_unique(
            preprocessing_dir,
            "feature_schema",
            {
                "input_features": report.features,
                "encoded_feature_names": report.preprocessing.get("output_feature_names", []),
                "target": TARGET_COLUMN,
                "traceability": list(TRACEABILITY),
                "key_columns_in_matrix": [f"{KEY_PREFIX}{column}" for column in GRAIN_COLUMNS],
                "note": (
                    "In features.parquet the model inputs are encoded_feature_names; the target and the "
                    f"'{KEY_PREFIX}' columns sit alongside them and must not be fed to a model."
                ),
            },
        )
        return artifacts

    @staticmethod
    def _missing_share(frame: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
        if frame.empty:
            return {}
        present = [column for column in columns if column in frame.columns]
        shares = (frame[present].isna().mean() * 100).round(2)
        return {column: float(value) for column, value in shares.items() if value > 0}

    def _warn_about_small_splits(self, report: ModelingReport, splits: Dict[str, pd.DataFrame]) -> None:
        """Say plainly when the history is too thin for the split to mean anything."""
        minimum = self._config.min_sessions_per_split
        for name in SPLITS:
            sessions = int(splits[name]["session_key"].nunique()) if not splits[name].empty else 0
            if sessions < minimum:
                report.warnings.append(
                    f"The {name} split holds {sessions} session(s), below the configured minimum of {minimum}. "
                    "The pipeline is technically valid but the split is not a statistically meaningful "
                    "evaluation; extract more races before drawing conclusions from it."
                )

    def _target_definition(self) -> Dict[str, Any]:
        return {
            "name": TARGET_COLUMN,
            "definition": "lap_duration(t+1), exactly as built in M5",
            "built_by": "M5",
            "rows_without_target": "dropped here and counted; M5 keeps them flagged",
        }

    def _config_snapshot(self) -> Dict[str, Any]:
        config = self._config
        return {
            "input_path": config.input_path,
            "output_path": config.output_path,
            "years": list(config.years),
            "session_keys": list(config.session_keys),
            "split_strategy": config.split_strategy,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "validation_share_of_remainder": config.validation_share_of_remainder,
            "train_years": list(config.train_years),
            "validation_years": list(config.validation_years),
            "test_years": list(config.test_years),
            "min_sessions_per_split": config.min_sessions_per_split,
            "numeric_imputation": config.numeric_imputation,
            "scale_numeric": config.scale_numeric,
            "add_missing_indicators": config.add_missing_indicators,
            "random_seed": config.random_seed,
            "fail_on_leakage": config.fail_on_leakage,
        }

    def _finish(self, report: ModelingReport, started: float) -> ModelingReport:
        report.finished_at = datetime.now(timezone.utc)
        path = save_report(report, self._config.report_path)
        logger.info(
            "modeling_summary",
            extra={
                "pipeline": PIPELINE_NAME,
                **report.summary(),
                "aborted": report.aborted,
                "duration_seconds": round(time.monotonic() - started, 3),
                "report_path": str(path),
            },
        )
        return report


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.modeling.runner``."""
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    report = ModelingRunner(settings).run()
    summary = report.summary()

    if report.aborted:
        print(f"Modeling dataset aborted: {report.aborted}")
        return

    rows = summary["rows_by_split"]
    print(
        f"Modeling dataset built: train {rows.get('train', 0)} / validation {rows.get('validation', 0)} / "
        f"test {rows.get('test', 0)} rows, {summary['encoded_feature_count']} encoded features."
    )
    for warning in report.warnings:
        print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
