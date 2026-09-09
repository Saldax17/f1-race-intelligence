"""Orchestration of the feature engineering stage.

Run it with::

    python -m f1_race_intelligence.features.runner

Per session: read the consolidated laps, build the target, add the history
features, keep the columns the catalogue approves, run the anti-leakage
guards, and write Parquet. A session whose guards fail is not written,
unless configuration says otherwise — publishing a dataset that is known to
leak is worse than publishing none.

M5 learns nothing from the data. No scaling, no imputation, no encoding
fitted on the rows: anything that estimates parameters belongs inside a
training pipeline, where it can be fitted on the training split alone.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.features import checks, selection
from f1_race_intelligence.features.loaders import LapDatasetLoader, SessionFile
from f1_race_intelligence.features.models import FeatureReport, SessionFeatureResult
from f1_race_intelligence.features.report import save_report
from f1_race_intelligence.features.selection import TARGET_COLUMN
from f1_race_intelligence.features.target import add_target
from f1_race_intelligence.features.temporal import add_temporal_features, window_settings
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PIPELINE_NAME = "feature_engineering"
PIPELINE_VERSION = "1.0.0"

MISSING_VALUE_POLICY = (
    "M5 does not impute. Nulls are kept and reported so the modelling stage can decide, and fit any "
    "imputation on the training split alone. A null here means one of: the source never reported the "
    "value (speed traps), the record does not apply (a lapped car has no numeric gap), or the value "
    "cannot be derived from the past (the first lap has no previous lap)."
)


class FeatureRunner:
    """Turns the consolidated lap dataset into a modelling dataset."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        loader: Optional[LapDatasetLoader] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._config = self._settings.features
        self._loader = loader or LapDatasetLoader(input_path=self._config.input_path)
        self._windows = tuple(self._config.rolling_windows)

    def run(self) -> FeatureReport:
        """Build features for every selected session and return the report."""
        report = FeatureReport(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config=self._config_snapshot(),
            target=self._target_definition(),
            features=self._feature_inventory(),
            excluded=[item.to_dict() for item in selection.excluded_columns()],
            rolling=window_settings(self._windows),
            missing_value_policy=MISSING_VALUE_POLICY,
        )

        sessions = self._loader.discover(
            years=list(self._config.years) or None,
            session_keys=list(self._config.session_keys) or None,
        )
        logger.info(
            "feature_engineering_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "sessions": len(sessions),
                "input_path": str(self._loader.input_path),
                "features": report.features["count"],
                "windows": list(self._windows),
            },
        )

        frames: List[pd.DataFrame] = []
        for session_file in sessions:
            frame = self._process_session(session_file, report)
            if frame is not None:
                frames.append(frame)

        if frames:
            report.missing_values = self._missing_value_summary(pd.concat(frames, ignore_index=True))

        report.finished_at = datetime.now(timezone.utc)
        report_path = save_report(report, self._config.report_path)

        logger.info(
            "feature_engineering_summary",
            extra={
                "pipeline": PIPELINE_NAME,
                **report.summary(),
                "duration_seconds": report.duration_seconds,
                "report_path": str(report_path),
            },
        )
        return report

    # -- per session ------------------------------------------------------

    def _process_session(self, session_file: SessionFile, report: FeatureReport) -> Optional[pd.DataFrame]:
        started = time.monotonic()
        destination = self._destination(session_file)

        if destination.exists() and not self._config.overwrite:
            logger.info(
                "session_skipped",
                extra={"session_key": session_file.session_key, "reason": "features already built"},
            )
            report.add(
                SessionFeatureResult(
                    session_key=session_file.session_key,
                    year=session_file.year,
                    skipped=True,
                    output_path=str(destination),
                )
            )
            return None

        source = self._loader.load(session_file)
        result = SessionFeatureResult(
            session_key=session_file.session_key,
            year=session_file.year,
            rows_in=len(source),
        )

        with_target, breakdown = add_target(source)
        enriched = add_temporal_features(with_target, self._windows)
        frame = self._select_columns(enriched)

        feature_names = [name for name in selection.feature_columns() if name in frame.columns]
        outcomes = checks.run_all(
            frame,
            feature_names,
            windows=self._windows,
            correlation_threshold=self._config.correlation_alert_threshold,
        )

        result.target_breakdown = breakdown
        result.checks = [item.to_dict() for item in outcomes]
        result.blocking_failures = checks.blocking_failures(outcomes)
        result.rows_out = len(frame)
        result.rows_with_target = int(frame["has_target"].sum())
        result.drivers = int(frame["driver_number"].nunique())
        result.duration_seconds = round(time.monotonic() - started, 3)

        if result.blocking_failures and self._config.fail_on_leakage:
            result.error = f"anti-leakage checks failed: {', '.join(result.blocking_failures)}"
            logger.error(
                "session_failed_leakage_checks",
                extra={"session_key": session_file.session_key, "failed": result.blocking_failures},
            )
            report.add(result)
            return None

        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, engine="pyarrow", index=False)
        result.output_path = str(destination)
        result.output_bytes = destination.stat().st_size
        report.add(result)

        logger.info(
            "session_features_built",
            extra={
                "session_key": session_file.session_key,
                "rows": result.rows_out,
                "rows_with_target": result.rows_with_target,
                "features": len(feature_names),
                "path": str(destination),
                "duration_seconds": result.duration_seconds,
            },
        )
        return frame

    def _select_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the catalogue's columns, in its order, with settled dtypes."""
        wanted = [name for name in selection.output_columns() if name in frame.columns]
        selected = frame[wanted].copy()

        for column in selection.categorical_columns():
            if column in selected.columns:
                selected[column] = selected[column].astype("category")

        selected["has_target"] = selected["has_target"].astype("boolean")
        return selected.sort_values(list(selection.GRAIN), kind="stable").reset_index(drop=True)

    def _destination(self, session_file: SessionFile) -> Path:
        base = Path(self._config.output_path)
        if session_file.year is None:
            return base / f"session_{session_file.session_key}.parquet"
        return base / str(session_file.year) / f"session_{session_file.session_key}.parquet"

    # -- report material --------------------------------------------------

    def _target_definition(self) -> Dict[str, Any]:
        return {
            "name": TARGET_COLUMN,
            "definition": "lap_duration(t+1)",
            "grouped_by": list(selection.GROUP_KEYS),
            "ordered_by": selection.ORDER_KEY,
            "requires_consecutive_laps": True,
            "last_lap_policy": (
                "Rows without a valid target are kept with next_lap_time null and has_target false, "
                "so nothing is dropped silently. The modelling stage filters on has_target."
            ),
        }

    def _feature_inventory(self) -> Dict[str, Any]:
        names = selection.feature_columns()
        return {
            "count": len(names),
            "columns": names,
            "categorical": selection.categorical_columns(),
            "derived": selection.derived_columns(),
            "traceability": list(selection.TRACEABILITY),
            "by_role": {
                role: [item.name for item in selection.CATALOGUE if item.role.value == role]
                for role in sorted({item.role.value for item in selection.CATALOGUE})
            },
            "forbidden": sorted(selection.FORBIDDEN_AS_FEATURES),
        }

    def _missing_value_summary(self, frame: pd.DataFrame) -> Dict[str, Any]:
        shares = (frame.isna().mean() * 100).round(2)
        return {
            "rows": int(len(frame)),
            "columns_with_nulls": {
                column: float(value) for column, value in shares.items() if value > 0
            },
        }

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "input_path": self._config.input_path,
            "output_path": self._config.output_path,
            "rolling_windows": list(self._windows),
            "correlation_alert_threshold": self._config.correlation_alert_threshold,
            "fail_on_leakage": self._config.fail_on_leakage,
            "years": list(self._config.years),
            "session_keys": list(self._config.session_keys),
            "overwrite": self._config.overwrite,
        }


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.features.runner``."""
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    report = FeatureRunner(settings).run()
    summary = report.summary()

    print(
        f"Feature engineering finished: {summary['rows_out']} rows, "
        f"{summary['rows_with_target']} with a target, {summary['feature_count']} features "
        f"across {summary['sessions_processed']} sessions."
    )
    if summary["sessions_with_blocking_failures"]:
        print(f"  anti-leakage checks failed for sessions: {summary['sessions_with_blocking_failures']}")


if __name__ == "__main__":
    main()
