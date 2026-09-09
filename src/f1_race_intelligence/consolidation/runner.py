"""Orchestration of a consolidation run.

Run it with::

    python -m f1_race_intelligence.consolidation.runner

Sessions are consolidated one at a time and written as one Parquet file
each, partitioned by season. That keeps memory bounded, makes re-runs
idempotent per session, and lets a later stage read one race or all of
them with the same call.

``data/raw`` is never written to.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.consolidation.consolidator import SessionConsolidator
from f1_race_intelligence.consolidation.loaders import SessionLoader
from f1_race_intelligence.consolidation.models import ConsolidationReport, SessionResult
from f1_race_intelligence.consolidation.report import save_report
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PIPELINE_NAME = "data_consolidation"
PIPELINE_VERSION = "1.0.0"

CONSOLIDATED_DESPITE_FAILURE = (
    "The most recent validation run did not pass. These data were consolidated anyway "
    "because require_validation_pass is false; treat the dataset accordingly."
)


class ConsolidationRunner:
    """Turns validated raw data into the lap-grain analytical dataset."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        catalog: Optional[RawDataCatalog] = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._config = self._settings.consolidation
        self._catalog = catalog or RawDataCatalog(base_path=self._config.raw_path)
        self._loader = SessionLoader(self._catalog)
        self._consolidator = SessionConsolidator(
            self._loader,
            weather_tolerance_seconds=self._config.weather_tolerance_seconds,
        )

    def run(self) -> ConsolidationReport:
        """Consolidate every selected session and return the run's report."""
        report = ConsolidationReport(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config=self._config_snapshot(),
            source_validation=self._read_validation_status(),
        )

        sessions = self._loader.discover_sessions(
            years=list(self._config.years) or None,
            session_keys=list(self._config.session_keys) or None,
        )

        logger.info(
            "consolidation_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "sessions": len(sessions),
                "raw_path": self._config.raw_path,
                "output_path": self._config.output_path,
                "validation_status": report.source_validation.get("status"),
            },
        )

        if not self._validation_allows_run(report):
            report.finished_at = datetime.now(timezone.utc)
            save_report(report, self._config.report_path)
            return report

        for session_key in sessions:
            self._consolidate_session(session_key, report)

        report.finished_at = datetime.now(timezone.utc)
        report_path = save_report(report, self._config.report_path)

        logger.info(
            "consolidation_summary",
            extra={
                "pipeline": PIPELINE_NAME,
                **report.summary(),
                "duration_seconds": report.duration_seconds,
                "report_path": str(report_path),
            },
        )
        return report

    # -- per session ------------------------------------------------------

    def _consolidate_session(self, session_key: int, report: ConsolidationReport) -> None:
        destination = self._destination(session_key)
        if destination.exists() and not self._config.overwrite:
            logger.info(
                "session_skipped",
                extra={"session_key": session_key, "path": str(destination), "reason": "already consolidated"},
            )
            report.add(SessionResult(session_key=session_key, skipped=True, output_path=str(destination)))
            return

        sources = self._loader.load(session_key)
        logger.info(
            "session_consolidating",
            extra={
                "session_key": session_key,
                "year": sources.year,
                "meeting_key": sources.meeting_key,
                "endpoints": sources.available_endpoints,
            },
        )

        frame, result = self._consolidator.consolidate(sources)
        if sources.unreadable:
            result.checks["unreadable_files"] = sources.unreadable

        if result.error:
            logger.error("session_failed", extra={"session_key": session_key, "error": result.error})
            report.add(result)
            return

        destination = self._destination(session_key, year=result.year)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, engine="pyarrow", index=False)

        result.output_path = str(destination)
        result.output_bytes = destination.stat().st_size
        report.add(result)

        logger.info(
            "session_consolidated",
            extra={
                "session_key": session_key,
                "rows_in": result.rows_in,
                "rows_out": result.rows_out,
                "drivers": result.drivers,
                "path": str(destination),
                "bytes": result.output_bytes,
                "duration_seconds": result.duration_seconds,
            },
        )

    def _destination(self, session_key: int, year: Optional[int] = None) -> Path:
        """Where one session's Parquet file lives.

        Seasons are plain directories, not ``year=2023`` — that spelling is
        Hive partitioning, and pyarrow would then synthesise a ``year``
        column that collides with the real one stored in the file. The year
        belongs inside the data, so that a single file read on its own is
        still traceable.
        """
        base = Path(self._config.output_path)
        if year is None:
            existing = sorted(base.glob(f"*/session_{session_key}.parquet"))
            if existing:
                return existing[0]
            return base / f"session_{session_key}.parquet"
        return base / str(year) / f"session_{session_key}.parquet"

    # -- validation gate --------------------------------------------------

    def _read_validation_status(self) -> Dict[str, Any]:
        """What the latest validation run concluded about this raw data."""
        directory = Path(self._config.validation_path)
        status: Dict[str, Any] = {
            "status": None,
            "report": None,
            "require_validation_pass": self._config.require_validation_pass,
        }
        if not directory.is_dir():
            return status

        for path in reversed(sorted(directory.glob("validation_report_*.json"))):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            status["status"] = payload.get("global_status")
            status["report"] = str(path)
            status["summary"] = payload.get("summary")
            break
        return status

    def _validation_allows_run(self, report: ConsolidationReport) -> bool:
        """Decide whether to consolidate given the validation verdict."""
        status = report.source_validation.get("status")

        if not self._config.require_validation_pass:
            if status != "PASS":
                # Consolidating unvalidated or failing data is allowed, but the
                # dataset must never pretend it was clean.
                report.source_validation["note"] = CONSOLIDATED_DESPITE_FAILURE
                logger.warning(
                    "consolidating_without_validation_pass",
                    extra={"validation_status": status, "report": report.source_validation.get("report")},
                )
            return True

        if status == "PASS":
            return True

        report.aborted = (
            f"require_validation_pass is true and the latest validation status is {status or 'unknown'}"
        )
        logger.error("consolidation_aborted", extra={"reason": report.aborted})
        return False

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "raw_path": self._config.raw_path,
            "output_path": self._config.output_path,
            "years": list(self._config.years),
            "session_keys": list(self._config.session_keys),
            "require_validation_pass": self._config.require_validation_pass,
            "weather_tolerance_seconds": self._config.weather_tolerance_seconds,
            "overwrite": self._config.overwrite,
        }


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.consolidation.runner``."""
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    report = ConsolidationRunner(settings).run()
    summary = report.summary()

    if report.aborted:
        print(f"Consolidation aborted: {report.aborted}")
        return

    print(
        f"Consolidation finished: {summary['rows_out']} rows across "
        f"{summary['sessions_processed']} sessions "
        f"({summary['sessions_skipped']} skipped), {summary['rows_lost']} rows lost."
    )


if __name__ == "__main__":
    main()
