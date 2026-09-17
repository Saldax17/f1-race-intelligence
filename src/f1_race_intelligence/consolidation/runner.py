"""Orchestration of a consolidation run.

Run it with::

    python -m f1_race_intelligence.consolidation.runner

Sessions are consolidated one at a time and written as one Parquet file
each, partitioned by season. That keeps memory bounded, makes re-runs
idempotent per session, and lets a later stage read one race or all of
them with the same call.

Raw data is read from an input store and the dataset written to an output
store, which may differ (raw on S3, output on local disk, for instance).
Given a :class:`~f1_race_intelligence.scope.SessionScope` the run touches
one session only, which is how it is executed as an independent batch job.

Raw data is never written to.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.consolidation.consolidator import SessionConsolidator
from f1_race_intelligence.consolidation.loaders import SessionLoader
from f1_race_intelligence.consolidation.models import ConsolidationReport, SessionResult
from f1_race_intelligence.consolidation.report import save_report
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends.base import key_name
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog
from f1_race_intelligence.storage.store import DataStore
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
        store: Optional[DataStore] = None,
        output_store: Optional[DataStore] = None,
        scope: Optional[SessionScope] = None,
    ) -> None:
        """Create a runner.

        Args:
            settings: Application settings; loaded from config when omitted.
            catalog: Reader over the raw tree; defaults to ``raw_path`` in ``store``.
            store: Where raw data and validation reports are read from;
                defaults to the configured ``storage`` section.
            output_store: Where the dataset and the report are written;
                defaults to ``store``.
            scope: Consolidate one session (or season) only. Takes precedence
                over ``years`` and ``session_keys`` in configuration.
        """
        self._settings = settings or load_settings()
        self._config = self._settings.consolidation
        self._store = store or (catalog.store if catalog is not None else DataStore.from_settings(self._settings))
        self._output_store = output_store or self._store
        self._scope = scope or SessionScope()
        self._catalog = catalog or RawDataCatalog(base_path=self._config.raw_path, store=self._store)
        self._output_root = self._output_store.root(Layer.CONSOLIDATED, self._config.output_path)
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
            years=[self._scope.year] if self._scope.year is not None else (list(self._config.years) or None),
            session_keys=(
                [self._scope.session_key] if self._scope.is_session else (list(self._config.session_keys) or None)
            ),
            meeting_key=self._scope.meeting_key,
        )

        logger.info(
            "consolidation_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "sessions": len(sessions),
                "raw_path": self._catalog.location,
                "output_path": self._output_store.backend.uri(self._output_root),
                "validation_status": report.source_validation.get("status"),
            },
        )

        if not self._validation_allows_run(report):
            report.finished_at = datetime.now(timezone.utc)
            self._save_report(report)
            return report

        for session_key in sessions:
            self._consolidate_session(session_key, report)

        report.finished_at = datetime.now(timezone.utc)
        report_path = self._save_report(report)

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

    def _save_report(self, report: ConsolidationReport):
        root = self._output_store.root(Layer.CONSOLIDATION_REPORTS, self._config.report_path)
        return save_report(report, root, backend=self._output_store.backend)

    def _consolidate_session(self, session_key: int, report: ConsolidationReport) -> None:
        backend = self._output_store.backend
        destination = self._destination(session_key)
        if not self._config.overwrite and backend.exists(destination):
            location = backend.uri(destination)
            logger.info(
                "session_skipped",
                extra={"session_key": session_key, "path": location, "reason": "already consolidated"},
            )
            report.add(SessionResult(session_key=session_key, skipped=True, output_path=location))
            return

        sources = self._loader.load(session_key, year=self._scope.year, meeting_key=self._scope.meeting_key)
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
        result.output_bytes = backend.write_parquet(destination, frame)
        result.output_path = backend.uri(destination)
        report.add(result)

        logger.info(
            "session_consolidated",
            extra={
                "session_key": session_key,
                "rows_in": result.rows_in,
                "rows_out": result.rows_out,
                "drivers": result.drivers,
                "path": result.output_path,
                "bytes": result.output_bytes,
                "duration_seconds": result.duration_seconds,
            },
        )

    def _destination(self, session_key: int, year: Optional[int] = None) -> str:
        """Key of one session's Parquet file.

        In the local tree seasons are plain directories, not ``year=2023`` —
        that spelling is Hive partitioning, and pyarrow reading the directory
        would synthesise a ``year`` column that collides with the real one
        stored in the file. The lake layout does use ``year=``, and is only
        ever read file by file through a buffer (see
        ``StorageBackend.read_parquet``). Either way the year belongs inside
        the data, so a single file read on its own is still traceable.
        """
        layout = self._output_store.layout
        if year is None:
            if self._scope.year is not None:
                return layout.session_dataset_key(self._output_root, self._scope.year, session_key)
            listing = layout.session_dataset_listing_prefix(self._output_root)
            for key in self._output_store.backend.list(listing):
                parsed = layout.parse_session_dataset_key(self._output_root, key)
                if parsed is not None and parsed[1] == session_key and parsed[0] is not None:
                    return key
        return layout.session_dataset_key(self._output_root, year, session_key)

    # -- validation gate --------------------------------------------------

    def _read_validation_status(self) -> Dict[str, Any]:
        """What the latest relevant validation run concluded about this raw data.

        A session job takes the newest report that covered its session: a
        full run, or a validation job scoped to that same session. A full
        run takes the newest full validation. Reports written before scoped
        runs existed carry no filter and count as full runs.
        """
        backend, layout = self._store.backend, self._store.layout
        root = self._store.root(Layer.VALIDATION_REPORTS, self._config.validation_path)
        status: Dict[str, Any] = {
            "status": None,
            "report": None,
            "require_validation_pass": self._config.require_validation_pass,
        }

        prefixes = [root]
        if self._scope.is_session:
            prefixes.insert(0, layout.records_prefix(root, year=self._scope.year, session_key=self._scope.session_key))

        seen = set()
        for prefix in prefixes:
            keys = [
                key
                for key in backend.list(prefix)
                if key not in seen and key_name(key).startswith("validation_report_") and key.endswith(".json")
            ]
            seen.update(keys)
            for key in sorted(keys, key=lambda item: (key_name(item), item), reverse=True):
                try:
                    payload = backend.read_json(key)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict) or not self._report_applies(payload):
                    continue
                status["status"] = payload.get("global_status")
                status["report"] = backend.uri(key)
                status["summary"] = payload.get("summary")
                return status
        return status

    def _report_applies(self, payload: Dict[str, Any]) -> bool:
        source = payload.get("source")
        report_filter = source.get("filter") if isinstance(source, dict) else None
        if not report_filter:
            return True
        return self._scope.is_session and report_filter.get("session_key") == self._scope.session_key

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
        snapshot = {
            "raw_path": self._config.raw_path,
            "output_path": self._config.output_path,
            "years": list(self._config.years),
            "session_keys": list(self._config.session_keys),
            "require_validation_pass": self._config.require_validation_pass,
            "weather_tolerance_seconds": self._config.weather_tolerance_seconds,
            "overwrite": self._config.overwrite,
            "input_storage": self._store.describe(),
            "output_storage": self._output_store.describe(),
        }
        if not self._scope.is_empty:
            snapshot["scope"] = self._scope.to_dict()
        return snapshot


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.consolidation.runner``."""
    settings = load_settings()
    configure_logging(
        level=settings.logging.level,
        json_format=settings.logging.json_format,
        file_path=settings.logging.file_path,
    )

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
