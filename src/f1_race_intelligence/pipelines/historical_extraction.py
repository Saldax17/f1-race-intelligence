"""Reproducible extraction of historical OpenF1 data.

Run it with::

    python -m f1_race_intelligence.pipelines.historical_extraction

This module owns the *what* of historical ingestion — which years, which
sessions, which endpoints, in which order, and what to do when something
already exists or fails. The *how* of talking to OpenF1 stays entirely in
:class:`~f1_race_intelligence.ingestion.openf1.F1Client` (rate limiting,
retries, timeouts, HTTP error translation) and the *where* of persistence
stays in :class:`~f1_race_intelligence.storage.raw_storage.RawDataStorage`.

The traversal is::

    year -> meetings -> sessions -> (filtered by session type) -> endpoints

Every raw file maps to a deterministic path, so a second run with the same
configuration skips what it already has (unless ``overwrite`` is set) and
a failure on one session/endpoint never stops the rest of the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.ingestion.exceptions import OpenF1Error
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.pipelines.manifest import ExecutionManifest, ExtractionStatus, ManifestEntry
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

PIPELINE_NAME = "historical_extraction"
PIPELINE_VERSION = "1.0.0"

# Errors that mean "this one unit of work failed" rather than "the run is
# broken". ValueError covers F1Client's own guards against unbounded
# queries (e.g. car_data without a driver or date filter).
_RECOVERABLE_ERRORS = (OpenF1Error, ValueError)


@dataclass(frozen=True)
class SessionContext:
    """Everything an endpoint planner needs to describe one session."""

    client: F1Client
    storage: RawDataStorage
    year: int
    meeting_key: int
    session_key: int

    @property
    def partition(self) -> Dict[str, Any]:
        """Keys identifying this session, used for the raw storage layout."""
        return {"year": self.year, "meeting_key": self.meeting_key, "session_key": self.session_key}


@dataclass(frozen=True)
class ExtractionUnit:
    """One raw file to fetch and persist: the smallest retryable/skippable step."""

    parameters: Dict[str, Any]
    file_name: str
    fetch: Callable[[], Any]


EndpointPlanner = Callable[[SessionContext], Iterable[ExtractionUnit]]


def _session_scoped(client_method: str) -> EndpointPlanner:
    """Build a planner for endpoints that yield exactly one file per session."""

    def plan(context: SessionContext) -> Iterator[ExtractionUnit]:
        fetch = getattr(context.client, client_method)
        yield ExtractionUnit(
            parameters=context.partition,
            file_name=f"session_{context.session_key}.json",
            fetch=partial(fetch, session_key=context.session_key),
        )

    return plan


def _driver_numbers(context: SessionContext) -> List[int]:
    """Driver numbers for a session, preferring already-saved raw data.

    Reading the drivers file we may already have on disk keeps re-runs from
    spending rate-limit budget on a lookup whose answer cannot change for a
    historical session.
    """
    drivers: Any = None
    path = context.storage.resolve_path(
        "drivers", context.partition, file_name=f"session_{context.session_key}.json"
    )
    if path.exists():
        try:
            drivers = context.storage.load(
                "drivers", context.partition, file_name=f"session_{context.session_key}.json"
            )
        except (OSError, ValueError):
            logger.warning("raw_file_unreadable", extra={"endpoint": "drivers", "path": str(path)})
            drivers = None

    if drivers is None:
        drivers = context.client.get_drivers(session_key=context.session_key)

    numbers = {
        driver["driver_number"]
        for driver in drivers or []
        if isinstance(driver, Mapping) and driver.get("driver_number") is not None
    }
    return sorted(numbers)


def _plan_car_data(context: SessionContext) -> Iterator[ExtractionUnit]:
    """Car telemetry must be requested per driver.

    ``F1Client.get_car_data`` refuses an unfiltered session-wide call
    because it can return millions of rows, so this fans out into one file
    per driver inside the session's directory.
    """
    for driver_number in _driver_numbers(context):
        yield ExtractionUnit(
            parameters={**context.partition, "driver_number": driver_number},
            file_name=f"driver_{driver_number}.json",
            fetch=partial(
                context.client.get_car_data,
                session_key=context.session_key,
                driver_number=driver_number,
            ),
        )


# The single source of truth for which endpoint names are valid in config.
ENDPOINT_PLANNERS: Dict[str, EndpointPlanner] = {
    "drivers": _session_scoped("get_drivers"),
    "laps": _session_scoped("get_laps"),
    "car_data": _plan_car_data,
    "position": _session_scoped("get_positions"),
    "intervals": _session_scoped("get_intervals"),
    "stints": _session_scoped("get_stints"),
    "pit": _session_scoped("get_pit"),
    "weather": _session_scoped("get_weather"),
    "race_control": _session_scoped("get_race_control"),
}


@dataclass
class _UnitResult:
    """Outcome of one :class:`ExtractionUnit`, including data when it was needed."""

    status: ExtractionStatus
    data: Any = None


class HistoricalExtractionPipeline:
    """Discovers meetings/sessions and extracts the configured endpoints."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        client: Optional[F1Client] = None,
        storage: Optional[RawDataStorage] = None,
    ) -> None:
        """Create a pipeline.

        Args:
            settings: Application settings; loaded from config when omitted.
            client: An OpenF1 client. When omitted one is built from
                ``settings`` and closed by :meth:`close`.
            storage: Raw data storage; defaults to the configured
                ``output_path``.

        Raises:
            ValueError: if the configuration names an unknown endpoint. This
                fails at construction time, before any HTTP call is made.
        """
        self._settings = settings or load_settings()
        self._config = self._settings.historical_extraction
        self._owns_client = client is None
        self._client = client or F1Client(settings=self._settings)
        self._storage = storage or RawDataStorage(base_path=self._config.output_path)

        unknown = [name for name in self._config.endpoints if name not in ENDPOINT_PLANNERS]
        if unknown:
            raise ValueError(
                f"Unknown endpoints in historical_extraction configuration: {sorted(unknown)}. "
                f"Supported endpoints: {sorted(ENDPOINT_PLANNERS)}"
            )

    def close(self) -> None:
        """Close the client if this pipeline created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HistoricalExtractionPipeline":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- orchestration ----------------------------------------------------

    def run(self) -> ExecutionManifest:
        """Extract every configured year and return the execution manifest."""
        manifest = ExecutionManifest(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config=self._config_snapshot(),
        )

        logger.info(
            "pipeline_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "version": PIPELINE_VERSION,
                "years": list(self._config.years),
                "session_types": list(self._config.session_types),
                "endpoints": list(self._config.endpoints),
                "overwrite": self._config.overwrite,
            },
        )

        for year in self._config.years:
            self._process_year(year, manifest)

        manifest.finished_at = datetime.now(timezone.utc)
        manifest_path = manifest.save(self._config.manifest_path)

        logger.info(
            "pipeline_summary",
            extra={
                "pipeline": PIPELINE_NAME,
                **manifest.summary(),
                "duration_seconds": manifest.duration_seconds,
                "manifest_path": str(manifest_path),
            },
        )
        return manifest

    def _process_year(self, year: int, manifest: ExecutionManifest) -> None:
        logger.info("year_processing", extra={"year": year})

        meetings = self.discover_meetings(year, manifest)
        for meeting in meetings:
            self._process_meeting(year, meeting, manifest)

    def _process_meeting(self, year: int, meeting: Mapping[str, Any], manifest: ExecutionManifest) -> None:
        meeting_key = meeting.get("meeting_key")
        if meeting_key is None:
            logger.warning("meeting_without_key_skipped", extra={"year": year})
            return

        logger.info(
            "meeting_processing",
            extra={"year": year, "meeting_key": meeting_key, "meeting_name": meeting.get("meeting_name")},
        )
        manifest.record_meeting(meeting_key)

        sessions = self.filter_sessions(self.discover_sessions(year, meeting_key, manifest))
        for session in sessions:
            self._process_session(year, meeting_key, session, manifest)

    def _process_session(
        self,
        year: int,
        meeting_key: int,
        session: Mapping[str, Any],
        manifest: ExecutionManifest,
    ) -> None:
        session_key = session.get("session_key")
        if session_key is None:
            logger.warning("session_without_key_skipped", extra={"year": year, "meeting_key": meeting_key})
            return

        logger.info(
            "session_processing",
            extra={
                "year": year,
                "meeting_key": meeting_key,
                "session_key": session_key,
                "session_name": session.get("session_name"),
            },
        )
        manifest.record_session(session_key)

        context = SessionContext(
            client=self._client,
            storage=self._storage,
            year=year,
            meeting_key=meeting_key,
            session_key=session_key,
        )
        for endpoint in self._config.endpoints:
            self.extract_endpoint(endpoint, context, manifest)

    # -- discovery --------------------------------------------------------

    def discover_meetings(self, year: int, manifest: ExecutionManifest) -> List[Mapping[str, Any]]:
        """Return the meetings of a season, sorted by ``meeting_key``.

        The response is persisted like any other raw data, so a re-run can
        reuse it instead of querying OpenF1 again.
        """
        result = self._process_unit(
            endpoint="meetings",
            parameters={"year": year},
            file_name="meetings.json",
            fetch=partial(self._client.get_meetings, year=year),
            manifest=manifest,
            load_existing=True,
        )
        return _sorted_by(result.data, "meeting_key")

    def discover_sessions(
        self,
        year: int,
        meeting_key: int,
        manifest: ExecutionManifest,
    ) -> List[Mapping[str, Any]]:
        """Return every session of a meeting, sorted by ``session_key``."""
        result = self._process_unit(
            endpoint="sessions",
            parameters={"year": year, "meeting_key": meeting_key},
            file_name="sessions.json",
            fetch=partial(self._client.get_sessions, year=year, meeting_key=meeting_key),
            manifest=manifest,
            load_existing=True,
        )
        return _sorted_by(result.data, "session_key")

    def filter_sessions(self, sessions: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Keep only the session types named in configuration.

        Matching is case-insensitive against OpenF1's ``session_name`` *or*
        ``session_type``, so ``"Race"``, ``"Sprint Qualifying"`` and a
        coarser ``"Practice"`` all select what a reader would expect. An
        empty ``session_types`` list keeps every session.
        """
        wanted = {value.strip().casefold() for value in self._config.session_types if value.strip()}
        if not wanted:
            return list(sessions)

        return [
            session
            for session in sessions
            if str(session.get("session_name", "")).casefold() in wanted
            or str(session.get("session_type", "")).casefold() in wanted
        ]

    # -- extraction -------------------------------------------------------

    def extract_endpoint(
        self,
        endpoint: str,
        context: SessionContext,
        manifest: ExecutionManifest,
    ) -> None:
        """Extract one endpoint for one session.

        Planning can itself fail (``car_data`` needs the session's drivers
        first), so that is treated like any other endpoint failure: recorded
        and stepped over.
        """
        planner = ENDPOINT_PLANNERS[endpoint]
        try:
            units = list(planner(context))
        except _RECOVERABLE_ERRORS as exc:
            self._record_failure(endpoint, context.partition, exc, manifest)
            return

        for unit in units:
            self._process_unit(
                endpoint=endpoint,
                parameters=unit.parameters,
                file_name=unit.file_name,
                fetch=unit.fetch,
                manifest=manifest,
            )

    def _process_unit(
        self,
        *,
        endpoint: str,
        parameters: Mapping[str, Any],
        file_name: str,
        fetch: Callable[[], Any],
        manifest: ExecutionManifest,
        load_existing: bool = False,
    ) -> _UnitResult:
        """Fetch-and-save one raw file, unless it is already there.

        This is the single place where the skip/download/fail decision is
        made, so every endpoint — and discovery itself — behaves the same
        way and lands in the manifest the same way.

        Args:
            load_existing: When the file is skipped, also read it back.
                Discovery needs the data to keep traversing; endpoint
                extraction does not, and avoids the read.
        """
        path = self._storage.resolve_path(endpoint, parameters, file_name=file_name)
        context = {"endpoint": endpoint, **_clean(parameters), "path": str(path)}

        if path.exists() and not self._config.overwrite:
            data, readable = self._read_existing(endpoint, parameters, file_name, path, load_existing)
            if readable:
                logger.info("raw_file_skipped", extra={**context, "status": ExtractionStatus.SKIPPED.value})
                manifest.add(
                    _entry(
                        endpoint,
                        parameters,
                        status=ExtractionStatus.SKIPPED,
                        file_path=str(path),
                        record_count=_record_count(data),
                    )
                )
                return _UnitResult(ExtractionStatus.SKIPPED, data)
            # An unreadable file is treated as missing: fall through and refetch.

        try:
            data = fetch()
        except _RECOVERABLE_ERRORS as exc:
            self._record_failure(endpoint, parameters, exc, manifest, path=path)
            return _UnitResult(ExtractionStatus.FAILED)

        saved_path = self._storage.save(endpoint, parameters, data=data, file_name=file_name)
        record_count = _record_count(data)
        logger.info(
            "raw_file_downloaded",
            extra={
                **context,
                "path": str(saved_path),
                "record_count": record_count,
                "status": ExtractionStatus.DOWNLOADED.value,
            },
        )
        manifest.add(
            _entry(
                endpoint,
                parameters,
                status=ExtractionStatus.DOWNLOADED,
                file_path=str(saved_path),
                record_count=record_count,
            )
        )
        return _UnitResult(ExtractionStatus.DOWNLOADED, data)

    def _read_existing(
        self,
        endpoint: str,
        parameters: Mapping[str, Any],
        file_name: str,
        path: Path,
        load_existing: bool,
    ) -> Tuple[Any, bool]:
        """Return ``(data, readable)`` for a file that is already on disk."""
        if not load_existing:
            return None, True
        try:
            return self._storage.load(endpoint, parameters, file_name=file_name), True
        except (OSError, ValueError):
            logger.warning("raw_file_unreadable", extra={"endpoint": endpoint, "path": str(path)})
            return None, False

    def _record_failure(
        self,
        endpoint: str,
        parameters: Mapping[str, Any],
        exc: Exception,
        manifest: ExecutionManifest,
        path: Optional[Path] = None,
    ) -> None:
        logger.error(
            "extraction_failed",
            extra={
                "endpoint": endpoint,
                **_clean(parameters),
                "path": str(path) if path else None,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "status": ExtractionStatus.FAILED.value,
            },
        )
        manifest.add(
            _entry(
                endpoint,
                parameters,
                status=ExtractionStatus.FAILED,
                error=str(exc),
                error_type=type(exc).__name__,
                # None for failures that never got a response (timeouts, connection errors).
                status_code=getattr(exc, "status_code", None),
            )
        )

    def _config_snapshot(self) -> Dict[str, Any]:
        """The configuration this run used, embedded in the manifest."""
        return {
            "years": list(self._config.years),
            "session_types": list(self._config.session_types),
            "endpoints": list(self._config.endpoints),
            "output_path": self._config.output_path,
            "overwrite": self._config.overwrite,
        }


def _entry(
    endpoint: str,
    parameters: Mapping[str, Any],
    *,
    status: ExtractionStatus,
    file_path: Optional[str] = None,
    record_count: Optional[int] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    status_code: Optional[int] = None,
) -> ManifestEntry:
    return ManifestEntry(
        endpoint=endpoint,
        status=status,
        year=parameters.get("year"),
        meeting_key=parameters.get("meeting_key"),
        session_key=parameters.get("session_key"),
        driver_number=parameters.get("driver_number"),
        file_path=file_path,
        record_count=record_count,
        error=error,
        error_type=error_type,
        status_code=status_code,
    )


def _clean(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in parameters.items() if value is not None}


def _record_count(data: Any) -> Optional[int]:
    return len(data) if isinstance(data, list) else None


def _sorted_by(items: Any, key: str) -> List[Mapping[str, Any]]:
    """Sort API results by a key so the traversal order never depends on the API."""
    if not isinstance(items, list):
        return []
    records = [item for item in items if isinstance(item, Mapping)]
    return sorted(records, key=lambda item: (item.get(key) is None, item.get(key)))


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.pipelines.historical_extraction``."""
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    with HistoricalExtractionPipeline(settings) as pipeline:
        manifest = pipeline.run()

    summary = manifest.summary()
    print(
        f"Historical extraction finished: "
        f"{summary['downloaded']} downloaded, {summary['skipped']} skipped, {summary['failed']} failed "
        f"across {summary['meetings']} meetings and {summary['sessions']} sessions."
    )


if __name__ == "__main__":
    main()
