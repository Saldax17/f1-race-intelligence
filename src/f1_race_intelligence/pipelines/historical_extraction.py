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

Every raw file maps to a deterministic key, so a second run with the same
configuration skips what it already has (unless ``overwrite`` is set) and
a failure on one session/endpoint never stops the rest of the run.

Besides the full traversal, two entry points exist for batch execution:
:meth:`HistoricalExtractionPipeline.plan` lists the sessions to extract (one
work unit each), and :meth:`HistoricalExtractionPipeline.run_session`
extracts exactly one of them. Session runs are independent, so they can be
distributed — within the rate limit OpenF1 allows per client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.ingestion.exceptions import OpenF1Error
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.pipelines.manifest import ExecutionManifest, ExtractionStatus, ManifestEntry
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFileError
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from f1_race_intelligence.storage.store import DataStore
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
    file_name = f"session_{context.session_key}.json"
    if context.storage.exists("drivers", context.partition, file_name=file_name):
        try:
            drivers = context.storage.load("drivers", context.partition, file_name=file_name)
        except (OSError, ValueError):
            location = context.storage.uri("drivers", context.partition, file_name=file_name)
            logger.warning("raw_file_unreadable", extra={"endpoint": "drivers", "path": location})
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
        store: Optional[DataStore] = None,
    ) -> None:
        """Create a pipeline.

        Args:
            settings: Application settings; loaded from config when omitted.
            client: An OpenF1 client. When omitted one is built from
                ``settings`` and closed by :meth:`close`.
            storage: Raw data storage; defaults to the configured
                ``output_path`` in ``store``.
            store: Where raw files and manifests go; defaults to the
                configured ``storage`` section.

        Raises:
            ValueError: if the configuration names an unknown endpoint. This
                fails at construction time, before any HTTP call is made.
        """
        self._settings = settings or load_settings()
        self._config = self._settings.historical_extraction
        self._store = store or (storage.store if storage is not None else DataStore.from_settings(self._settings))
        self._storage = storage or RawDataStorage(base_path=self._config.output_path, store=self._store)
        self._manifest_root = self._store.root(Layer.EXTRACTION_MANIFESTS, self._config.manifest_path)
        self._owns_client = client is None
        self._client = client or F1Client(settings=self._settings)

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

        return self._finish(manifest)

    def run_session(self, year: int, session_key: int, meeting_key: Optional[int] = None) -> ExecutionManifest:
        """Extract every configured endpoint of one session: the batch unit of work.

        The meeting is taken from the argument, else from a sessions
        catalogue already in storage (written by :meth:`plan` or an earlier
        run), and only as a last resort from one ``/sessions`` lookup. The
        session is extracted even if its type is not in ``session_types``:
        asking for it by key is an explicit choice, so it is logged, not refused.

        The manifest is written under a per-session prefix on the lake
        layout, and records the scope in ``config.scope``.
        """
        scope = SessionScope(year=year, meeting_key=meeting_key, session_key=session_key)
        manifest = ExecutionManifest(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config=self._config_snapshot(scope=scope),
        )
        logger.info(
            "pipeline_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "version": PIPELINE_VERSION,
                "mode": "session",
                **scope.log_fields(),
                "endpoints": list(self._config.endpoints),
                "overwrite": self._config.overwrite,
            },
        )

        meeting_key = meeting_key if meeting_key is not None else self._meeting_for_session(year, session_key)
        if meeting_key is None:
            self._record_failure(
                "sessions",
                {"year": year, "session_key": session_key},
                LookupError(f"session {session_key} was not found for {year}"),
                manifest,
            )
            return self._finish(manifest, scope=scope)

        manifest.config["scope"]["meeting_key"] = meeting_key
        manifest.record_meeting(meeting_key)
        sessions = self.discover_sessions(year, meeting_key, manifest)
        session = next((item for item in sessions if item.get("session_key") == session_key), None)
        resolved = SessionScope(year=year, meeting_key=meeting_key, session_key=session_key)
        if session is None and sessions:
            # The meeting's catalogue is there and does not list this session:
            # extracting would file the data under the wrong meeting.
            self._record_failure(
                "sessions",
                {"year": year, "meeting_key": meeting_key, "session_key": session_key},
                LookupError(f"session {session_key} is not part of meeting {meeting_key} in {year}"),
                manifest,
            )
            return self._finish(manifest, scope=resolved)
        if session is None:
            # The catalogue itself could not be fetched (recorded as a failure);
            # a bare record still lets the endpoints be extracted.
            session = {"session_key": session_key}
        elif not self.filter_sessions([session]):
            logger.warning(
                "session_outside_configured_types",
                extra={"year": year, "session_key": session_key, "session_name": session.get("session_name")},
            )

        self._process_session(year, meeting_key, session, manifest)
        return self._finish(manifest, scope=resolved)

    def plan(self, years: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        """List the sessions a run would extract, one work unit per session.

        Discovery is the only thing downloaded: the meetings and sessions
        catalogues, which are persisted (and reused on a re-run) like any
        other raw data, so the session jobs that follow need no discovery
        calls of their own. A manifest of the discovery is written too.

        Returns:
            ``[{"year": …, "meeting_key": …, "session_key": …, "session_name": …}, …]``
            in a stable order.
        """
        years = list(years) if years is not None else list(self._config.years)
        manifest = ExecutionManifest(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            config={**self._config_snapshot(), "years": years, "scope": {"mode": "plan"}},
        )

        units: List[Dict[str, Any]] = []
        for year in years:
            for meeting in self.discover_meetings(year, manifest):
                meeting_key = meeting.get("meeting_key")
                if meeting_key is None:
                    continue
                manifest.record_meeting(meeting_key)
                for session in self.filter_sessions(self.discover_sessions(year, meeting_key, manifest)):
                    if session.get("session_key") is None:
                        continue
                    units.append(
                        {
                            "year": year,
                            "meeting_key": meeting_key,
                            "session_key": session["session_key"],
                            "session_name": session.get("session_name"),
                        }
                    )

        self._finish(manifest)
        logger.info("pipeline_plan", extra={"pipeline": PIPELINE_NAME, "years": years, "work_units": len(units)})
        return units

    def _finish(self, manifest: ExecutionManifest, scope: Optional[SessionScope] = None) -> ExecutionManifest:
        """Close the manifest, store it, and log the run summary."""
        manifest.finished_at = datetime.now(timezone.utc)
        prefix = self._store.layout.records_prefix(
            self._manifest_root,
            year=scope.year if scope else None,
            session_key=scope.session_key if scope else None,
        )
        manifest_path = manifest.save(prefix, backend=self._store.backend)

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

    def _meeting_for_session(self, year: int, session_key: int) -> Optional[int]:
        """Find a session's meeting, preferring catalogues already in storage."""
        catalog = RawDataCatalog(base_path=self._config.output_path, store=self._store)
        for raw_file in catalog.discover(endpoints=["sessions"], year=year):
            try:
                records = catalog.load(raw_file).records
            except RawFileError:  # an unreadable catalogue is just not a source here
                continue
            for record in records:
                if isinstance(record, Mapping) and record.get("session_key") == session_key:
                    return record.get("meeting_key") or raw_file.meeting_key

        try:
            records = self._client.get_sessions(session_key=session_key)
        except _RECOVERABLE_ERRORS as exc:
            logger.error(
                "session_lookup_failed",
                extra={"year": year, "session_key": session_key, "error": str(exc), "error_type": type(exc).__name__},
            )
            return None
        for record in records or []:
            if isinstance(record, Mapping) and record.get("session_key") == session_key:
                if record.get("year") not in (None, year):
                    # Storing it under the requested year would misfile the data.
                    logger.error(
                        "session_year_mismatch",
                        extra={"session_key": session_key, "requested_year": year, "actual_year": record.get("year")},
                    )
                    return None
                return record.get("meeting_key")
        return None

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
        path = self._storage.uri(endpoint, parameters, file_name=file_name)
        context = {"endpoint": endpoint, **_clean(parameters), "path": path}

        if not self._config.overwrite and self._storage.exists(endpoint, parameters, file_name=file_name):
            data, readable = self._read_existing(endpoint, parameters, file_name, path, load_existing)
            if readable:
                logger.info("raw_file_skipped", extra={**context, "status": ExtractionStatus.SKIPPED.value})
                manifest.add(
                    _entry(
                        endpoint,
                        parameters,
                        status=ExtractionStatus.SKIPPED,
                        file_path=path,
                        record_count=_record_count(data),
                    )
                )
                return _UnitResult(ExtractionStatus.SKIPPED, data)
            # An unreadable file is treated as missing: fall through and refetch.

        started = time.monotonic()
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
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
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
        path: str,
        load_existing: bool,
    ) -> Tuple[Any, bool]:
        """Return ``(data, readable)`` for a file that is already stored."""
        if not load_existing:
            return None, True
        try:
            return self._storage.load(endpoint, parameters, file_name=file_name), True
        except (OSError, ValueError):
            logger.warning("raw_file_unreadable", extra={"endpoint": endpoint, "path": path})
            return None, False

    def _record_failure(
        self,
        endpoint: str,
        parameters: Mapping[str, Any],
        exc: Exception,
        manifest: ExecutionManifest,
        path: Optional[str] = None,
    ) -> None:
        logger.error(
            "extraction_failed",
            extra={
                "endpoint": endpoint,
                **_clean(parameters),
                "path": path,
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

    def _config_snapshot(self, scope: Optional[SessionScope] = None) -> Dict[str, Any]:
        """The configuration this run used, embedded in the manifest.

        A session run records its scope; a full run has no ``scope`` key,
        which is how a reader tells the two apart.
        """
        snapshot: Dict[str, Any] = {
            "years": [scope.year] if scope else list(self._config.years),
            "session_types": list(self._config.session_types),
            "endpoints": list(self._config.endpoints),
            "output_path": self._config.output_path,
            "overwrite": self._config.overwrite,
        }
        if scope is not None:
            snapshot["scope"] = {"mode": "session", **scope.to_dict()}
            snapshot["storage"] = self._store.describe()
        return snapshot


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
    configure_logging(
        level=settings.logging.level,
        json_format=settings.logging.json_format,
        file_path=settings.logging.file_path,
    )

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
