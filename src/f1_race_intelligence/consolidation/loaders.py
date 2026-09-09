"""Reading one session's raw data.

Everything here goes through :class:`RawDataCatalog`, so consolidation
never learns the raw layout or the envelope format. The one asymmetry is
``car_data``: its files are handed back as *paths to iterate*, not as
loaded records, because a single session holds around 180 MB of telemetry
and the whole point of the design is that only one driver's file is in
memory at a time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFile, RawFileError

CAR_DATA = "car_data"

# Endpoints loaded into memory whole. They are small: the largest, intervals,
# is around 30 000 records for a race.
EAGER_ENDPOINTS: Tuple[str, ...] = (
    "laps",
    "drivers",
    "stints",
    "pit",
    "position",
    "intervals",
    "weather",
    "race_control",
)


@dataclass
class SessionSources:
    """One session's raw data, plus the telemetry files left unread."""

    session_key: int
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    records: Dict[str, List[Mapping[str, Any]]] = field(default_factory=dict)
    car_data_files: List[RawFile] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)

    def get(self, endpoint: str) -> List[Mapping[str, Any]]:
        return self.records.get(endpoint, [])

    @property
    def available_endpoints(self) -> List[str]:
        found = [name for name, rows in self.records.items() if rows]
        if self.car_data_files:
            found.append(CAR_DATA)
        return sorted(found)


class SessionLoader:
    """Groups the raw tree by session and loads one session at a time."""

    def __init__(self, catalog: RawDataCatalog) -> None:
        self._catalog = catalog

    def discover_sessions(
        self,
        years: Optional[List[int]] = None,
        session_keys: Optional[List[int]] = None,
    ) -> List[int]:
        """Session keys present in the raw tree, in a stable order.

        A session counts as present when it has a ``laps`` file: laps is the
        base of the dataset, and without it there is nothing to consolidate.
        """
        found = set()
        for raw_file in self._catalog.discover(endpoints=["laps"]):
            if raw_file.session_key is None:
                continue
            if years and raw_file.year not in years:
                continue
            if session_keys and raw_file.session_key not in session_keys:
                continue
            found.add(raw_file.session_key)
        return sorted(found)

    def load(self, session_key: int) -> SessionSources:
        """Load one session: everything but telemetry, which stays on disk."""
        by_endpoint: Dict[str, List[RawFile]] = defaultdict(list)
        year = meeting_key = None

        for raw_file in self._catalog.discover():
            if raw_file.session_key != session_key:
                continue
            by_endpoint[raw_file.endpoint].append(raw_file)
            year = year or raw_file.year
            meeting_key = meeting_key or raw_file.meeting_key

        sources = SessionSources(session_key=session_key, year=year, meeting_key=meeting_key)

        for endpoint in EAGER_ENDPOINTS:
            rows: List[Mapping[str, Any]] = []
            for raw_file in sorted(by_endpoint.get(endpoint, []), key=lambda item: item.sort_key):
                try:
                    rows.extend(cleaned(self._catalog.load(raw_file).records))
                except RawFileError as exc:
                    # One damaged file must not sink the session; the caller
                    # reports it and consolidates what is readable.
                    sources.unreadable.append(f"{raw_file.path}: {exc}")
            sources.records[endpoint] = rows

        sources.car_data_files = sorted(by_endpoint.get(CAR_DATA, []), key=lambda item: item.sort_key)
        return sources

    def iter_car_data(self, raw_file: RawFile) -> Iterator[Mapping[str, Any]]:
        """Records of one telemetry file. Raises :class:`RawFileError`."""
        yield from cleaned(self._catalog.load(raw_file).records)


def cleaned(records: List[Any]) -> List[Mapping[str, Any]]:
    """Keep only the entries that are actually objects.

    Malformed entries are reported by validation (M3); here they are simply
    not consolidated, rather than crashing a join.
    """
    return [record for record in records if isinstance(record, Mapping)]
