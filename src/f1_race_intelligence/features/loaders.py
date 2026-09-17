"""Reading the consolidated lap dataset produced by M4.

Sessions are read one file at a time. The consolidated dataset is small
compared with the raw telemetry it came from — a race is roughly a
thousand rows — but per-session reading keeps the output partitioned the
same way and lets a full history be processed race by race.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import List, Optional, Union

import pandas as pd

from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore


@dataclass(frozen=True)
class SessionFile:
    """One consolidated session in storage."""

    key: str
    session_key: int
    year: Optional[int] = None
    uri: str = ""

    @property
    def path(self) -> PurePath:
        """A local path on disk; a pure path of the key on object storage."""
        location = self.uri or self.key
        return PurePosixPath(self.key) if "://" in location else Path(location)

    @property
    def sort_key(self) -> tuple:
        return (self.year if self.year is not None else -1, self.session_key)


class LapDatasetLoader:
    """Finds and reads the Parquet files M4 wrote."""

    def __init__(
        self,
        input_path: Union[str, Path] = "data/processed/lap_dataset",
        *,
        store: Optional[DataStore] = None,
    ) -> None:
        self._configured = str(input_path)
        self._store = store or DataStore.local()
        self._root = self._store.root(Layer.CONSOLIDATED, self._configured)

    @property
    def input_path(self) -> Path:
        """The configured root, as a path. Prefer :attr:`location` for display."""
        return Path(self._configured)

    @property
    def location(self) -> str:
        return self._store.backend.uri(self._root)

    def discover(
        self,
        years: Optional[List[int]] = None,
        session_keys: Optional[List[int]] = None,
    ) -> List[SessionFile]:
        """List consolidated sessions, in a stable chronological order."""
        backend, layout = self._store.backend, self._store.layout
        season = years[0] if years and len(years) == 1 else None

        found: List[SessionFile] = []
        for key in backend.list(layout.session_dataset_listing_prefix(self._root, season)):
            parsed = layout.parse_session_dataset_key(self._root, key)
            if parsed is None:
                continue
            year, session_key = parsed
            if years and year not in years:
                continue
            if session_keys and session_key not in session_keys:
                continue
            found.append(SessionFile(key=key, session_key=session_key, year=year, uri=backend.uri(key)))

        return sorted(found, key=lambda item: item.sort_key)

    def load(self, session_file: SessionFile) -> pd.DataFrame:
        """Read one session and add the columns M5 needs for ordering."""
        frame = self._store.backend.read_parquet(session_file.key)
        return add_session_date(frame)


def add_session_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Stamp every row with when its session started.

    Used to order races chronologically for a temporal split later on. It is
    traceability, not a predictor: the catalogue keeps it out of the feature
    set, since the calendar date of a race says nothing causal about a lap.
    """
    frame = frame.copy()
    if "lap_start_time" in frame.columns and frame["lap_start_time"].notna().any():
        starts = frame.groupby("session_key")["lap_start_time"].transform("min")
        frame["session_date"] = starts
    else:
        frame["session_date"] = pd.NaT
    return frame
