"""Reading the consolidated lap dataset produced by M4.

Sessions are read one file at a time. The consolidated dataset is small
compared with the raw telemetry it came from — a race is roughly a
thousand rows — but per-session reading keeps the output partitioned the
same way and lets a full history be processed race by race.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


@dataclass(frozen=True)
class SessionFile:
    """One consolidated session on disk."""

    path: Path
    session_key: int
    year: Optional[int] = None

    @property
    def sort_key(self) -> tuple:
        return (self.year if self.year is not None else -1, self.session_key)


class LapDatasetLoader:
    """Finds and reads the Parquet files M4 wrote."""

    def __init__(self, input_path: Union[str, Path] = "data/processed/lap_dataset") -> None:
        self._input_path = Path(input_path)

    @property
    def input_path(self) -> Path:
        return self._input_path

    def discover(
        self,
        years: Optional[List[int]] = None,
        session_keys: Optional[List[int]] = None,
    ) -> List[SessionFile]:
        """List consolidated sessions, in a stable chronological order."""
        if not self._input_path.is_dir():
            return []

        found: List[SessionFile] = []
        for path in self._input_path.rglob("session_*.parquet"):
            session_key = _session_key_from_name(path.name)
            if session_key is None:
                continue
            year = _year_from_parent(path.parent.name)
            if years and year not in years:
                continue
            if session_keys and session_key not in session_keys:
                continue
            found.append(SessionFile(path=path, session_key=session_key, year=year))

        return sorted(found, key=lambda item: item.sort_key)

    def load(self, session_file: SessionFile) -> pd.DataFrame:
        """Read one session and add the columns M5 needs for ordering."""
        frame = pd.read_parquet(session_file.path, engine="pyarrow")
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


def _session_key_from_name(file_name: str) -> Optional[int]:
    stem = Path(file_name).stem
    prefix, separator, value = stem.partition("_")
    if prefix != "session" or not separator:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _year_from_parent(directory_name: str) -> Optional[int]:
    try:
        return int(directory_name)
    except ValueError:
        return None
