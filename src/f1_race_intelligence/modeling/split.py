"""Splitting the dataset in time, by whole races.

Two rules decide everything here.

The split is **chronological**: a model that will predict future races must
be validated on races that came after the ones it learned from. A random
split would let it learn from a race it is later scored on, which flatters
the metric and tells us nothing about the future.

The split is **by session, never by row**. Laps of the same race share a
track, a weather pattern, a safety car and a tyre allocation, so putting
some laps of a race in train and others in validation leaks the answer
through the circumstances. A session therefore belongs to exactly one
split, with all of its drivers and laps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)


class SplitStrategy(str, Enum):
    """How sessions are handed to the splits."""

    FRACTION = "fraction"
    """Order races in time and cut by proportion. Adapts to any history size."""

    YEARS = "years"
    """Assign whole seasons explicitly. Clearer once several seasons exist."""

    TRAIN_YEARS_THEN_SPLIT = "train_years_then_split"
    """Whole seasons to train, then cut what remains into validation and test.

    The shape an academic evaluation usually wants: learn from the closed
    seasons, then hold out the most recent one, split in two so that test
    sits strictly after validation. Neither of the other strategies can
    express it — ``years`` cannot divide a season, and ``fraction`` ignores
    season boundaries.
    """


@dataclass(frozen=True)
class SessionAssignment:
    """One race and the split it belongs to."""

    session_key: int
    year: Optional[int]
    session_date: Optional[pd.Timestamp]
    split: str
    rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": int(self.session_key),
            "year": int(self.year) if self.year is not None else None,
            "session_date": self.session_date.isoformat() if self.session_date is not None else None,
            "split": self.split,
            "rows": self.rows,
        }


def session_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per race, ordered in time.

    Sorted by date and then by key: the key breaks ties so that two races on
    the same day always land in the same order, which keeps the split
    reproducible.
    """
    calendar = (
        frame.groupby("session_key", as_index=False)
        .agg(year=("year", "first"), session_date=("session_date", "min"), rows=("lap_number", "size"))
        .sort_values(["session_date", "session_key"], kind="stable")
        .reset_index(drop=True)
    )
    return calendar


def assign_sessions(
    frame: pd.DataFrame,
    strategy: SplitStrategy = SplitStrategy.FRACTION,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    train_years: Optional[Sequence[int]] = None,
    validation_years: Optional[Sequence[int]] = None,
    test_years: Optional[Sequence[int]] = None,
    validation_share_of_remainder: float = 0.5,
) -> List[SessionAssignment]:
    """Decide which split each race belongs to."""
    calendar = session_calendar(frame)
    if calendar.empty:
        return []

    if strategy is SplitStrategy.YEARS:
        return _assign_by_years(calendar, train_years or [], validation_years or [], test_years or [])
    if strategy is SplitStrategy.TRAIN_YEARS_THEN_SPLIT:
        return _assign_train_years_then_split(calendar, train_years or [], validation_share_of_remainder)
    return _assign_by_fraction(calendar, train_fraction, validation_fraction)


def _assign_train_years_then_split(
    calendar: pd.DataFrame,
    train_years: Sequence[int],
    validation_share: float,
) -> List[SessionAssignment]:
    """Whole seasons to train; cut the rest chronologically in two.

    The remaining races keep their calendar order, so validation holds the
    earlier half of the season and test the later one. With an odd number
    of races the extra one goes to validation, which is stated here rather
    than left to whichever way a floating point rounds.
    """
    if not train_years:
        raise ValueError("The 'train_years_then_split' strategy needs train_years to be set")

    wanted = set(train_years)
    is_train = calendar["year"].isin(wanted)
    if not is_train.any():
        raise ValueError(f"No session belongs to the seasons named for training: {sorted(wanted)}")

    remainder = calendar[~is_train].reset_index(drop=True)
    # Rounded up on purpose, so an odd race count leaves the extra race in
    # validation as documented. Python's round() would decide it by banker's
    # rounding — round(2.5) is 2 — which is not a rule anyone reading the
    # configuration would predict.
    cut = max(0, min(math.ceil(len(remainder) * validation_share), len(remainder)))

    assignments: List[SessionAssignment] = []
    for row in calendar[is_train].itertuples(index=False):
        assignments.append(_assignment(row, TRAIN))
    for position, row in enumerate(remainder.itertuples(index=False)):
        assignments.append(_assignment(row, VALIDATION if position < cut else TEST))

    return sorted(assignments, key=lambda item: (item.session_date is None, item.session_date, item.session_key))


def _assign_by_fraction(
    calendar: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> List[SessionAssignment]:
    """Cut the chronological list of races at two boundaries."""
    total = len(calendar)
    train_end = int(round(total * train_fraction))
    validation_end = int(round(total * (train_fraction + validation_fraction)))

    # With a handful of races the rounding can leave train empty, which is
    # never what the caller wants; the later splits are allowed to be empty
    # and the report says so.
    train_end = max(1, min(train_end, total))
    validation_end = max(train_end, min(validation_end, total))

    assignments: List[SessionAssignment] = []
    for position, row in enumerate(calendar.itertuples(index=False)):
        if position < train_end:
            split = TRAIN
        elif position < validation_end:
            split = VALIDATION
        else:
            split = TEST
        assignments.append(_assignment(row, split))
    return assignments


def _assign_by_years(
    calendar: pd.DataFrame,
    train_years: Sequence[int],
    validation_years: Sequence[int],
    test_years: Sequence[int],
) -> List[SessionAssignment]:
    """Assign whole seasons. A season named twice is a configuration error."""
    overlap = (set(train_years) & set(validation_years)) | (set(train_years) & set(test_years)) | (
        set(validation_years) & set(test_years)
    )
    if overlap:
        raise ValueError(f"Seasons assigned to more than one split: {sorted(overlap)}")

    lookup = {}
    for year in train_years:
        lookup[year] = TRAIN
    for year in validation_years:
        lookup[year] = VALIDATION
    for year in test_years:
        lookup[year] = TEST

    assignments: List[SessionAssignment] = []
    unassigned: List[int] = []
    for row in calendar.itertuples(index=False):
        split = lookup.get(row.year)
        if split is None:
            unassigned.append(int(row.session_key))
            continue
        assignments.append(_assignment(row, split))

    if unassigned:
        raise ValueError(
            f"{len(unassigned)} sessions belong to seasons not named in the split configuration: "
            f"{sorted({int(a) for a in unassigned})[:5]}"
        )
    return assignments


def _assignment(row: Any, split: str) -> SessionAssignment:
    date = row.session_date
    return SessionAssignment(
        session_key=int(row.session_key),
        year=int(row.year) if pd.notna(row.year) else None,
        session_date=date if pd.notna(date) else None,
        split=split,
        rows=int(row.rows),
    )


def apply_assignments(
    frame: pd.DataFrame,
    assignments: Sequence[SessionAssignment],
) -> Dict[str, pd.DataFrame]:
    """Cut the dataset into the three splits, keeping row order stable."""
    by_session = {item.session_key: item.split for item in assignments}
    labelled = frame.assign(_split=frame["session_key"].map(by_session))

    return {
        name: labelled[labelled["_split"] == name]
        .drop(columns=["_split"])
        .sort_values(["session_key", "driver_number", "lap_number"], kind="stable")
        .reset_index(drop=True)
        for name in SPLITS
    }


def describe(assignments: Sequence[SessionAssignment], splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """The split summary that goes into the report."""
    summary: Dict[str, Any] = {}
    for name in SPLITS:
        chosen = [item for item in assignments if item.split == name]
        dates = [item.session_date for item in chosen if item.session_date is not None]
        summary[name] = {
            "sessions": len(chosen),
            "session_keys": sorted(item.session_key for item in chosen),
            "years": sorted({item.year for item in chosen if item.year is not None}),
            "rows": int(len(splits[name])),
            "first_session_date": min(dates).isoformat() if dates else None,
            "last_session_date": max(dates).isoformat() if dates else None,
        }
    return summary
