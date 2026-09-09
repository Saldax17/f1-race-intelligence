"""Building the prediction target.

``next_lap_time`` is ``lap_duration(t+1)`` for the same driver in the same
session. The shift is the only place in M5 that is allowed to look forward,
and it is confined to the target: no feature may derive from it.

Two boundaries are enforced rather than assumed:

* the shift never crosses a driver or a session, and
* it only pairs laps that are genuinely consecutive. M4 measured lap
  numbers to be contiguous, but a hole in the sequence would otherwise
  silently pair lap 5 with lap 7 and produce a target that never existed.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from f1_race_intelligence.features.selection import GROUP_KEYS, ORDER_KEY, TARGET_COLUMN


def add_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Add ``next_lap_time`` and ``has_target``.

    Returns the frame and a breakdown of why rows lack a target, so the
    report can account for every row instead of just counting survivors.
    """
    ordered = frame.sort_values(list(GROUP_KEYS) + [ORDER_KEY], kind="stable").copy()
    grouped = ordered.groupby(list(GROUP_KEYS), sort=False)

    candidate = grouped["lap_duration"].shift(-1)
    next_lap_number = grouped[ORDER_KEY].shift(-1)

    # Only a genuinely consecutive lap can supply the target.
    contiguous = next_lap_number == ordered[ORDER_KEY] + 1
    ordered[TARGET_COLUMN] = candidate.where(contiguous)
    ordered["has_target"] = ordered[TARGET_COLUMN].notna()

    is_last = ordered.get("is_last_lap_for_driver")
    last_lap_rows = int(is_last.fillna(False).sum()) if is_last is not None else 0

    missing = ~ordered["has_target"]
    non_contiguous = int((missing & next_lap_number.notna() & ~contiguous).sum())
    next_duration_missing = int((missing & contiguous & candidate.isna()).sum())

    breakdown = {
        "rows": int(len(ordered)),
        "rows_with_target": int(ordered["has_target"].sum()),
        "rows_without_target": int(missing.sum()),
        "without_target_last_lap": int((missing & next_lap_number.isna()).sum()),
        "without_target_next_lap_duration_missing": next_duration_missing,
        "without_target_lap_sequence_gap": non_contiguous,
        "last_lap_rows": last_lap_rows,
    }
    return ordered, breakdown
