"""History features: what the driver had done up to and including lap t.

Every window is computed inside ``(session_key, driver_number)`` ordered by
``lap_number``, and every one of them is trailing: it covers laps ``t-k+1``
through ``t`` and never touches ``t+1``. Lap ``t`` is included because it is
complete at the moment the prediction is made.

Behaviour at the start of a race, which is a deliberate choice and not an
accident of the implementation:

* lap 1 — ``lap_time_prev_1`` and ``lap_time_delta_1`` are null (there is no
  earlier lap), the rolling means and minima equal the lap's own duration,
  the rolling standard deviations are null (a spread needs two values), and
  ``lap_time_vs_roll_mean_5`` is zero.
* laps 2 to 4 — the windows use however many laps exist so far rather than
  waiting to fill; a shorter window is still an honest summary of the past.

Nothing is imputed. A null here means "this could not be computed from the
past", which is information a model should see rather than have hidden.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

from f1_race_intelligence.features.selection import GROUP_KEYS, ORDER_KEY

# A standard deviation needs at least two observations to mean anything.
MIN_PERIODS_SPREAD = 2
MIN_PERIODS_LEVEL = 1

# The window used for the "pace relative to my own recent pace" feature.
RELATIVE_WINDOW = 5


def add_temporal_features(frame: pd.DataFrame, windows: Sequence[int] = (3, 5)) -> pd.DataFrame:
    """Add lag, rolling and expanding features derived from ``lap_duration``."""
    ordered = frame.sort_values(list(GROUP_KEYS) + [ORDER_KEY], kind="stable").copy()

    # Rolling on the nullable Float64 dtype is avoided deliberately: plain
    # float64 keeps the numerics predictable, and the results are cast back.
    ordered["_lap_duration_f"] = pd.to_numeric(ordered["lap_duration"], errors="coerce").astype("float64")
    grouped = ordered.groupby(list(GROUP_KEYS), sort=False)["_lap_duration_f"]

    ordered["lap_time_prev_1"] = grouped.shift(1)
    ordered["lap_time_delta_1"] = ordered["_lap_duration_f"] - ordered["lap_time_prev_1"]

    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=MIN_PERIODS_LEVEL)
        ordered[f"lap_time_roll_mean_{window}"] = _align(rolling.mean(), ordered)
        ordered[f"lap_time_roll_min_{window}"] = _align(rolling.min(), ordered)

        spread = grouped.rolling(window=window, min_periods=MIN_PERIODS_SPREAD)
        ordered[f"lap_time_roll_std_{window}"] = _align(spread.std(), ordered)

    relative_column = f"lap_time_roll_mean_{RELATIVE_WINDOW}"
    if relative_column in ordered.columns:
        ordered["lap_time_vs_roll_mean_5"] = ordered["_lap_duration_f"] - ordered[relative_column]

    expanding = grouped.expanding(min_periods=MIN_PERIODS_LEVEL).mean()
    ordered["lap_time_expanding_mean"] = _align(expanding, ordered)

    ordered = ordered.drop(columns=["_lap_duration_f"])
    for column in temporal_columns(windows):
        if column in ordered.columns:
            ordered[column] = ordered[column].astype("Float64")
    return ordered


def _align(result: pd.Series, frame: pd.DataFrame) -> pd.Series:
    """Put a grouped rolling result back on the frame's own index."""
    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(level=list(range(result.index.nlevels - 1)), drop=True)
    return result.reindex(frame.index)


def temporal_columns(windows: Sequence[int] = (3, 5)) -> list[str]:
    """Names of everything :func:`add_temporal_features` produces."""
    names = ["lap_time_prev_1", "lap_time_delta_1", "lap_time_expanding_mean"]
    for window in windows:
        names.extend(
            [
                f"lap_time_roll_mean_{window}",
                f"lap_time_roll_min_{window}",
                f"lap_time_roll_std_{window}",
            ]
        )
    if RELATIVE_WINDOW in windows:
        names.append("lap_time_vs_roll_mean_5")
    return names


def window_settings(windows: Sequence[int] = (3, 5)) -> Dict[str, Any]:
    """The parameters that produced these features, for the report."""
    return {
        "windows": list(windows),
        "grouped_by": list(GROUP_KEYS),
        "ordered_by": ORDER_KEY,
        "includes_current_lap": True,
        "min_periods_level": MIN_PERIODS_LEVEL,
        "min_periods_spread": MIN_PERIODS_SPREAD,
        "relative_window": RELATIVE_WINDOW,
        "first_lap_behaviour": (
            "prev/delta and rolling std are null; rolling mean and min equal the lap's own duration; "
            "lap_time_vs_roll_mean_5 is 0."
        ),
    }
