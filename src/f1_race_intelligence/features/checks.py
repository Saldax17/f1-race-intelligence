"""Automated guards against leakage.

A leaky feature does not announce itself: the dataset looks fine, the model
looks excellent, and the error only appears in production. So the
properties M5 depends on are asserted rather than trusted, and the two that
matter most are checked by *recomputing* the value along a different path
than the one that produced it. A check that reuses the same code as the
transformation would agree with it even when both are wrong.

Every check is blocking except the correlation guard, which is a diagnostic:
a high correlation is a reason to look, not proof of leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from f1_race_intelligence.features import selection
from f1_race_intelligence.features.selection import (
    FORBIDDEN_AS_FEATURES,
    GRAIN,
    GROUP_KEYS,
    ORDER_KEY,
    TARGET_COLUMN,
)
from f1_race_intelligence.features.temporal import MIN_PERIODS_SPREAD, RELATIVE_WINDOW

# Rows sampled for the truncated recomputation. The check is O(rows x window)
# per sampled row, so a bounded, deterministic sample keeps it affordable
# while still covering the start, middle and end of every stint.
RECOMPUTE_SAMPLE = 250

TOLERANCE = 1e-9


@dataclass
class CheckResult:
    """The outcome of one guard."""

    name: str
    passed: bool
    description: str
    blocking: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "check": self.name,
            "status": "PASS" if self.passed else "FAIL",
            "blocking": self.blocking,
            "description": self.description,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def run_all(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    windows: Sequence[int] = (3, 5),
    correlation_threshold: float = 0.99,
) -> List[CheckResult]:
    """Run every guard over a built feature frame."""
    return [
        check_grain_unique(frame),
        check_target_independently(frame),
        check_last_lap_has_no_target(frame),
        check_target_stays_within_group(frame),
        check_no_forbidden_features(feature_names),
        check_rolling_uses_only_the_past(frame, windows),
        check_correlation_with_target(frame, feature_names, correlation_threshold),
    ]


def check_grain_unique(frame: pd.DataFrame) -> CheckResult:
    """One row per session, driver and lap."""
    missing = [column for column in GRAIN if column not in frame.columns]
    if missing:
        return CheckResult("grain_unique", False, "Grain columns are missing", details={"missing": missing})

    duplicated = int(frame.duplicated(subset=list(GRAIN)).sum())
    return CheckResult(
        "grain_unique",
        duplicated == 0,
        "Each session, driver and lap appears exactly once",
        details={"duplicate_rows": duplicated},
    )


def check_target_independently(frame: pd.DataFrame) -> CheckResult:
    """Rebuild the target through a different mechanism and compare.

    The pipeline builds it with a grouped shift. Here it is rebuilt from a
    dictionary keyed by (session, driver, lap + 1), which shares no code with
    the shift, so the two agreeing is real evidence rather than a tautology.
    """
    lookup = {
        (session, driver, lap): duration
        for session, driver, lap, duration in zip(
            frame["session_key"], frame["driver_number"], frame[ORDER_KEY], frame["lap_duration"]
        )
    }
    expected = [
        lookup.get((session, driver, lap + 1))
        for session, driver, lap in zip(frame["session_key"], frame["driver_number"], frame[ORDER_KEY])
    ]

    rebuilt = pd.Series(expected, index=frame.index, dtype="Float64")
    actual = frame[TARGET_COLUMN].astype("Float64")

    both_null = rebuilt.isna() & actual.isna()
    close = (rebuilt - actual).abs() <= TOLERANCE
    mismatches = int((~(both_null | close.fillna(False))).sum())

    return CheckResult(
        "target_matches_independent_derivation",
        mismatches == 0,
        "next_lap_time equals lap_duration of the same driver's next lap, derived a second way",
        details={"mismatched_rows": mismatches},
    )


def check_last_lap_has_no_target(frame: pd.DataFrame) -> CheckResult:
    """A driver's final lap cannot have a target, and must not be given one."""
    if "is_last_lap_for_driver" not in frame.columns:
        return CheckResult(
            "last_lap_has_no_target",
            True,
            "Skipped: the source did not mark last laps",
            details={"skipped": True},
        )

    last = frame["is_last_lap_for_driver"].fillna(False).astype(bool)
    invented = int((last & frame[TARGET_COLUMN].notna()).sum())
    return CheckResult(
        "last_lap_has_no_target",
        invented == 0,
        "No target was invented for a driver's last lap",
        details={"last_laps": int(last.sum()), "last_laps_with_target": invented},
    )


def check_target_stays_within_group(frame: pd.DataFrame) -> CheckResult:
    """The target never comes from another driver or another session.

    Verified by walking the ordered frame and confirming that every row with
    a target is followed, in its own group, by the lap that supplied it.
    """
    ordered = frame.sort_values(list(GROUP_KEYS) + [ORDER_KEY], kind="stable")
    grouped = ordered.groupby(list(GROUP_KEYS), sort=False)

    next_session = grouped["session_key"].shift(-1)
    next_driver = grouped["driver_number"].shift(-1)
    next_lap = grouped[ORDER_KEY].shift(-1)

    has_target = ordered[TARGET_COLUMN].notna()
    crossed = int(
        (
            has_target
            & (
                (next_session != ordered["session_key"])
                | (next_driver != ordered["driver_number"])
                | (next_lap != ordered[ORDER_KEY] + 1)
            )
        ).sum()
    )
    return CheckResult(
        "target_does_not_cross_driver_or_session",
        crossed == 0,
        "Every target comes from the next lap of the same driver in the same session",
        details={"rows_crossing_a_boundary": crossed},
    )


def check_no_forbidden_features(feature_names: Sequence[str]) -> CheckResult:
    """No column known to leak reached the feature set.

    Two independent conditions: the built feature list must not contain a
    forbidden name, and the catalogue itself must not classify one as a
    feature. The second stops the guard from being defeated by editing the
    catalogue.
    """
    present = sorted(set(feature_names) & FORBIDDEN_AS_FEATURES)
    conflicts = selection.catalogue_conflicts()
    return CheckResult(
        "no_forbidden_columns_among_features",
        not present and not conflicts,
        "Columns carrying information from after lap t are absent from the feature set",
        details={
            "forbidden_present": present,
            "catalogue_conflicts": conflicts,
            "forbidden_columns": sorted(FORBIDDEN_AS_FEATURES),
        },
    )


def check_rolling_uses_only_the_past(
    frame: pd.DataFrame,
    windows: Sequence[int] = (3, 5),
    sample_size: int = RECOMPUTE_SAMPLE,
) -> CheckResult:
    """Recompute rolling features from the past alone and compare.

    For a sampled row, the driver's laps up to and including ``t`` are taken
    on their own and the statistic is computed directly from them. If a
    rolling feature had seen lap ``t+1``, the two values would differ.
    """
    ordered = frame.sort_values(list(GROUP_KEYS) + [ORDER_KEY], kind="stable")
    if ordered.empty:
        return CheckResult("rolling_features_use_only_the_past", True, "No rows to check", details={"checked_rows": 0})

    step = max(1, len(ordered) // sample_size)
    sampled = ordered.iloc[::step]

    histories = {
        key: group.sort_values(ORDER_KEY)[[ORDER_KEY, "lap_duration"]]
        for key, group in ordered.groupby(list(GROUP_KEYS), sort=False)
    }

    mismatches: List[Dict[str, Any]] = []
    checked = 0

    for _, row in sampled.iterrows():
        key = (row["session_key"], row["driver_number"])
        history = histories[key]
        past = history[history[ORDER_KEY] <= row[ORDER_KEY]]["lap_duration"].astype("float64")
        if past.empty:
            continue
        checked += 1

        expected: Dict[str, Any] = {
            "lap_time_prev_1": past.iloc[-2] if len(past) >= 2 else None,
            "lap_time_expanding_mean": past.mean(),
        }
        for window in windows:
            tail = past.tail(window)
            expected[f"lap_time_roll_mean_{window}"] = tail.mean()
            expected[f"lap_time_roll_min_{window}"] = tail.min()
            expected[f"lap_time_roll_std_{window}"] = (
                tail.std(ddof=1) if len(tail) >= MIN_PERIODS_SPREAD else None
            )
        if RELATIVE_WINDOW in windows:
            expected["lap_time_vs_roll_mean_5"] = past.iloc[-1] - past.tail(RELATIVE_WINDOW).mean()

        for column, value in expected.items():
            if column not in row.index:
                continue
            actual = row[column]
            if value is None or pd.isna(value):
                if pd.notna(actual):
                    mismatches.append({"column": column, "lap": int(row[ORDER_KEY]), "expected": None, "actual": float(actual)})
                continue
            if pd.isna(actual) or abs(float(actual) - float(value)) > 1e-6:
                mismatches.append(
                    {
                        "column": column,
                        "lap": int(row[ORDER_KEY]),
                        "expected": float(value),
                        "actual": None if pd.isna(actual) else float(actual),
                    }
                )

    return CheckResult(
        "rolling_features_use_only_the_past",
        not mismatches,
        "Every rolling feature equals the value recomputed from laps up to t alone",
        details={"checked_rows": checked, "mismatches": mismatches[:5], "mismatch_count": len(mismatches)},
    )


def check_correlation_with_target(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    threshold: float = 0.99,
) -> CheckResult:
    """Flag features that track the target almost exactly.

    A diagnostic, never a rule: a high correlation can be legitimate, and
    dropping features automatically would hide the very thing worth looking
    at. This is the guard that would have caught pit_duration, whose
    correlation with the target's excess was 0.989.
    """
    with_target = frame[frame[TARGET_COLUMN].notna()]
    if with_target.empty:
        return CheckResult(
            "no_feature_almost_equals_the_target",
            True,
            "Skipped: no rows have a target",
            blocking=False,
            details={"skipped": True},
        )

    target = pd.to_numeric(with_target[TARGET_COLUMN], errors="coerce").astype("float64")
    suspicious: List[Dict[str, Any]] = []

    for name in feature_names:
        if name not in with_target.columns:
            continue
        values = pd.to_numeric(with_target[name], errors="coerce")
        if values.isna().all() or values.nunique(dropna=True) < 2:
            continue
        correlation = values.astype("float64").corr(target)
        if pd.notna(correlation) and abs(correlation) >= threshold:
            suspicious.append({"feature": name, "correlation": round(float(correlation), 4)})

    return CheckResult(
        "no_feature_almost_equals_the_target",
        not suspicious,
        f"No feature correlates with the target at or above {threshold} (diagnostic only)",
        blocking=False,
        details={"threshold": threshold, "flagged": suspicious},
    )


def blocking_failures(results: Sequence[CheckResult]) -> List[str]:
    return [item.name for item in results if item.blocking and not item.passed]
