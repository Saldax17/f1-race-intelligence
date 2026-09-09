"""The feature catalogue: what goes into the model, and why.

This module is the single source of truth for every column decision. It is
data rather than logic on purpose — arguing about whether a column belongs
in the model is a matter of evidence, and the evidence should sit next to
the decision instead of being buried in a transformation.

Four roles decide inclusion:

* ``FEATURE`` — safe: describes lap ``t``, which is complete at the moment
  the prediction for ``t+1`` is made.
* ``REVIEW`` — included, but with a caveat worth reading before modelling
  (heavy nulls, redundancy, near-zero variance in the data seen so far).
* ``LEAKY`` — excluded because it carries information from after ``t``.
  These are also listed in :data:`FORBIDDEN_AS_FEATURES`, which the
  anti-leakage checks enforce so a later edit cannot quietly re-add them.
* ``EXCLUDED`` — excluded for reasons other than leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Tuple


class FeatureRole(str, Enum):
    FEATURE = "feature"
    REVIEW = "review"
    LEAKY = "leaky"
    EXCLUDED = "excluded"
    IDENTIFIER = "identifier"
    TARGET = "target"
    METADATA = "metadata"


class FeatureKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class ColumnDecision:
    """One column and the reason it is treated the way it is."""

    name: str
    role: FeatureRole
    reason: str
    kind: FeatureKind = FeatureKind.NUMERIC
    derived: bool = False

    @property
    def is_feature(self) -> bool:
        return self.role in (FeatureRole.FEATURE, FeatureRole.REVIEW)

    def to_dict(self) -> Dict[str, object]:
        return {
            "column": self.name,
            "role": self.role.value,
            "kind": self.kind.value,
            "derived": self.derived,
            "reason": self.reason,
        }


TARGET_COLUMN = "next_lap_time"
GROUP_KEYS: Tuple[str, ...] = ("session_key", "driver_number")
ORDER_KEY = "lap_number"
GRAIN: Tuple[str, ...] = ("session_key", "driver_number", "lap_number")

TRACEABILITY: Tuple[str, ...] = (
    "year",
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "session_date",
)

# Columns that must never reach the model. The anti-leakage checks read this
# set directly, so adding a column to the catalogue as a feature is not
# enough to get it past them.
FORBIDDEN_AS_FEATURES: FrozenSet[str] = frozenset(
    {
        "pit_duration",
        "pit_lane_duration",
        "lap_end_time",
        "is_last_lap_for_driver",
        "has_complete_window",
        "next_lap_time",
        "has_target",
    }
)


def _feature(name: str, reason: str, kind: FeatureKind = FeatureKind.NUMERIC, derived: bool = False) -> ColumnDecision:
    return ColumnDecision(name, FeatureRole.FEATURE, reason, kind, derived)


def _review(name: str, reason: str, kind: FeatureKind = FeatureKind.NUMERIC) -> ColumnDecision:
    return ColumnDecision(name, FeatureRole.REVIEW, reason, kind)


def _leaky(name: str, reason: str, kind: FeatureKind = FeatureKind.NUMERIC) -> ColumnDecision:
    return ColumnDecision(name, FeatureRole.LEAKY, reason, kind)


def _excluded(name: str, reason: str, kind: FeatureKind = FeatureKind.NUMERIC) -> ColumnDecision:
    return ColumnDecision(name, FeatureRole.EXCLUDED, reason, kind)


CATALOGUE: Tuple[ColumnDecision, ...] = (
    # -- identifiers, kept so a row can be traced back to M4 ---------------
    ColumnDecision("year", FeatureRole.IDENTIFIER, "Traceability. Not a feature: with few races it identifies the race."),
    ColumnDecision("meeting_key", FeatureRole.IDENTIFIER, "Traceability only."),
    ColumnDecision("session_key", FeatureRole.IDENTIFIER, "Traceability and grouping key. Not a feature."),
    ColumnDecision(
        "driver_number",
        FeatureRole.IDENTIFIER,
        "Traceability and grouping key. Driver identity enters the model as driver_acronym instead, "
        "because the magnitude of a car number means nothing.",
    ),
    ColumnDecision(
        "session_date",
        FeatureRole.IDENTIFIER,
        "The session's start instant. Lets a later stage order races chronologically for a temporal "
        "split. Deliberately not a feature.",
        FeatureKind.TEMPORAL,
        derived=True,
    ),
    # -- lap itself --------------------------------------------------------
    _feature("lap_number", "Fuel burns off as the race goes on, so the car gets lighter and faster."),
    _feature("lap_duration", "The driver's current pace: the base predictor of the next lap."),
    _feature("duration_sector_1", "Shows where in the lap the time is being gained or lost."),
    _feature("duration_sector_2", "Shows where in the lap the time is being gained or lost."),
    _feature("duration_sector_3", "Includes the pit entry, so it reacts when a driver commits to a stop."),
    _feature("is_pit_out_lap", "Lap t started from the pit lane on cold tyres.", FeatureKind.BOOLEAN),
    # -- tyres -------------------------------------------------------------
    _feature("stint_number", "How many sets of tyres the driver has already used."),
    _feature("compound", "The compound sets the pace envelope of the lap.", FeatureKind.CATEGORICAL),
    _feature("tyre_age", "Tyre degradation is one of the strongest drivers of lap time."),
    _feature("stint_lap", "Laps completed on the current set."),
    # -- pit ---------------------------------------------------------------
    _feature(
        "pit_in_lap",
        "The car entered the pit lane before crossing the line that closes lap t, so this is observable "
        "at prediction time. It is the legitimate signal that lap t+1 will be an out-lap. Not to be "
        "confused with pit_duration, which is measured afterwards.",
        FeatureKind.BOOLEAN,
    ),
    # -- position and gaps -------------------------------------------------
    _feature("position_at_lap_start", "Track position governs how much traffic the driver meets."),
    _feature("position_at_lap_end", "Position once lap t is complete."),
    _feature("position_changes_in_lap", "Overtaking, and being overtaken, costs time."),
    _feature("gap_to_leader_s", "Relative pace, measured by OpenF1 within lap t."),
    _feature("interval_s", "Distance to the car ahead: the proxy for dirty air."),
    _feature("is_lapped", "A lapped car races differently and has no numeric gap.", FeatureKind.BOOLEAN),
    # -- telemetry ---------------------------------------------------------
    _feature("speed_min", "Slowest point of the lap: reflects traffic and corner entry."),
    _feature("speed_mean", "Average speed over lap t."),
    _feature("speed_max", "Top speed reached on lap t."),
    _feature("rpm_min", "Engine usage across lap t."),
    _feature("rpm_mean", "Engine usage across lap t."),
    _feature("rpm_max", "Engine usage across lap t."),
    _feature("throttle_mean", "How much of the lap was spent on power."),
    _feature("brake_applied_share", "Share of the lap under braking."),
    _feature("drs_active_share", "DRS use indicates following another car on a straight."),
    # -- weather -----------------------------------------------------------
    _feature("air_temperature", "Ambient conditions affect engine and tyre behaviour."),
    _feature("track_temperature", "Track temperature drives tyre degradation."),
    _feature("wind_speed", "Wind changes drag and stability."),
    # -- race control ------------------------------------------------------
    _feature("rc_events_in_lap", "Safety cars and flags slow the whole field."),
    _feature("rc_driver_events_in_lap", "Events aimed at this driver, such as a penalty or a blue flag."),
    # -- identity ----------------------------------------------------------
    _feature("driver_acronym", "Driver identity: skill and style differ.", FeatureKind.CATEGORICAL),
    _feature("team_name", "Car performance differs between teams.", FeatureKind.CATEGORICAL),
    # -- derived temporal features ----------------------------------------
    _feature("lap_time_prev_1", "The lap before the current one. Null on lap 1.", derived=True),
    _feature("lap_time_delta_1", "Whether the driver is speeding up or slowing down. Null on lap 1.", derived=True),
    _feature("lap_time_roll_mean_3", "Recent pace over 3 laps ending at t.", derived=True),
    _feature("lap_time_roll_mean_5", "Recent pace over 5 laps ending at t.", derived=True),
    _feature("lap_time_roll_std_3", "Consistency over 3 laps. Needs 2 laps, so null on lap 1.", derived=True),
    _feature("lap_time_roll_std_5", "Consistency over 5 laps. Needs 2 laps, so null on lap 1.", derived=True),
    _feature("lap_time_roll_min_3", "Best of the last 3 laps: the pace the driver can reach.", derived=True),
    _feature("lap_time_roll_min_5", "Best of the last 5 laps.", derived=True),
    _feature(
        "lap_time_vs_roll_mean_5",
        "Current lap against the driver's own recent average. Absolute lap times differ between "
        "circuits and seasons; this deviation transfers.",
        derived=True,
    ),
    _feature("lap_time_expanding_mean", "The driver's average pace so far in this race.", derived=True),
    # -- included, with a caveat ------------------------------------------
    _review("i1_speed", "Speed trap 1. Around a quarter of laps have no reading; the gap is legitimate."),
    _review("i2_speed", "Speed trap 2. Nearly complete."),
    _review("st_speed", "Speed trap on the straight. About 15% of laps have no reading."),
    _review("humidity", "Almost constant within a race; with few races it behaves like a race identifier."),
    _review("pressure", "Almost constant within a race; same caveat as humidity."),
    _review("n_gear_max", "Only two distinct values observed (7 and 8), so it carries little information."),
    _review("car_data_samples", "Telemetry sample count is proportional to lap duration, so it largely repeats it."),
    _review("interval_samples", "Same redundancy as car_data_samples."),
    # -- leakage -----------------------------------------------------------
    _leaky(
        "pit_duration",
        "Measured after lap t ends. Of 43 stops, none had its timestamp inside lap t; the median was "
        "23.1 s after it. The value correlates 0.989 with the excess time of lap t+1, so it is close to "
        "a direct measurement of the target.",
    ),
    _leaky("pit_lane_duration", "Same timing as pit_duration: the pit lane transit happens during lap t+1."),
    _leaky("lap_end_time", "It is literally date_start(t+1).", FeatureKind.TEMPORAL),
    _leaky("is_last_lap_for_driver", "Only knowable by checking whether lap t+1 exists.", FeatureKind.BOOLEAN),
    _leaky("has_complete_window", "Derived from the existence of lap t+1.", FeatureKind.BOOLEAN),
    # -- excluded for other reasons ---------------------------------------
    _excluded("lap_start_time", "Absolute timestamp; kept upstream for traceability, meaningless as a level.", FeatureKind.TEMPORAL),
    _excluded(
        "rainfall",
        "Zero variance in the races available (always 0), so it cannot inform the model today. It stays "
        "in M4 and should be reconsidered once the full history is extracted and wet races appear.",
    ),
    _excluded("weather_age_s", "Describes how our own pipeline matched a weather reading, not the car."),
    _excluded("car_data_invalid_samples", "A data quality counter, not a physical quantity."),
    _excluded(
        "wind_direction",
        "A circular quantity: 0 and 359 degrees are adjacent, so the raw degree value is misleading. A "
        "sine/cosine encoding is not justified by evidence yet.",
    ),
    # -- target and metadata ----------------------------------------------
    ColumnDecision(
        TARGET_COLUMN,
        FeatureRole.TARGET,
        "lap_duration(t+1) for the same driver in the same session.",
        derived=True,
    ),
    ColumnDecision(
        "has_target",
        FeatureRole.METADATA,
        "Whether this row has a valid target. False on a driver's last lap; those rows are kept rather "
        "than dropped, so nothing disappears silently.",
        FeatureKind.BOOLEAN,
        derived=True,
    ),
)

BY_NAME: Dict[str, ColumnDecision] = {item.name: item for item in CATALOGUE}


def feature_columns() -> List[str]:
    """Columns that go to the model, in catalogue order."""
    return [item.name for item in CATALOGUE if item.is_feature]


def categorical_columns() -> List[str]:
    return [item.name for item in CATALOGUE if item.is_feature and item.kind is FeatureKind.CATEGORICAL]


def derived_columns() -> List[str]:
    return [item.name for item in CATALOGUE if item.derived]


def excluded_columns() -> List[ColumnDecision]:
    return [item for item in CATALOGUE if item.role in (FeatureRole.LEAKY, FeatureRole.EXCLUDED)]


def output_columns() -> List[str]:
    """Traceability first, then features, then the target and its flag."""
    ordered: List[str] = list(TRACEABILITY)
    for name in feature_columns():
        if name not in ordered:
            ordered.append(name)
    ordered.extend([TARGET_COLUMN, "has_target"])
    return ordered


def catalogue_conflicts() -> List[str]:
    """Names that are both marked as features and forbidden.

    An empty list is the invariant the anti-leakage checks rely on: it means
    nobody has promoted a leaky column to a feature by editing the catalogue.
    """
    return sorted(set(feature_columns()) & FORBIDDEN_AS_FEATURES)
