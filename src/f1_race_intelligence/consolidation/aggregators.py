"""Reducing every source to one row per driver and lap.

The whole consolidation rests on one measured fact: for a driver, the laps
of a race tile the clock exactly. Taking each lap's window as
``[date_start(t), date_start(t+1))`` and comparing it against the reported
``lap_duration`` over 2 184 real laps produced zero gaps and zero overlaps
beyond a second. So a high-frequency record belongs to exactly one lap,
and "the last window that started at or before this record" identifies it
without ambiguity.

That is also the leakage rule in operational form: a value describing lap
``t`` comes from inside ``t``'s own window, or from the last thing known
before it started. Never from ``t+1``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

# Physically impossible telemetry, measured in real OpenF1 data. These are
# excluded from the aggregate they would distort — never from the raw file,
# which stays immutable — and counted so the exclusion is auditable.
MAX_GEAR = 8          # F1 gearboxes have 8 forward ratios; values up to 49 appear
MAX_THROTTLE = 100    # documented 0-100; 104 occurs
MAX_BRAKE = 100       # documented 0 or 100; 104 occurs

# DRS state codes that mean the flap is open. 0-3 are closed and 8 means
# "eligible but not open", so neither counts as active.
DRS_ACTIVE_CODES = (10, 12, 14)

LAP_KEYS = ["session_key", "driver_number", "lap_number"]


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse OpenF1 timestamps, keeping them timezone-aware in UTC."""
    return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")


def build_lap_windows(laps: pd.DataFrame) -> pd.DataFrame:
    """Give every lap the time window it occupies.

    A lap ends when the next one starts. The final lap of a driver has no
    next lap, so it falls back to ``lap_duration``; when that is missing too
    the window cannot be closed, and the lap is kept with
    ``has_complete_window`` false rather than dropped — it is a real lap,
    and silently discarding it would bias the dataset towards drivers who
    finished.
    """
    frame = laps.sort_values(["driver_number", "lap_number"], kind="stable").copy()
    frame["lap_start_time"] = to_datetime(frame["date_start"])

    next_start = frame.groupby("driver_number", sort=False)["lap_start_time"].shift(-1)
    fallback = frame["lap_start_time"] + pd.to_timedelta(frame["lap_duration"], unit="s")
    frame["lap_end_time"] = next_start.fillna(fallback)

    frame["is_last_lap_for_driver"] = next_start.isna()
    frame["has_complete_window"] = frame["lap_start_time"].notna() & frame["lap_end_time"].notna()
    return frame


def assign_records_to_laps(
    records: pd.DataFrame,
    windows: pd.DataFrame,
    time_column: str,
    by_driver: bool = True,
) -> pd.DataFrame:
    """Label each high-frequency record with the lap that contains it.

    Uses a backward as-of match on the window starts, which is exact
    precisely because the windows tile without gaps, then drops anything
    past the end of its candidate lap — that is what removes the pre-race
    and post-race telemetry rather than misfiling it into lap 1 or the last
    lap.
    """
    if records.empty or windows.empty:
        return records.iloc[0:0].assign(lap_number=pd.Series(dtype="Int64"))

    usable = windows[windows["has_complete_window"]]
    if usable.empty:
        return records.iloc[0:0].assign(lap_number=pd.Series(dtype="Int64"))

    left = records.dropna(subset=[time_column]).sort_values(time_column, kind="stable")
    right_columns = ["lap_start_time", "lap_end_time", "lap_number"]
    if by_driver:
        right_columns.append("driver_number")
    right = usable[right_columns].sort_values("lap_start_time", kind="stable")

    merged = pd.merge_asof(
        left,
        right,
        left_on=time_column,
        right_on="lap_start_time",
        by="driver_number" if by_driver else None,
        direction="backward",
    )
    merged = merged[merged["lap_number"].notna() & (merged[time_column] < merged["lap_end_time"])]
    return merged


def aggregate_car_data(records: List[Mapping[str, Any]], windows: pd.DataFrame) -> Dict[str, Any]:
    """Summarize one driver's telemetry into one row per lap.

    Returns the per-lap frame together with the counts that make the run
    auditable: how many samples were read, how many fell inside a lap, and
    how many carried a physically impossible value.

    Impossible values are dropped per field, not per sample: a broken gear
    reading does not discredit the speed recorded alongside it.
    """
    empty = {
        "frame": _empty_car_data_frame(),
        "samples_read": len(records),
        "samples_in_window": 0,
        "invalid_samples": 0,
    }
    if not records:
        return empty

    raw = pd.DataFrame.from_records(records)
    for column in ("speed", "rpm", "n_gear", "throttle", "brake", "drs", "driver_number"):
        if column not in raw.columns:
            raw[column] = pd.NA
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["date"] = to_datetime(raw["date"]) if "date" in raw.columns else pd.NaT

    assigned = assign_records_to_laps(raw, windows, "date")
    if assigned.empty:
        return empty

    gear_ok = assigned["n_gear"] <= MAX_GEAR
    throttle_ok = assigned["throttle"] <= MAX_THROTTLE
    brake_ok = assigned["brake"] <= MAX_BRAKE

    assigned = assigned.assign(
        n_gear_valid=assigned["n_gear"].where(gear_ok),
        throttle_valid=assigned["throttle"].where(throttle_ok),
        brake_valid=assigned["brake"].where(brake_ok),
        invalid_sample=(~gear_ok & assigned["n_gear"].notna())
        | (~throttle_ok & assigned["throttle"].notna())
        | (~brake_ok & assigned["brake"].notna()),
    )
    assigned["brake_applied"] = (assigned["brake_valid"] > 0).where(assigned["brake_valid"].notna())
    assigned["drs_active"] = assigned["drs"].isin(DRS_ACTIVE_CODES).where(assigned["drs"].notna())

    grouped = assigned.groupby("lap_number", sort=True)
    frame = grouped.agg(
        speed_min=("speed", "min"),
        speed_mean=("speed", "mean"),
        speed_max=("speed", "max"),
        rpm_min=("rpm", "min"),
        rpm_mean=("rpm", "mean"),
        rpm_max=("rpm", "max"),
        throttle_mean=("throttle_valid", "mean"),
        brake_applied_share=("brake_applied", "mean"),
        drs_active_share=("drs_active", "mean"),
        n_gear_max=("n_gear_valid", "max"),
        car_data_samples=("speed", "size"),
        car_data_invalid_samples=("invalid_sample", "sum"),
    ).reset_index()

    driver_number = assigned["driver_number"].iloc[0]
    frame.insert(0, "driver_number", driver_number)

    return {
        "frame": frame,
        "samples_read": len(raw),
        "samples_in_window": int(len(assigned)),
        "invalid_samples": int(assigned["invalid_sample"].sum()),
    }


def _empty_car_data_frame() -> pd.DataFrame:
    columns = [
        "driver_number",
        "lap_number",
        "speed_min",
        "speed_mean",
        "speed_max",
        "rpm_min",
        "rpm_mean",
        "rpm_max",
        "throttle_mean",
        "brake_applied_share",
        "drs_active_share",
        "n_gear_max",
        "car_data_samples",
        "car_data_invalid_samples",
    ]
    return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})


def aggregate_intervals(records: List[Mapping[str, Any]], windows: pd.DataFrame) -> pd.DataFrame:
    """Gap and interval as they stood at the end of each lap.

    ``gap_to_leader`` is a float for most records but text ("+1 LAP") for
    lapped cars, so the numeric gap and the fact of being lapped are kept as
    separate columns instead of forcing one into the other.
    """
    columns = ["driver_number", "lap_number", "gap_to_leader_s", "interval_s", "is_lapped", "interval_samples"]
    if not records:
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})

    raw = pd.DataFrame.from_records(records)
    raw["date"] = to_datetime(raw["date"])
    raw["driver_number"] = pd.to_numeric(raw.get("driver_number"), errors="coerce")

    gap = raw.get("gap_to_leader")
    raw["gap_to_leader_s"] = pd.to_numeric(gap, errors="coerce")
    raw["is_lapped_sample"] = gap.map(lambda value: isinstance(value, str)) if gap is not None else False
    raw["interval_s"] = pd.to_numeric(raw.get("interval"), errors="coerce")

    assigned = assign_records_to_laps(raw, windows, "date")
    if assigned.empty:
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})

    assigned = assigned.sort_values("date", kind="stable")
    group_keys = ["driver_number", "lap_number"]

    # Read the gap, the interval and the lapped flag from the *same* final
    # record of the lap, so they describe one instant. A groupby "last"
    # would skip nulls per column and could pair a numeric gap from
    # mid-lap with a lapped flag from the end of it.
    final = assigned.groupby(group_keys, sort=True).tail(1)
    frame = final[group_keys + ["gap_to_leader_s", "interval_s", "is_lapped_sample"]].rename(
        columns={"is_lapped_sample": "is_lapped"}
    )

    counts = assigned.groupby(group_keys, sort=True).size().rename("interval_samples").reset_index()
    return frame.merge(counts, on=group_keys, how="left")


def aggregate_position(records: List[Mapping[str, Any]], windows: pd.DataFrame) -> pd.DataFrame:
    """Position at the start and end of each lap, plus changes within it.

    Position is only reported when it changes — about a quarter of laps
    carry a record — so a plain join would leave most laps blank. The last
    position known at the lap boundary is carried forward instead, which is
    both correct and free of future information.
    """
    columns = ["driver_number", "lap_number", "position_at_lap_start", "position_at_lap_end", "position_changes_in_lap"]
    base = windows[["driver_number", "lap_number", "lap_start_time", "lap_end_time"]].copy()

    if not records:
        for column in columns[2:]:
            base[column] = pd.NA
        return base[columns]

    raw = pd.DataFrame.from_records(records)
    raw["date"] = to_datetime(raw["date"])
    raw["driver_number"] = pd.to_numeric(raw.get("driver_number"), errors="coerce")
    raw["position"] = pd.to_numeric(raw.get("position"), errors="coerce")
    raw = raw.dropna(subset=["date", "driver_number"]).sort_values("date", kind="stable")

    base["position_at_lap_start"] = _asof_value(base, raw, "lap_start_time", "position")
    base["position_at_lap_end"] = _asof_value(base, raw, "lap_end_time", "position")

    assigned = assign_records_to_laps(raw, windows, "date")
    if assigned.empty:
        base["position_changes_in_lap"] = 0
    else:
        counts = assigned.groupby(["driver_number", "lap_number"], sort=True).size().rename("position_changes_in_lap")
        base = base.merge(counts.reset_index(), on=["driver_number", "lap_number"], how="left")
        base["position_changes_in_lap"] = base["position_changes_in_lap"].fillna(0)

    return base[columns]


def _asof_value(
    base: pd.DataFrame,
    records: pd.DataFrame,
    time_column: str,
    value_column: str,
) -> pd.Series:
    """Last known value at each boundary, per driver.

    Rows whose boundary is unknown — the final lap of a driver who has no
    closing timestamp — get no value rather than a guessed one.
    """
    left = base.reset_index().rename(columns={"index": "_row"})
    usable = left.dropna(subset=[time_column]).sort_values(time_column, kind="stable")
    if usable.empty:
        return pd.Series(pd.NA, index=base.index, dtype="Float64")

    merged = pd.merge_asof(
        usable,
        records[["date", "driver_number", value_column]].sort_values("date", kind="stable"),
        left_on=time_column,
        right_on="date",
        by="driver_number",
        direction="backward",
    )
    result = pd.Series(pd.NA, index=base.index, dtype="Float64")
    result.loc[merged["_row"].to_numpy()] = merged[value_column].to_numpy()
    return result


def attach_weather(base: pd.DataFrame, records: List[Mapping[str, Any]], tolerance_seconds: float) -> pd.DataFrame:
    """Conditions as last measured before each lap began.

    Readings arrive every 60 seconds, so a backward match always finds one
    for a race in progress. Backward and not nearest: the nearest reading
    can lie up to half a cadence in the future, and a lap must not be
    described by weather recorded after it started. ``weather_age_s`` keeps
    the choice auditable, and a lap further from a reading than the
    tolerance is left empty rather than filled from too far away.
    """
    fields = [
        "air_temperature",
        "track_temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
    ]
    if not records:
        for column in fields + ["weather_age_s"]:
            base[column] = pd.NA
        return base

    raw = pd.DataFrame.from_records(records)
    raw["date"] = to_datetime(raw["date"])
    for column in fields:
        raw[column] = pd.to_numeric(raw.get(column), errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date", kind="stable")

    left = base.reset_index().rename(columns={"index": "_row"})
    usable = left.dropna(subset=["lap_start_time"]).sort_values("lap_start_time", kind="stable")

    merged = pd.merge_asof(
        usable,
        raw[["date"] + fields],
        left_on="lap_start_time",
        right_on="date",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    merged["weather_age_s"] = (merged["lap_start_time"] - merged["date"]).dt.total_seconds()

    for column in fields + ["weather_age_s"]:
        values = pd.Series(pd.NA, index=base.index, dtype="Float64")
        values.loc[merged["_row"].to_numpy()] = merged[column].to_numpy()
        base[column] = values
    return base


def count_race_control(records: List[Mapping[str, Any]], base: pd.DataFrame) -> pd.DataFrame:
    """Count race control events per lap, and per lap and driver.

    Only counts. The messages carry flags, safety cars and incidents, but
    reading their meaning out of free text is interpretation, and belongs to
    the stage that builds features rather than to the one that assembles the
    table.
    """
    base["rc_events_in_lap"] = 0
    base["rc_driver_events_in_lap"] = 0
    if not records:
        return base

    raw = pd.DataFrame.from_records(records)
    raw["lap_number"] = pd.to_numeric(raw.get("lap_number"), errors="coerce")
    raw = raw.dropna(subset=["lap_number"])
    if raw.empty:
        return base

    per_lap = raw.groupby("lap_number", sort=True).size().rename("rc_events")
    base["rc_events_in_lap"] = base["lap_number"].map(per_lap).fillna(0).astype("int64")

    if "driver_number" in raw.columns:
        driver_events = raw.dropna(subset=["driver_number"]).copy()
        driver_events["driver_number"] = pd.to_numeric(driver_events["driver_number"], errors="coerce")
        driver_events = driver_events.dropna(subset=["driver_number"])
        if not driver_events.empty:
            per_driver = (
                driver_events.groupby(["driver_number", "lap_number"], sort=True)
                .size()
                .rename("rc_driver_events")
                .reset_index()
            )
            base = base.merge(per_driver, on=["driver_number", "lap_number"], how="left")
            base["rc_driver_events_in_lap"] = base["rc_driver_events"].fillna(0).astype("int64")
            base = base.drop(columns=["rc_driver_events"])
    return base


def attach_stints(base: pd.DataFrame, records: List[Mapping[str, Any]]) -> pd.DataFrame:
    """Map each lap onto the tyre stint that covers it.

    Stints were measured to tile a driver's race contiguously — no gaps, no
    overlaps, for every driver in both races checked — so the stint of a lap
    is the last one that started at or before it, confirmed against its end.
    Tyre age follows arithmetically from the stint in force; it never looks
    at a later stint.
    """
    columns = ["stint_number", "compound", "tyre_age", "stint_lap"]
    if not records:
        for column in columns:
            base[column] = pd.NA
        return base

    stints = pd.DataFrame.from_records(records)
    for column in ("driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"):
        stints[column] = pd.to_numeric(stints.get(column), errors="coerce")
    stints = stints.dropna(subset=["driver_number", "lap_start"]).sort_values("lap_start", kind="stable")

    left = base.reset_index().rename(columns={"index": "_row"}).sort_values("lap_number", kind="stable")
    merged = pd.merge_asof(
        left,
        stints[["lap_start", "lap_end", "driver_number", "stint_number", "compound", "tyre_age_at_start"]],
        left_on="lap_number",
        right_on="lap_start",
        by="driver_number",
        direction="backward",
    )
    # A lap past the end of the last stint belongs to no stint at all.
    outside = merged["lap_end"].notna() & (merged["lap_number"] > merged["lap_end"])
    merged.loc[outside, ["stint_number", "compound", "tyre_age_at_start", "lap_start"]] = pd.NA

    merged["tyre_age"] = merged["tyre_age_at_start"] + (merged["lap_number"] - merged["lap_start"])
    merged["stint_lap"] = merged["lap_number"] - merged["lap_start"] + 1

    for column in columns:
        values = pd.Series(pd.NA, index=base.index, dtype="object")
        values.loc[merged["_row"].to_numpy()] = merged[column].to_numpy()
        base[column] = values
    return base


def attach_pit(base: pd.DataFrame, records: List[Mapping[str, Any]]) -> pd.DataFrame:
    """Attach pit stops to the lap on which the car entered the pits.

    ``pit.lap_number`` is the in-lap: in the race checked, all 43 stops had
    ``lap_number + 1`` flagged ``is_pit_out_lap``, and the count of flags
    matched the count of stops exactly. The 2023 race has no pit endpoint at
    all, and there ``is_pit_out_lap`` on the laps themselves remains the
    only record that a stop happened — which is why it is kept as a column
    in its own right.
    """
    base["pit_in_lap"] = False
    base["pit_duration"] = pd.NA
    base["pit_lane_duration"] = pd.NA
    if not records:
        return base

    pit = pd.DataFrame.from_records(records)
    for column in ("driver_number", "lap_number", "pit_duration", "lane_duration"):
        pit[column] = pd.to_numeric(pit.get(column), errors="coerce")
    pit = pit.dropna(subset=["driver_number", "lap_number"])
    if pit.empty:
        return base

    pit = (
        pit.sort_values(["driver_number", "lap_number"], kind="stable")
        .groupby(["driver_number", "lap_number"], as_index=False)
        .agg(pit_duration_new=("pit_duration", "first"), pit_lane_duration_new=("lane_duration", "first"))
    )
    merged = base.merge(pit, on=["driver_number", "lap_number"], how="left")
    merged["pit_in_lap"] = merged["pit_duration_new"].notna() | merged["pit_lane_duration_new"].notna()
    merged["pit_duration"] = merged["pit_duration_new"]
    merged["pit_lane_duration"] = merged["pit_lane_duration_new"]
    return merged.drop(columns=["pit_duration_new", "pit_lane_duration_new"])


def attach_drivers(base: pd.DataFrame, records: List[Mapping[str, Any]]) -> pd.DataFrame:
    """Add the identity columns that make a row readable."""
    base["team_name"] = pd.NA
    base["driver_acronym"] = pd.NA
    if not records:
        return base

    drivers = pd.DataFrame.from_records(records)
    drivers["driver_number"] = pd.to_numeric(drivers.get("driver_number"), errors="coerce")
    keep = drivers.dropna(subset=["driver_number"]).drop_duplicates(subset=["driver_number"], keep="first")
    lookup = keep.set_index("driver_number")

    for column, source in (("team_name", "team_name"), ("driver_acronym", "name_acronym")):
        if source in lookup.columns:
            base[column] = base["driver_number"].map(lookup[source])
    return base
