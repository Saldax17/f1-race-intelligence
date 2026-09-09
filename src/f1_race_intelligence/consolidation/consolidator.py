"""Assembling one session into the lap dataset.

Laps is the base and nothing is allowed to change its row count: every
source is reduced to one row per driver and lap *before* it is joined, so
each join is a left join onto a grain that already exists. That is what
keeps a cartesian blow-up impossible by construction rather than by
vigilance, and it is why every stage can assert ``rows_out == rows_in``.

Where a source has nothing to say about a lap, the columns stay empty. No
value is invented to fill a gap.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from f1_race_intelligence.consolidation import aggregators
from f1_race_intelligence.consolidation.loaders import SessionLoader, SessionSources
from f1_race_intelligence.consolidation.models import SessionResult, StageCounter
from f1_race_intelligence.storage.raw_catalog import RawFileError
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

KEY_COLUMNS = ["year", "meeting_key", "session_key", "driver_number", "lap_number"]

LAP_COLUMNS = [
    "lap_start_time",
    "lap_end_time",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "is_pit_out_lap",
    "is_last_lap_for_driver",
    "has_complete_window",
]

COLUMN_ORDER = (
    KEY_COLUMNS
    + ["team_name", "driver_acronym"]
    + LAP_COLUMNS
    + ["stint_number", "compound", "tyre_age", "stint_lap"]
    + ["pit_in_lap", "pit_duration", "pit_lane_duration"]
    + ["position_at_lap_start", "position_at_lap_end", "position_changes_in_lap"]
    + ["gap_to_leader_s", "interval_s", "is_lapped", "interval_samples"]
    + [
        "air_temperature",
        "track_temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "weather_age_s",
    ]
    + [
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
    + ["rc_events_in_lap", "rc_driver_events_in_lap"]
)

INTEGER_COLUMNS = (
    "year",
    "meeting_key",
    "session_key",
    "driver_number",
    "lap_number",
    "stint_number",
    "tyre_age",
    "stint_lap",
    "position_at_lap_start",
    "position_at_lap_end",
    "position_changes_in_lap",
    "interval_samples",
    "i1_speed",
    "i2_speed",
    "st_speed",
    "speed_min",
    "speed_max",
    "rpm_min",
    "rpm_max",
    "n_gear_max",
    "car_data_samples",
    "car_data_invalid_samples",
    "rc_events_in_lap",
    "rc_driver_events_in_lap",
    "rainfall",
    "wind_direction",
)

BOOLEAN_COLUMNS = (
    "is_pit_out_lap",
    "is_last_lap_for_driver",
    "has_complete_window",
    "pit_in_lap",
    "is_lapped",
)


class SessionConsolidator:
    """Builds the lap-grain table for one session."""

    def __init__(self, loader: SessionLoader, weather_tolerance_seconds: float = 120.0) -> None:
        self._loader = loader
        self._weather_tolerance = weather_tolerance_seconds

    def consolidate(self, sources: SessionSources) -> tuple[pd.DataFrame, SessionResult]:
        """Return the consolidated frame and the account of how it was built."""
        started = time.monotonic()
        result = SessionResult(
            session_key=sources.session_key,
            year=sources.year,
            meeting_key=sources.meeting_key,
        )

        laps = sources.get("laps")
        if not laps:
            result.error = "no laps data for this session"
            result.duration_seconds = round(time.monotonic() - started, 3)
            return pd.DataFrame(columns=COLUMN_ORDER), result

        base = self._build_base(laps, sources, result)
        windows = base[["driver_number", "lap_number", "lap_start_time", "lap_end_time", "has_complete_window"]]

        base = self._stage(result, "drivers", base, lambda frame: aggregators.attach_drivers(frame, sources.get("drivers")))
        base = self._stage(result, "stints", base, lambda frame: aggregators.attach_stints(frame, sources.get("stints")))
        base = self._stage(result, "pit", base, lambda frame: aggregators.attach_pit(frame, sources.get("pit")))
        base = self._stage(
            result,
            "position",
            base,
            lambda frame: self._merge_lap_frame(frame, aggregators.aggregate_position(sources.get("position"), windows)),
        )
        base = self._stage(
            result,
            "intervals",
            base,
            lambda frame: self._merge_lap_frame(frame, aggregators.aggregate_intervals(sources.get("intervals"), windows)),
        )
        base = self._stage(
            result,
            "weather",
            base,
            lambda frame: aggregators.attach_weather(frame, sources.get("weather"), self._weather_tolerance),
        )
        base = self._stage(
            result,
            "race_control",
            base,
            lambda frame: aggregators.count_race_control(sources.get("race_control"), frame),
        )
        base = self._stage(result, "car_data", base, lambda frame: self._attach_car_data(frame, sources, windows, result))

        base = self._finalize(base)
        result.rows_out = len(base)
        result.drivers = int(base["driver_number"].nunique())
        result.laps = int(base["lap_number"].nunique())
        result.checks = self._run_checks(base, result)
        result.duration_seconds = round(time.monotonic() - started, 3)
        return base, result

    # -- stages -----------------------------------------------------------

    def _build_base(self, laps: List[Mapping[str, Any]], sources: SessionSources, result: SessionResult) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(laps)
        result.rows_in = len(frame)

        for column in ("session_key", "meeting_key", "driver_number", "lap_number", "lap_duration",
                       "duration_sector_1", "duration_sector_2", "duration_sector_3",
                       "i1_speed", "i2_speed", "st_speed"):
            if column not in frame.columns:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "date_start" not in frame.columns:
            frame["date_start"] = pd.NA
        if "is_pit_out_lap" not in frame.columns:
            frame["is_pit_out_lap"] = pd.NA

        frame = aggregators.build_lap_windows(frame)
        frame["year"] = sources.year
        frame["session_key"] = frame["session_key"].fillna(sources.session_key)
        frame["meeting_key"] = frame["meeting_key"].fillna(sources.meeting_key)

        dropped = frame["driver_number"].isna() | frame["lap_number"].isna()
        if dropped.any():
            # A lap with no driver or no number cannot be placed in the grain.
            result.add_stage(
                StageCounter(
                    stage="base_keys",
                    rows_in=len(frame),
                    rows_out=int((~dropped).sum()),
                    note="laps without a driver_number or lap_number cannot be keyed",
                )
            )
            frame = frame[~dropped]

        incomplete = int((~frame["has_complete_window"]).sum())
        result.add_stage(
            StageCounter(
                stage="lap_windows",
                rows_in=len(frame),
                rows_out=len(frame),
                matched=len(frame) - incomplete,
                unmatched=incomplete,
                note="laps without a closable window keep their row; window-based columns stay null",
            )
        )
        return frame.reset_index(drop=True)

    def _stage(self, result: SessionResult, name: str, frame: pd.DataFrame, operation) -> pd.DataFrame:
        """Run one join and record what it did to the row count."""
        rows_in = len(frame)
        out = operation(frame)
        rows_out = len(out)

        matched = unmatched = None
        marker = _STAGE_MARKERS.get(name)
        if marker and marker in out.columns:
            matched = int(out[marker].notna().sum())
            unmatched = rows_out - matched

        result.add_stage(StageCounter(stage=name, rows_in=rows_in, rows_out=rows_out, matched=matched, unmatched=unmatched))
        if rows_out != rows_in:
            logger.error(
                "consolidation_row_count_changed",
                extra={"stage": name, "rows_in": rows_in, "rows_out": rows_out, "session_key": result.session_key},
            )
        return out

    def _merge_lap_frame(self, base: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
        """Left-join an already lap-grained frame onto the base."""
        if addition.empty:
            for column in addition.columns:
                if column not in base.columns and column not in ("driver_number", "lap_number"):
                    base[column] = pd.NA
            return base
        return base.merge(addition, on=["driver_number", "lap_number"], how="left")

    def _attach_car_data(
        self,
        base: pd.DataFrame,
        sources: SessionSources,
        windows: pd.DataFrame,
        result: SessionResult,
    ) -> pd.DataFrame:
        """Aggregate telemetry one driver file at a time.

        Only one file is ever held in memory. A session's telemetry is
        around 180 MB on disk, and a full history several gigabytes, so the
        per-driver file is the unit that keeps this bounded — which is
        exactly how the extraction pipeline already partitions it.
        """
        frames: List[pd.DataFrame] = []
        totals = {"files": 0, "files_unreadable": 0, "samples_read": 0, "samples_in_window": 0, "invalid_samples": 0}

        for raw_file in sources.car_data_files:
            try:
                records = list(self._loader.iter_car_data(raw_file))
            except RawFileError as exc:
                totals["files_unreadable"] += 1
                logger.error(
                    "car_data_file_unreadable",
                    extra={"path": str(raw_file.path), "error": str(exc), "session_key": sources.session_key},
                )
                continue

            driver_windows = windows
            if raw_file.driver_number is not None:
                driver_windows = windows[windows["driver_number"] == raw_file.driver_number]

            outcome = aggregators.aggregate_car_data(records, driver_windows)
            totals["files"] += 1
            totals["samples_read"] += outcome["samples_read"]
            totals["samples_in_window"] += outcome["samples_in_window"]
            totals["invalid_samples"] += outcome["invalid_samples"]
            if not outcome["frame"].empty:
                frames.append(outcome["frame"])

        result.car_data = totals

        if not frames:
            for column in aggregators._empty_car_data_frame().columns:
                if column not in ("driver_number", "lap_number"):
                    base[column] = pd.NA
            return base

        combined = pd.concat(frames, ignore_index=True)
        return base.merge(combined, on=["driver_number", "lap_number"], how="left")

    # -- finishing --------------------------------------------------------

    def _finalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Order rows and columns, and settle dtypes for a typed Parquet file."""
        for column in COLUMN_ORDER:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame = frame[COLUMN_ORDER].sort_values(
            ["session_key", "driver_number", "lap_number"], kind="stable"
        ).reset_index(drop=True)

        for column in INTEGER_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round().astype("Int64")
        for column in BOOLEAN_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        for column in ("team_name", "driver_acronym", "compound"):
            frame[column] = frame[column].astype("string")
        for column in frame.columns:
            if column not in INTEGER_COLUMNS and column not in BOOLEAN_COLUMNS:
                if column in ("team_name", "driver_acronym", "compound", "lap_start_time", "lap_end_time"):
                    continue
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        return frame

    def _run_checks(self, frame: pd.DataFrame, result: SessionResult) -> Dict[str, Any]:
        """Confirm the consolidation itself did not introduce problems.

        This does not re-validate the raw data — that is M3's job — it asks
        whether assembling it produced a sound table.
        """
        grain = ["session_key", "driver_number", "lap_number"]
        duplicated = int(frame.duplicated(subset=grain).sum())
        laps_per_driver = frame.groupby("driver_number", sort=True)["lap_number"].count()

        return {
            "grain_unique": duplicated == 0,
            "duplicate_rows": duplicated,
            "rows_match_source": result.rows_in == len(frame),
            "laps_without_window": int((~frame["has_complete_window"].fillna(False)).sum()),
            "laps_without_car_data": int(frame["car_data_samples"].isna().sum()),
            "laps_without_weather": int(frame["air_temperature"].isna().sum()),
            "laps_without_stint": int(frame["stint_number"].isna().sum()),
            "laps_without_position": int(frame["position_at_lap_start"].isna().sum()),
            "laps_per_driver_min": int(laps_per_driver.min()) if len(laps_per_driver) else 0,
            "laps_per_driver_max": int(laps_per_driver.max()) if len(laps_per_driver) else 0,
        }


# Column whose presence tells whether a stage found a match for a row. It has
# to be one that is null exactly when nothing matched: gap_to_leader_s would
# be wrong for intervals, because a lapped car legitimately has a record but
# no numeric gap, and would be counted as unmatched.
#
# pit and race_control are deliberately absent. They describe events, so most
# laps having none is normal, and an "unmatched" count would read as loss
# where there is none.
_STAGE_MARKERS = {
    "drivers": "team_name",
    "stints": "stint_number",
    "position": "position_at_lap_start",
    "intervals": "interval_samples",
    "weather": "air_temperature",
    "car_data": "car_data_samples",
}
