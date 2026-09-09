"""Builders for realistic OpenF1 records.

The shapes and values mirror what the real API returns for a race weekend,
so a test that trips a rule trips it for a reason that could actually
happen.

Times are coherent on purpose: the race starts at 15:00:00 UTC and every
lap takes exactly ``LAP_SECONDS``, so lap ``n`` spans
``[start + (n-1)*100s, start + n*100s)``. That lets a consolidation test
say precisely which lap a telemetry sample belongs to.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

YEAR = 2023
MEETING_KEY = 1141
SESSION_KEY = 7953

RACE_START = datetime(2023, 3, 5, 15, 0, 0, tzinfo=timezone.utc)
LAP_SECONDS = 100.0


def at(offset_seconds: float) -> str:
    """An ISO timestamp this many seconds after the race start."""
    return (RACE_START + timedelta(seconds=offset_seconds)).isoformat()


def lap_offset(lap_number: int) -> float:
    """When lap ``lap_number`` starts, in seconds after the race start."""
    return (lap_number - 1) * LAP_SECONDS


MEETING_RECORD: Dict[str, Any] = {
    "meeting_key": MEETING_KEY,
    "year": YEAR,
    "meeting_name": "Bahrain Grand Prix",
    "meeting_official_name": "FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2023",
    "country_name": "Bahrain",
    "circuit_key": 63,
    "circuit_short_name": "Sakhir",
    "date_start": "2023-03-03T11:30:00+00:00",
    "is_cancelled": False,
}

SESSION_RECORD: Dict[str, Any] = {
    "session_key": SESSION_KEY,
    "meeting_key": MEETING_KEY,
    "year": YEAR,
    "session_name": "Race",
    "session_type": "Race",
    "date_start": "2023-03-05T15:00:00+00:00",
    "date_end": "2023-03-05T17:00:00+00:00",
    "circuit_key": 63,
    "country_name": "Bahrain",
}


def driver_record(driver_number: int, acronym: str = "VER", team: str = "Red Bull") -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "full_name": f"Driver {driver_number}",
        "name_acronym": acronym,
        "team_name": team,
    }


def lap_record(driver_number: int, lap_number: int, **overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "lap_number": lap_number,
        "lap_duration": LAP_SECONDS,
        "duration_sector_1": 30.1,
        "duration_sector_2": 40.2,
        "duration_sector_3": 29.7,
        "i1_speed": 220,
        "i2_speed": 250,
        "st_speed": 300,
        "is_pit_out_lap": False,
        "date_start": at(lap_offset(lap_number)),
    }
    record.update(overrides)
    return record


def car_data_record(driver_number: int, offset_seconds: float, **overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "date": at(offset_seconds),
        "speed": 280,
        "rpm": 11000,
        "n_gear": 7,
        "throttle": 100,
        "brake": 0,
        "drs": 12,
    }
    record.update(overrides)
    return record


def weather_record(offset_seconds: float, **overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "date": at(offset_seconds),
        "air_temperature": 27.5,
        "track_temperature": 31.2,
        "humidity": 22.0,
        "pressure": 1017.0,
        "rainfall": 0,
        "wind_direction": 180,
        "wind_speed": 1.4,
    }
    record.update(overrides)
    return record


def position_record(driver_number: int, offset_seconds: float, position: int) -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "date": at(offset_seconds),
        "position": position,
    }


def interval_record(
    driver_number: int,
    offset_seconds: float,
    gap_to_leader: Any = 1.5,
    interval: Any = 0.8,
) -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "date": at(offset_seconds),
        "gap_to_leader": gap_to_leader,
        "interval": interval,
    }


def stint_record(
    driver_number: int,
    stint_number: int,
    lap_start: int,
    lap_end: int,
    compound: str = "SOFT",
    tyre_age_at_start: int = 0,
) -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "stint_number": stint_number,
        "lap_start": lap_start,
        "lap_end": lap_end,
        "compound": compound,
        "tyre_age_at_start": tyre_age_at_start,
    }


def pit_record(driver_number: int, lap_number: int, pit_duration: float = 24.5) -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "lap_number": lap_number,
        "date": at(lap_offset(lap_number) + 50),
        "pit_duration": pit_duration,
        "lane_duration": pit_duration + 3.0,
        "stop_duration": None,
    }


def race_control_record(lap_number: int, driver_number: Any = None, message: str = "YELLOW") -> Dict[str, Any]:
    return {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "date": at(lap_offset(lap_number) + 10),
        "lap_number": lap_number,
        "category": "Flag",
        "message": message,
        "driver_number": driver_number,
        "flag": "YELLOW",
        "scope": "Driver" if driver_number else "Track",
        "sector": None,
        "qualifying_phase": None,
    }
