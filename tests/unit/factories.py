"""Builders for realistic OpenF1 records.

The shapes and values here mirror what the real API returns for a race
weekend, so a test that trips a rule trips it for a reason that could
actually happen.
"""

from typing import Any, Dict

YEAR = 2023
MEETING_KEY = 1141
SESSION_KEY = 7953

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
        "lap_duration": 95.5 + lap_number,
        "duration_sector_1": 30.1,
        "duration_sector_2": 40.2,
        "duration_sector_3": 25.2,
        "i1_speed": 220,
        "i2_speed": 250,
        "st_speed": 300,
        "is_pit_out_lap": False,
        "date_start": f"2023-03-05T15:{lap_number:02d}:00+00:00",
    }
    record.update(overrides)
    return record


def car_data_record(driver_number: int, second: int, **overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "driver_number": driver_number,
        "date": f"2023-03-05T15:00:{second:02d}+00:00",
        "speed": 280,
        "rpm": 11000,
        "n_gear": 7,
        "throttle": 100,
        "brake": 0,
        "drs": 12,
    }
    record.update(overrides)
    return record


def weather_record(minute: int, **overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "session_key": SESSION_KEY,
        "meeting_key": MEETING_KEY,
        "date": f"2023-03-05T15:{minute:02d}:00+00:00",
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
