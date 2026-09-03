"""High-level client for the OpenF1 historical data API.

``F1Client`` is the *only* place in this codebase allowed to know OpenF1's
endpoint paths and query parameters. Every other module (notebooks,
future feature-engineering code, etc.) must go through it instead of
calling ``requests``/``httpx`` directly — that is what keeps rate
limiting, retries, timeouts, error handling and logging centralized.

See https://openf1.org/docs/ for the full list of endpoints and filters.
Not every possible filter is exposed as an explicit keyword argument yet;
arbitrary additional filters can be passed via ``**filters``.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.ingestion.client import BaseAPIClient
from f1_race_intelligence.ingestion.models import build_params


class F1Client(BaseAPIClient):
    """Client for the public OpenF1 API (https://api.openf1.org/v1)."""

    def __init__(self, settings: Optional[AppSettings] = None, client: Optional[httpx.Client] = None) -> None:
        """Create a client.

        Args:
            settings: Application settings. If omitted, loaded via
                :func:`~f1_race_intelligence.config.settings.load_settings`.
            client: An optional pre-built ``httpx.Client``, mainly for tests.
        """
        settings = settings or load_settings()
        super().__init__(settings.openf1, client=client)

    def get_meetings(
        self,
        *,
        year: Optional[int] = None,
        country_name: Optional[str] = None,
        meeting_key: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /meetings"""
        params = build_params(year=year, country_name=country_name, meeting_key=meeting_key, **filters)
        return self._get("/meetings", params)

    def get_sessions(
        self,
        *,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        session_key: Optional[int] = None,
        session_name: Optional[str] = None,
        **filters: Any,
    ) -> Any:
        """GET /sessions"""
        params = build_params(
            year=year,
            meeting_key=meeting_key,
            session_key=session_key,
            session_name=session_name,
            **filters,
        )
        return self._get("/sessions", params)

    def get_drivers(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /drivers"""
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/drivers", params)

    def get_laps(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        lap_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /laps"""
        params = build_params(
            session_key=session_key,
            driver_number=driver_number,
            lap_number=lap_number,
            **filters,
        )
        return self._get("/laps", params)

    def get_car_data(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /car_data

        Car telemetry is sampled at ~3.7 Hz per car, so an unfiltered call for
        a full session can return millions of rows in one response. Requires
        ``driver_number`` and/or a date-range filter (e.g. ``**{"date>": "...",
        "date<": "..."}``) to keep responses bounded.
        """
        if driver_number is None and not any(key.startswith("date") for key in filters):
            raise ValueError(
                "get_car_data requires 'driver_number' and/or a date range filter "
                "(e.g. date>=...) — an unfiltered call can return millions of rows"
            )
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/car_data", params)

    def get_positions(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /position"""
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/position", params)

    def get_intervals(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /intervals"""
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/intervals", params)

    def get_stints(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /stints"""
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/stints", params)

    def get_pit(
        self,
        *,
        session_key: Optional[int] = None,
        driver_number: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /pit"""
        params = build_params(session_key=session_key, driver_number=driver_number, **filters)
        return self._get("/pit", params)

    def get_weather(
        self,
        *,
        session_key: Optional[int] = None,
        meeting_key: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /weather"""
        params = build_params(session_key=session_key, meeting_key=meeting_key, **filters)
        return self._get("/weather", params)

    def get_race_control(
        self,
        *,
        session_key: Optional[int] = None,
        meeting_key: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        """GET /race_control"""
        params = build_params(session_key=session_key, meeting_key=meeting_key, **filters)
        return self._get("/race_control", params)
