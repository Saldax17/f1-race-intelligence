"""Fixtures for building synthetic raw data trees.

Valid files are written through the real
:class:`~f1_race_intelligence.storage.raw_storage.RawDataStorage`, so the
validation tests exercise the envelope and layout the extraction pipeline
actually produces instead of a hand-rolled imitation that could drift.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from factories import (
    MEETING_KEY,
    MEETING_RECORD,
    SESSION_KEY,
    SESSION_RECORD,
    YEAR,
    driver_record,
    lap_record,
)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture
def write_raw(raw_root: Path) -> Callable[..., Path]:
    """Write one raw file the way the extraction pipeline would."""
    storage = RawDataStorage(base_path=raw_root)

    def _write(
        endpoint: str,
        data: Any,
        *,
        year: Optional[int] = YEAR,
        meeting_key: Optional[int] = None,
        session_key: Optional[int] = None,
        file_name: Optional[str] = None,
    ) -> Path:
        parameters: Dict[str, Any] = {}
        if year is not None:
            parameters["year"] = year
        if meeting_key is not None:
            parameters["meeting_key"] = meeting_key
        if session_key is not None:
            parameters["session_key"] = session_key

        if file_name is None:
            file_name = f"session_{session_key}.json" if session_key is not None else f"{endpoint}.json"
        return storage.save(endpoint, parameters, data=data, file_name=file_name)

    return _write


@pytest.fixture
def valid_dataset(write_raw: Callable[..., Path]) -> Callable[..., None]:
    """Build a small, internally consistent raw tree: one meeting, one race."""

    def _build(
        laps: Optional[List[Dict[str, Any]]] = None,
        drivers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        write_raw("meetings", [MEETING_RECORD])
        write_raw("sessions", [SESSION_RECORD], meeting_key=MEETING_KEY)
        write_raw(
            "drivers",
            drivers if drivers is not None else [driver_record(1, "VER", "Red Bull"), driver_record(44, "HAM", "Mercedes")],
            meeting_key=MEETING_KEY,
            session_key=SESSION_KEY,
        )
        write_raw(
            "laps",
            laps if laps is not None else [lap_record(1, 1), lap_record(1, 2), lap_record(44, 1), lap_record(44, 2)],
            meeting_key=MEETING_KEY,
            session_key=SESSION_KEY,
        )

    return _build


@pytest.fixture
def validation_settings(raw_root: Path, tmp_path: Path) -> Callable[..., AppSettings]:
    """Settings pointing at the temporary raw tree."""

    def _settings(**overrides: Any) -> AppSettings:
        config: Dict[str, Any] = {
            "raw_path": str(raw_root),
            "manifest_path": str(tmp_path / "manifests"),
            "report_path": str(tmp_path / "validation"),
        }
        config.update(overrides)
        return AppSettings(validation=config)

    return _settings
