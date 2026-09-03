"""Integration tests that call the real OpenF1 API.

These require internet access and are excluded by default (see the
``addopts`` in pyproject.toml). Run them explicitly with:

    pytest -m integration
"""

import pytest

from f1_race_intelligence.ingestion.openf1 import F1Client

pytestmark = pytest.mark.integration


def test_get_sessions_returns_data_for_a_known_year() -> None:
    with F1Client() as client:
        sessions = client.get_sessions(year=2023, session_name="Race")

    assert isinstance(sessions, list)
    assert len(sessions) > 0
    assert "session_key" in sessions[0]


def test_get_meetings_returns_data_for_a_known_year() -> None:
    with F1Client() as client:
        meetings = client.get_meetings(year=2023)

    assert isinstance(meetings, list)
    assert len(meetings) > 0
