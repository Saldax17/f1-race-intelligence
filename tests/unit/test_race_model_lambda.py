"""The forecast Lambda: live-race detection and event dispatch (no AWS, no network)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from f1_race_intelligence.race_model import lambda_handler
from f1_race_intelligence.storage.store import DataStore


class FakeClient:
    def __init__(self, sessions, laps=()):
        self.sessions, self.laps = sessions, list(laps)

    def get_sessions(self, **_):
        return self.sessions

    def get_laps(self, **_):
        return self.laps


RACE = {"session_key": 42, "year": 2026, "date_start": "2026-03-08T04:00:00+00:00",
        "date_end": "2026-03-08T06:00:00+00:00", "session_name": "Race"}


def test_live_race_is_found_only_inside_its_window():
    client = FakeClient([RACE])
    assert lambda_handler.live_race(client, datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc))["session_key"] == 42
    assert lambda_handler.live_race(client, datetime(2026, 3, 8, 6, 20, tzinfo=timezone.utc)) is not None  # grace
    assert lambda_handler.live_race(client, datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)) is None


class FakeRunner:
    def __init__(self, tmp_path, client):
        self.output_store = DataStore.local(str(tmp_path))
        self.cfg = SimpleNamespace(output_path="data/models/m7")
        self.client_factory = lambda: client
        self.calls = []

    def forecast(self, session_key, **kwargs):
        self.calls.append((session_key, kwargs))
        return SimpleNamespace(ok=True, summary={"session_key": session_key, **kwargs}, reason=None)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    def make(client):
        fake = FakeRunner(tmp_path, client)
        monkeypatch.setattr(lambda_handler, "_runner", lambda: fake)
        return fake

    return make


def test_live_mode_is_idle_outside_a_race(runner, monkeypatch):
    monkeypatch.setattr(lambda_handler, "live_race", lambda client: None)
    fake = runner(FakeClient([]))
    assert lambda_handler.handler({"mode": "live"}) == {"status": "idle"}
    assert fake.calls == []


def test_live_mode_forecasts_the_latest_completed_lap_once(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(lambda_handler, "live_race", lambda client: RACE)
    laps = [{"lap_number": 1, "lap_duration": 95.0}, {"lap_number": 2, "lap_duration": 93.0}, {"lap_number": 3}]
    fake = runner(FakeClient([RACE], laps))
    out = lambda_handler.handler({"mode": "live"})
    assert out["status"] == "ok" and fake.calls == [(42, {"lap": 2, "source": "live"})]

    done = tmp_path / "data/models/m7/forecasts/year=2026/session=42/lap=002.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    assert lambda_handler.handler({"mode": "live"})["status"] == "up_to_date"


def test_on_demand_event_and_failures(runner):
    fake = runner(FakeClient([]))
    lambda_handler.handler({"session_key": 7, "lap": 12})
    assert fake.calls[0][0] == 7 and fake.calls[0][1]["lap"] == 12
    with pytest.raises(ValueError):
        lambda_handler.handler({})
    fake.forecast = lambda *a, **k: SimpleNamespace(ok=False, summary={}, reason="no promoted model")
    with pytest.raises(RuntimeError, match="no promoted model"):
        lambda_handler.handler({"session_key": 7})
