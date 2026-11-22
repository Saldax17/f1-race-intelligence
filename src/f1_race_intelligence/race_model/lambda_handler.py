"""AWS Lambda entry point for race forecasts (container image, see ``Dockerfile.forecast``).

Events:

* ``{"session_key": 9998, "lap": 30}`` — forecast one race at one lap from the
  raw tree in S3 (``"source": "live"`` reads OpenF1 instead).
* ``{"mode": "live"}`` — what the in-race schedule sends: find the Grand Prix
  running now, and forecast from the leader's latest completed lap if that
  lap has no forecast yet. Outside a race it returns ``{"status": "idle"}``.

Configuration comes from the environment, exactly as for the batch image:
``F1_ENVIRONMENT=aws``, ``F1_STORAGE_BUCKET`` and ``F1_CONFIG_PATH``. The
function reads ``raw/`` and ``models/`` and writes only ``forecasts/``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

from f1_race_intelligence.config.settings import load_settings
from f1_race_intelligence.race_model.contracts import validate_forecast_request
from f1_race_intelligence.storage.backends.base import join_key
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

LIVE_GRACE = timedelta(minutes=30)
"""A race is still "live" for this long after its scheduled end (red flags, slow finishes)."""

_RUNNER = None


def _runner():
    global _RUNNER
    if _RUNNER is None:  # reused across warm invocations
        from f1_race_intelligence.ingestion.openf1 import F1Client
        from f1_race_intelligence.race_model.runner import RaceModelRunner

        settings = load_settings()
        configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)
        store = DataStore.from_settings(settings)
        _RUNNER = RaceModelRunner(settings, store, store, client_factory=lambda: F1Client(settings=settings))
    return _RUNNER


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def live_race(client: Any, now: Optional[datetime] = None) -> Optional[Mapping[str, Any]]:
    """The Grand Prix session in progress at ``now``, if any (sprints are not forecast)."""
    now = now or datetime.now(timezone.utc)
    sessions = client.get_sessions(year=now.year, session_name="Race") or []
    for session in sessions:
        start, end = _parse(session.get("date_start")), _parse(session.get("date_end"))
        if start and end and start <= now <= end + LIVE_GRACE and not session.get("is_cancelled"):
            return session
    return None


def handler(event: Optional[Mapping[str, Any]], context: Any = None) -> Dict[str, Any]:
    event = validate_forecast_request(dict(event or {}))
    runner = _runner()
    if event.get("mode") == "live":
        client = runner.client_factory()
        session = live_race(client)
        if session is None:
            return {"status": "idle"}
        laps = client.get_laps(session_key=session["session_key"]) or []
        lap = max((l.get("lap_number") or 0 for l in laps if l.get("lap_duration")), default=0)
        if lap < 1:
            return {"status": "waiting", "session_key": session["session_key"]}
        from f1_race_intelligence.race_model.artifact import forecasts_prefix

        key = join_key(
            forecasts_prefix(runner.output_store, runner.cfg.output_path),
            f"year={session['year']}",
            f"session={session['session_key']}",
            f"lap={lap:03d}.json",
        )
        if runner.output_store.backend.exists(key):
            return {"status": "up_to_date", "session_key": session["session_key"], "lap": lap}
        live_kwargs: Dict[str, Any] = {"lap": lap, "source": "live"}
        if event.get("model_version") is not None:
            live_kwargs["version"] = event["model_version"]
        result = runner.forecast(int(session["session_key"]), **live_kwargs)
    else:
        if "session_key" not in event:
            raise ValueError("event needs 'session_key' (and optionally 'lap', 'source') or 'mode': 'live'")
        result = runner.forecast(
            int(event["session_key"]),
            lap=event.get("lap"),
            source=event.get("source", "raw"),
            version=event.get("model_version"),
            total_laps=event.get("total_laps"),
            year=event.get("year"),
        )
    if not result.ok:
        # Raise so Lambda records an error: the alarm and the async OnFailure destination see it.
        raise RuntimeError(result.reason)
    logger.info("forecast_written", extra=result.summary)
    return {"status": "ok", **result.summary}
