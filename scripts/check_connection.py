"""Manual smoke test: OpenF1 -> F1Client -> GET /sessions -> log.

Run with:

    python scripts/check_connection.py

This performs a real HTTP call to the public OpenF1 API (no credentials
needed) and prints how many sessions were returned. It exists to give a
quick, runnable confirmation that configuration, the HTTP client, rate
limiting and logging are all wired together correctly end to end.
"""

from __future__ import annotations

from f1_race_intelligence.config.settings import load_settings
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    logger.info("connection_check_start", extra={"base_url": settings.openf1.base_url})

    with F1Client(settings=settings) as client:
        sessions = client.get_sessions(year=2023, session_name="Race")

    logger.info("connection_check_success", extra={"sessions_returned": len(sessions)})
    print(f"OK: retrieved {len(sessions)} race sessions from OpenF1 for 2023.")


if __name__ == "__main__":
    main()
