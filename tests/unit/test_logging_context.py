"""Run context on every log line, without depending on a log file."""

import json
import logging

from f1_race_intelligence.utils.logging import ContextFilter, JSONFormatter, current_log_context, log_context


def render(message: str, **extra) -> dict:
    record = logging.LogRecord("f1.test", logging.INFO, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    ContextFilter().filter(record)
    return json.loads(JSONFormatter().format(record))


def test_context_fields_are_stamped_on_every_record() -> None:
    with log_context(stage="m4", year=2024, session_key=9472, meeting_key=None):
        payload = render("session_consolidated", records=1058)

    assert payload["stage"] == "m4"
    assert payload["year"] == 2024
    assert payload["session_key"] == 9472
    assert payload["records"] == 1058
    assert "meeting_key" not in payload, "None values are not logged"


def test_an_explicit_extra_wins_over_the_context() -> None:
    with log_context(session_key=1):
        payload = render("raw_file_downloaded", session_key=2)

    assert payload["session_key"] == 2


def test_contexts_nest_and_restore() -> None:
    with log_context(stage="m2"):
        with log_context(endpoint="laps"):
            assert current_log_context() == {"stage": "m2", "endpoint": "laps"}
        assert current_log_context() == {"stage": "m2"}
    assert current_log_context() == {}
    assert "stage" not in render("outside")
