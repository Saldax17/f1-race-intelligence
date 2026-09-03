"""Structured logging setup.

Every ``F1Client`` request logs endpoint, method, params, status code,
duration and retry attempt via the standard ``logging`` module's ``extra``
mechanism (see ``client.py``). This module just wires up a formatter that
renders those extra fields as a single JSON line per log record, so logs
stay greppable/parseable without pulling in an external logging library.

No secrets are logged: OpenF1's historical endpoints are unauthenticated,
and request logging only ever includes endpoint names, query filters,
status codes and timings.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

_RESERVED_RECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JSONFormatter(logging.Formatter):
    """Renders each log record as a single JSON line, including ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure the root logger once. Safe to call multiple times (no-op after the first)."""
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper around ``logging.getLogger`` for consistent module-level loggers."""
    return logging.getLogger(name)
