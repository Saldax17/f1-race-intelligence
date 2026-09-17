"""Structured logging setup.

Every ``F1Client`` request logs endpoint, method, params, status code,
duration and retry attempt via the standard ``logging`` module's ``extra``
mechanism (see ``client.py``). This module just wires up a formatter that
renders those extra fields as a single JSON line per log record, so logs
stay greppable/parseable without pulling in an external logging library.

Logs go to stdout, which is what a container runtime ships to its log
service; a local file is an optional extra for development. Fields that
describe *what a run is working on* — stage, year, meeting and session —
are set once with :func:`log_context` and stamped onto every record logged
inside it, so a line from deep inside a stage is still attributable to the
job that produced it.

No secrets are logged: OpenF1's historical endpoints are unauthenticated,
and request logging only ever includes endpoint names, query filters,
status codes and timings.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_RESERVED_RECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("f1_log_context", default={})


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


class ContextFilter(logging.Filter):
    """Copies the active :func:`log_context` fields onto each record.

    A field passed explicitly through ``extra`` wins over the context, so a
    log call can always say something more specific than its surroundings.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _CONTEXT.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every record logged inside the block. ``None`` values are ignored."""
    token = _CONTEXT.set({**_CONTEXT.get(), **{key: value for key, value in fields.items() if value is not None}})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_log_context() -> Dict[str, Any]:
    return dict(_CONTEXT.get())


def _ensure_context_filter(handler: logging.Handler) -> None:
    if not any(isinstance(existing, ContextFilter) for existing in handler.filters):
        handler.addFilter(ContextFilter())


def configure_logging(level: str = "INFO", json_format: bool = True, file_path: Optional[str] = None) -> None:
    """Configure the root logger once. Safe to call multiple times (no-op after the first).

    Args:
        level: Root log level.
        json_format: One JSON object per line (for log services) or plain text.
        file_path: Optional development log file, written in addition to stdout.
    """
    root = logging.getLogger()
    if root.handlers:
        # Someone (a test runner, a hosting runtime) already set logging up.
        # Leave their handlers alone, but still stamp the run context on them.
        for handler in root.handlers:
            _ensure_context_filter(handler)
        return

    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    handlers: list = [logging.StreamHandler(sys.stdout)]
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        _ensure_context_filter(handler)
        root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper around ``logging.getLogger`` for consistent module-level loggers."""
    return logging.getLogger(name)
