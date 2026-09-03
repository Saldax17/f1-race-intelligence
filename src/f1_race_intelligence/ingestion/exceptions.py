"""Domain exceptions for the OpenF1 ingestion layer.

Business code should only ever catch these, never ``httpx`` exceptions
directly. That keeps the HTTP library an implementation detail of
:mod:`f1_race_intelligence.ingestion.client` and free to change later.
"""

from __future__ import annotations

from typing import Any, Optional


class OpenF1Error(Exception):
    """Base class for all errors raised by the OpenF1 ingestion layer."""


class OpenF1ConnectionError(OpenF1Error):
    """Raised when a network-level connection to OpenF1 could not be established."""


class OpenF1TimeoutError(OpenF1Error):
    """Raised when a request to OpenF1 exceeds the configured connect/read timeout."""


class OpenF1HTTPError(OpenF1Error):
    """Raised when OpenF1 responds with an HTTP error status code."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        endpoint: Optional[str] = None,
        body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint
        self.body = body


class OpenF1RateLimitError(OpenF1HTTPError):
    """Raised when OpenF1 responds with HTTP 429 (Too Many Requests)."""

    def __init__(
        self,
        message: str = "OpenF1 rate limit exceeded",
        *,
        endpoint: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(status_code=429, message=message, endpoint=endpoint)
        self.retry_after = retry_after
