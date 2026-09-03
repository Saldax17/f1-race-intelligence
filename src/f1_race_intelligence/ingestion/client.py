"""Generic, reusable HTTP client: rate limiting, retries, timeouts, error
translation and structured logging.

This module knows nothing about OpenF1's specific endpoints — that belongs
to :mod:`f1_race_intelligence.ingestion.openf1`. Keeping the two separate
means the transport concerns (rate limiting, retries, timeouts, logging)
can be reused for any future API client without duplicating logic, and the
endpoint layer stays a thin, readable list of "which path + which filters".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from f1_race_intelligence.config.settings import OpenF1Settings
from f1_race_intelligence.ingestion.exceptions import (
    OpenF1ConnectionError,
    OpenF1Error,
    OpenF1HTTPError,
    OpenF1RateLimitError,
    OpenF1TimeoutError,
)
from f1_race_intelligence.ingestion.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an error is transient and worth retrying.

    Only network-level failures, timeouts, and the specific HTTP status
    codes OpenF1/servers use for transient failures are retried. A 4xx
    error like 400 or 404 is a permanent client error and retrying it
    would just waste the retry budget and rate-limit allowance.
    """
    if isinstance(exc, (OpenF1TimeoutError, OpenF1ConnectionError, OpenF1RateLimitError)):
        return True
    if isinstance(exc, OpenF1HTTPError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


class BaseAPIClient:
    """A rate-limited, retrying, timeout-bounded JSON HTTP client."""

    def __init__(self, settings: OpenF1Settings, client: Optional[httpx.Client] = None) -> None:
        self._settings = settings
        self._rate_limiter = SlidingWindowRateLimiter(
            requests_per_second=settings.rate_limit.requests_per_second,
            requests_per_minute=settings.rate_limit.requests_per_minute,
        )
        # A trailing slash on base_url plus relative (no leading slash) request
        # paths is required for httpx's URL joining to keep the "/v1" prefix.
        base_url = settings.base_url.rstrip("/") + "/"
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=settings.timeout.connect,
                read=settings.timeout.read,
                write=settings.timeout.read,
                pool=settings.timeout.connect,
            ),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BaseAPIClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        """Perform a rate-limited, retried GET request and return parsed JSON."""
        return self._request("GET", path, params=params)

    def _request(self, method: str, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        params = dict(params or {})
        retryer = Retrying(
            stop=stop_after_attempt(self._settings.retry.max_attempts),
            wait=self._wait_for_retry,
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._log_retry,
            reraise=True,
        )
        return retryer(self._do_request, method, path, params)

    def _wait_for_retry(self, retry_state: RetryCallState) -> float:
        """Exponential backoff, unless the server told us a longer wait via ``Retry-After``."""
        backoff = wait_exponential(multiplier=self._settings.retry.backoff_factor, min=1, max=60)(retry_state)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return max(backoff, retry_after)
        return backoff

    def _do_request(self, method: str, path: str, params: Mapping[str, Any]) -> Any:
        endpoint = path.lstrip("/")
        self._rate_limiter.acquire()

        logger.debug(
            "openf1_request_start",
            extra={"endpoint": endpoint, "method": method, "params": params},
        )

        start = time.monotonic()
        try:
            response = self._client.request(method, endpoint, params=params)
        except httpx.TimeoutException as exc:
            duration = time.monotonic() - start
            logger.warning(
                "openf1_request_timeout",
                extra={"endpoint": endpoint, "method": method, "duration_ms": round(duration * 1000, 1)},
            )
            raise OpenF1TimeoutError(f"Timeout calling {endpoint}") from exc
        except httpx.HTTPError as exc:
            duration = time.monotonic() - start
            logger.warning(
                "openf1_request_connection_error",
                extra={
                    "endpoint": endpoint,
                    "method": method,
                    "duration_ms": round(duration * 1000, 1),
                    "error": str(exc),
                },
            )
            raise OpenF1ConnectionError(f"Connection error calling {endpoint}") from exc

        duration = time.monotonic() - start
        log_fields = {
            "endpoint": endpoint,
            "method": method,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 1),
        }

        if response.status_code == 429:
            logger.warning("openf1_rate_limited", extra=log_fields)
            retry_after = response.headers.get("Retry-After")
            raise OpenF1RateLimitError(
                endpoint=endpoint,
                retry_after=float(retry_after) if retry_after else None,
            )

        if response.status_code >= 400:
            logger.error("openf1_http_error", extra=log_fields)
            raise OpenF1HTTPError(
                status_code=response.status_code,
                message=f"OpenF1 returned HTTP {response.status_code} for {endpoint}",
                endpoint=endpoint,
                body=response.text,
            )

        logger.info("openf1_request_success", extra=log_fields)

        try:
            return response.json()
        except ValueError as exc:
            raise OpenF1Error(f"Invalid JSON response from {endpoint}") from exc

    @staticmethod
    def _log_retry(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        wait_seconds = retry_state.next_action.sleep if retry_state.next_action else None
        logger.warning(
            "openf1_request_retry",
            extra={
                "attempt": retry_state.attempt_number,
                "wait_seconds": round(wait_seconds, 3) if wait_seconds is not None else None,
                "error": str(exc) if exc else None,
                "error_type": type(exc).__name__ if exc else None,
            },
        )
