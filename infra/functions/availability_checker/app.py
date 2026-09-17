"""OpenF1 availability checker.

A small, stateless Lambda that answers one question before any pipeline work
is scheduled: *can the historical OpenF1 API be used right now?*

It sends a single lightweight GET to a historical endpoint and returns a
structured verdict that Step Functions branches on. It never starts an
extraction, never writes to S3 and never raises for an HTTP or network
problem: an unavailable API is a normal answer, not a failure. Only a bug in
this function makes the invocation fail, and that is what alerts on.

Verdicts:

    HTTP 200        available=True
    HTTP 401        available=False, reason="openf1_live_restriction"
                    (OpenF1 restricts unauthenticated access while a live
                    session is running)
    HTTP 429        available=False, reason="rate_limited"
    other 4xx/5xx   available=False, reason="openf1_error"
    no response     available=False, reason="network_error"

Only the Python standard library is used, so the deployment package is the
source file itself.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

SOURCE = "openf1"

REASON_LIVE_RESTRICTION = "openf1_live_restriction"
REASON_RATE_LIMITED = "rate_limited"
REASON_OPENF1_ERROR = "openf1_error"
REASON_NETWORK_ERROR = "network_error"

DEFAULT_BASE_URL = "https://api.openf1.org/v1"
# One race session: a historical record that always exists and weighs a few
# hundred bytes, so a check costs OpenF1 (and us) almost nothing.
DEFAULT_PROBE_PATH = "/sessions?session_key=7953"
DEFAULT_TIMEOUT_SECONDS = 5.0
USER_AGENT = "f1-race-intelligence-availability-checker/1.0"

Opener = Callable[..., Any]


def classify(status_code: Optional[int]) -> Dict[str, Any]:
    """Turn an HTTP status (``None`` when no response arrived) into a verdict."""
    if status_code == 200:
        return {"available": True, "reason": None}
    if status_code == 401:
        return {"available": False, "reason": REASON_LIVE_RESTRICTION}
    if status_code == 429:
        return {"available": False, "reason": REASON_RATE_LIMITED}
    if status_code is None:
        return {"available": False, "reason": REASON_NETWORK_ERROR}
    return {"available": False, "reason": REASON_OPENF1_ERROR}


def check_availability(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    opener: Opener = urllib.request.urlopen,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    """Probe ``url`` once and describe the outcome. Never raises for HTTP or network errors."""
    checked_at = now().isoformat()
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    started = time.monotonic()

    status_code: Optional[int] = None
    body: bytes = b""
    headers: Mapping[str, str] = {}
    error: Optional[str] = None

    try:
        with opener(request, timeout=timeout) as response:
            status_code = int(response.status)
            headers = response.headers
            # The probe is tiny; cap the read so a misbehaving server cannot
            # make the function hold megabytes.
            body = response.read(65536)
    except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a status
        status_code = int(exc.code)
        headers = exc.headers or {}
        error = f"HTTP {exc.code}"
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        error = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"

    result: Dict[str, Any] = {
        **classify(status_code),
        "status_code": status_code,
        "checked_at": checked_at,
        "source": SOURCE,
        "endpoint": url,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
    }
    # Same verdict under the name the orchestration contract also uses.
    result["historical_available"] = result["available"]

    if status_code == 200:
        result["records"] = _record_count(body)
    retry_after = _retry_after_seconds(headers)
    if retry_after is not None:
        result["retry_after_seconds"] = retry_after
    if error is not None:
        result["error"] = error[:300]
    return result


def handler(event: Any, context: Any) -> Dict[str, Any]:
    """Lambda entry point. The event (a scheduled EventBridge event) carries nothing needed."""
    base_url = os.environ.get("OPENF1_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    probe_path = os.environ.get("OPENF1_PROBE_PATH", DEFAULT_PROBE_PATH)
    timeout = float(os.environ.get("OPENF1_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))

    result = check_availability(f"{base_url}/{probe_path.lstrip('/')}", timeout=timeout)
    result["environment"] = os.environ.get("ENVIRONMENT", "unknown")
    if context is not None and getattr(context, "aws_request_id", None):
        result["check_id"] = context.aws_request_id

    _log(result)
    return result


def _record_count(body: bytes) -> Optional[int]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return len(payload) if isinstance(payload, list) else None


def _retry_after_seconds(headers: Mapping[str, str]) -> Optional[int]:
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _log(result: Dict[str, Any]) -> None:
    """One JSON log line that CloudWatch also reads as metrics (Embedded Metric Format).

    EMF needs no ``cloudwatch:PutMetricData`` permission: the metric is
    extracted from the log line, which the function may already write.
    """
    namespace = os.environ.get("METRICS_NAMESPACE")
    line: Dict[str, Any] = {"message": "openf1_availability_checked", **result}
    if namespace:
        line["_aws"] = {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [["Environment"]],
                    "Metrics": [
                        {"Name": "OpenF1Available", "Unit": "Count"},
                        {"Name": "OpenF1LatencyMs", "Unit": "Milliseconds"},
                    ],
                }
            ],
        }
        line["Environment"] = result.get("environment", "unknown")
        line["OpenF1Available"] = 1 if result["available"] else 0
        line["OpenF1LatencyMs"] = result["latency_ms"]
    print(json.dumps(line, default=str))
