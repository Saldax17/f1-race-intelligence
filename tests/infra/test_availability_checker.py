"""The OpenF1 availability checker, against a real local HTTP server.

No request leaves the machine: a throwaway server on 127.0.0.1 answers with
whatever status the test needs, so the real ``urllib`` code path — including
how it raises for 4xx/5xx — is what gets exercised.
"""

import http.server
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import app


@contextmanager
def openf1_stub(status: int, body: bytes = b"[]", headers=None, delay: float = 0.0):
    """Serve every GET with the given status on an ephemeral local port."""

    class Handler(http.server.BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):  # noqa: N802 - http.server naming
            Handler.requests.append({"path": self.path, "user_agent": self.headers.get("User-Agent")})
            if delay:
                threading.Event().wait(delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", Handler.requests
    finally:
        server.shutdown()
        server.server_close()


SESSION = json.dumps([{"session_key": 7953, "session_name": "Race", "year": 2023}]).encode()


def test_http_200_means_historical_data_is_available() -> None:
    with openf1_stub(200, SESSION) as (base, requests):
        result = app.check_availability(f"{base}/sessions?session_key=7953")

    assert result["available"] is True
    assert result["historical_available"] is True
    assert result["status_code"] == 200
    assert result["reason"] is None
    assert result["source"] == "openf1"
    assert result["records"] == 1
    assert requests == [{"path": "/sessions?session_key=7953", "user_agent": app.USER_AGENT}]


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "openf1_live_restriction"),
        (429, "rate_limited"),
        (400, "openf1_error"),
        (403, "openf1_error"),
        (404, "openf1_error"),
        (500, "openf1_error"),
        (502, "openf1_error"),
        (503, "openf1_error"),
    ],
)
def test_non_200_statuses_are_unavailable_with_a_reason(status: int, reason: str) -> None:
    with openf1_stub(status, b'{"detail": "nope"}') as (base, _):
        result = app.check_availability(f"{base}/sessions")

    assert result["available"] is False
    assert result["historical_available"] is False
    assert result["status_code"] == status
    assert result["reason"] == reason
    assert "records" not in result


def test_rate_limit_reports_retry_after() -> None:
    with openf1_stub(429, b"{}", headers={"Retry-After": "60"}) as (base, _):
        result = app.check_availability(f"{base}/sessions")

    assert result["retry_after_seconds"] == 60


def test_a_timeout_is_a_network_error_not_an_exception() -> None:
    with openf1_stub(200, SESSION, delay=1.0) as (base, _):
        result = app.check_availability(f"{base}/sessions", timeout=0.2)

    assert result["available"] is False
    assert result["status_code"] is None
    assert result["reason"] == "network_error"
    assert "error" in result


def test_an_unreachable_host_is_a_network_error() -> None:
    # Port 9 on loopback: nothing listens there, so the connection is refused.
    result = app.check_availability("http://127.0.0.1:9/v1/sessions", timeout=1)

    assert (result["available"], result["status_code"], result["reason"]) == (False, None, "network_error")


def test_a_200_with_an_unexpected_body_is_still_available_but_says_so() -> None:
    with openf1_stub(200, b"not json") as (base, _):
        result = app.check_availability(f"{base}/sessions")

    assert result["available"] is True
    assert result["records"] is None


def test_the_result_is_json_serialisable_and_stamped_in_utc() -> None:
    frozen = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    with openf1_stub(200, SESSION) as (base, _):
        result = app.check_availability(f"{base}/sessions", now=lambda: frozen)

    assert json.loads(json.dumps(result)) == result
    assert result["checked_at"] == "2026-09-16T12:00:00+00:00"
    assert isinstance(result["latency_ms"], float)


def test_handler_uses_environment_configuration_and_logs_an_emf_metric(monkeypatch, capsys) -> None:
    with openf1_stub(401, b"{}") as (base, requests):
        monkeypatch.setenv("OPENF1_BASE_URL", base + "/v1/")
        monkeypatch.setenv("OPENF1_PROBE_PATH", "/sessions?session_key=7953")
        monkeypatch.setenv("ENVIRONMENT", "dev")
        monkeypatch.setenv("METRICS_NAMESPACE", "f1-race-intelligence/dev")

        result = app.handler({"source": "aws.events"}, SimpleNamespace(aws_request_id="req-1"))

    assert requests[0]["path"] == "/v1/sessions?session_key=7953"
    assert result["reason"] == "openf1_live_restriction"
    assert result["environment"] == "dev"
    assert result["check_id"] == "req-1"

    log = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert log["message"] == "openf1_availability_checked"
    assert log["OpenF1Available"] == 0
    assert log["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "f1-race-intelligence/dev"


def test_handler_defaults_do_not_depend_on_optional_variables(monkeypatch, capsys) -> None:
    calls = []

    def fake_check(url, timeout):
        calls.append((url, timeout))
        return {**app.classify(200), "status_code": 200, "checked_at": "x", "source": "openf1", "endpoint": url, "latency_ms": 1.0, "historical_available": True}

    for name in ("OPENF1_BASE_URL", "OPENF1_PROBE_PATH", "OPENF1_TIMEOUT_SECONDS", "METRICS_NAMESPACE", "ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app, "check_availability", fake_check)

    result = app.handler({}, None)

    assert calls == [("https://api.openf1.org/v1/sessions?session_key=7953", 5.0)]
    assert result["environment"] == "unknown"
    assert "_aws" not in json.loads(capsys.readouterr().out)


def test_the_function_never_starts_anything_or_touches_aws() -> None:
    source = (app.__file__ and open(app.__file__, encoding="utf-8").read())

    for forbidden in ("boto3", "botocore", "start_execution", "StartExecution", "subprocess"):
        assert forbidden not in source
