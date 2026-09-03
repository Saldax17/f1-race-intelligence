import httpx
import pytest
import respx

from f1_race_intelligence.config.settings import OpenF1Settings, RetrySettings
from f1_race_intelligence.ingestion.client import BaseAPIClient
from f1_race_intelligence.ingestion.exceptions import (
    OpenF1ConnectionError,
    OpenF1Error,
    OpenF1HTTPError,
    OpenF1RateLimitError,
    OpenF1TimeoutError,
)

BASE_URL = "https://example.test/v1"


def make_client(max_attempts: int = 3, backoff_factor: float = 1.0) -> BaseAPIClient:
    settings = OpenF1Settings(
        base_url=BASE_URL,
        retry=RetrySettings(max_attempts=max_attempts, backoff_factor=backoff_factor),
    )
    # Rate limits high enough that the limiter never has to sleep in tests.
    settings.rate_limit.requests_per_second = 1000
    settings.rate_limit.requests_per_minute = 1000
    return BaseAPIClient(settings)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Keep retry-with-backoff tests fast and deterministic."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@respx.mock
def test_get_builds_url_with_base_and_path_and_returns_json():
    route = respx.get(f"{BASE_URL}/sessions").mock(
        return_value=httpx.Response(200, json=[{"session_key": 1}])
    )
    client = make_client()

    result = client._get("/sessions", {"year": 2025})

    assert route.called
    assert route.calls.last.request.url.params["year"] == "2025"
    assert result == [{"session_key": 1}]


@respx.mock
def test_retries_on_500_then_succeeds():
    route = respx.get(f"{BASE_URL}/laps").mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json={"ok": True})]
    )
    client = make_client(max_attempts=3)

    result = client._get("/laps")

    assert result == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_exhausts_retries_and_raises_http_error():
    route = respx.get(f"{BASE_URL}/laps").mock(return_value=httpx.Response(503))
    client = make_client(max_attempts=3)

    with pytest.raises(OpenF1HTTPError) as exc_info:
        client._get("/laps")

    assert exc_info.value.status_code == 503
    assert route.call_count == 3


@respx.mock
def test_non_retryable_http_error_raises_immediately():
    route = respx.get(f"{BASE_URL}/laps").mock(return_value=httpx.Response(404))
    client = make_client(max_attempts=3)

    with pytest.raises(OpenF1HTTPError) as exc_info:
        client._get("/laps")

    assert exc_info.value.status_code == 404
    assert route.call_count == 1  # not retried


@respx.mock
def test_429_raises_rate_limit_error_after_retries_exhausted():
    respx.get(f"{BASE_URL}/laps").mock(return_value=httpx.Response(429, headers={"Retry-After": "2"}))
    client = make_client(max_attempts=2)

    with pytest.raises(OpenF1RateLimitError) as exc_info:
        client._get("/laps")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 2.0


@respx.mock
def test_retry_after_header_overrides_exponential_backoff(monkeypatch):
    """A 429 with Retry-After must wait at least that long, not just the exponential backoff."""
    respx.get(f"{BASE_URL}/laps").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "45"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = make_client(max_attempts=2, backoff_factor=1.0)

    waits = []
    monkeypatch.setattr("time.sleep", waits.append)

    result = client._get("/laps")

    assert result == {"ok": True}
    assert waits == [45.0]


@respx.mock
def test_timeout_is_translated_to_domain_exception():
    respx.get(f"{BASE_URL}/laps").mock(side_effect=httpx.ReadTimeout("timed out"))
    client = make_client(max_attempts=2)

    with pytest.raises(OpenF1TimeoutError):
        client._get("/laps")


@respx.mock
def test_connection_error_is_translated_to_domain_exception():
    respx.get(f"{BASE_URL}/laps").mock(side_effect=httpx.ConnectError("boom"))
    client = make_client(max_attempts=2)

    with pytest.raises(OpenF1ConnectionError):
        client._get("/laps")


@respx.mock
def test_invalid_json_raises_domain_error():
    respx.get(f"{BASE_URL}/laps").mock(return_value=httpx.Response(200, content=b"not json"))
    client = make_client(max_attempts=1)

    with pytest.raises(OpenF1Error):
        client._get("/laps")
