from f1_race_intelligence.ingestion.exceptions import (
    OpenF1ConnectionError,
    OpenF1Error,
    OpenF1HTTPError,
    OpenF1RateLimitError,
    OpenF1TimeoutError,
)


def test_openf1_http_error_carries_status_and_endpoint() -> None:
    error = OpenF1HTTPError(status_code=500, message="boom", endpoint="/laps", body="server error")

    assert isinstance(error, OpenF1Error)
    assert error.status_code == 500
    assert error.endpoint == "/laps"
    assert error.body == "server error"
    assert str(error) == "boom"


def test_openf1_rate_limit_error_is_an_http_error_with_429() -> None:
    error = OpenF1RateLimitError(endpoint="/sessions", retry_after=1.5)

    assert isinstance(error, OpenF1HTTPError)
    assert error.status_code == 429
    assert error.retry_after == 1.5


def test_timeout_and_connection_errors_are_openf1_errors() -> None:
    assert isinstance(OpenF1TimeoutError("timeout"), OpenF1Error)
    assert isinstance(OpenF1ConnectionError("conn"), OpenF1Error)
