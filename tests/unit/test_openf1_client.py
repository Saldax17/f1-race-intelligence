from unittest.mock import MagicMock

import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.ingestion.openf1 import F1Client


@pytest.fixture
def client() -> F1Client:
    instance = F1Client(settings=AppSettings())
    instance._get = MagicMock(return_value=[])  # type: ignore[method-assign]
    return instance


def test_get_sessions_builds_expected_params_and_path(client: F1Client) -> None:
    client.get_sessions(year=2025)

    client._get.assert_called_once_with("/sessions", {"year": 2025})


def test_get_laps_builds_expected_params_and_path(client: F1Client) -> None:
    client.get_laps(session_key=12345, driver_number=1)

    client._get.assert_called_once_with("/laps", {"session_key": 12345, "driver_number": 1})


def test_get_car_data_builds_expected_params_and_path(client: F1Client) -> None:
    client.get_car_data(session_key=12345, driver_number=1)

    client._get.assert_called_once_with("/car_data", {"session_key": 12345, "driver_number": 1})


def test_get_car_data_accepts_date_range_filter_without_driver(client: F1Client) -> None:
    client.get_car_data(session_key=12345, **{"date>": "2023-09-15T13:00:00", "date<": "2023-09-15T13:05:00"})

    client._get.assert_called_once_with(
        "/car_data",
        {"session_key": 12345, "date>": "2023-09-15T13:00:00", "date<": "2023-09-15T13:05:00"},
    )


def test_get_car_data_without_driver_or_date_filter_raises(client: F1Client) -> None:
    with pytest.raises(ValueError):
        client.get_car_data(session_key=12345)

    client._get.assert_not_called()


def test_none_filters_are_dropped(client: F1Client) -> None:
    client.get_meetings()

    client._get.assert_called_once_with("/meetings", {})


def test_extra_filters_are_passed_through(client: F1Client) -> None:
    client.get_drivers(session_key=1, full_name="Max Verstappen")

    client._get.assert_called_once_with("/drivers", {"session_key": 1, "full_name": "Max Verstappen"})


@pytest.mark.parametrize(
    "method_name, expected_path",
    [
        ("get_meetings", "/meetings"),
        ("get_sessions", "/sessions"),
        ("get_drivers", "/drivers"),
        ("get_laps", "/laps"),
        ("get_positions", "/position"),
        ("get_intervals", "/intervals"),
        ("get_stints", "/stints"),
        ("get_pit", "/pit"),
        ("get_weather", "/weather"),
        ("get_race_control", "/race_control"),
    ],
)
def test_every_endpoint_method_calls_get_with_its_path(client: F1Client, method_name: str, expected_path: str) -> None:
    getattr(client, method_name)()

    client._get.assert_called_once_with(expected_path, {})
