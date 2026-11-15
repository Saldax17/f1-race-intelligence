"""HTTP layer of the serving prototype (M8).

Skipped when FastAPI is not installed (it is the optional ``serve`` extra).
The M6 artifacts come from the real M6 runner on synthetic races; the model
is a recording stub.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from f1_race_intelligence.serving import app as app_module  # noqa: E402
from f1_race_intelligence.serving.app import create_app  # noqa: E402
from f1_race_intelligence.serving.service import PredictionService  # noqa: E402
from serving_support import RecordingModel, StubLoader, build_m6_run, to_records  # noqa: E402

MODEL_URI = "models:/example/1"


@pytest.fixture(scope="module")
def m6_run(tmp_path_factory):
    return build_m6_run(tmp_path_factory.mktemp("m6"))


def client_for(m6_run, model=None, max_records=1000) -> TestClient:
    def factory() -> PredictionService:
        return PredictionService.load(
            preprocessing_uri=str(m6_run.preprocessing_dir),
            model_uri=MODEL_URI,
            loader=StubLoader(model or RecordingModel()),
            max_records=max_records,
        )

    return TestClient(create_app(factory))


def records(m6_run, count=2):
    rows = m6_run.split_file("test", "dataset.parquet").head(count)
    return to_records(rows, m6_run.numeric + m6_run.categorical)


def test_health_and_model_info_when_loaded(m6_run) -> None:
    with client_for(m6_run) as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/model-info").json()

    assert info["model"] == {
        "uri": MODEL_URI,
        "run_id": "stub-run",
        "flavors": ["python_function"],
        "input_signature_verified": False,
    }
    assert info["preprocessing"]["uri"] == str(m6_run.preprocessing_dir)
    assert info["preprocessing"]["target"] == "next_lap_time"
    assert info["preprocessing"]["numeric_features"] == list(m6_run.numeric)
    assert info["preprocessing"]["categorical_features"] == list(m6_run.categorical)


def test_predict_returns_one_value_per_record_with_its_keys(m6_run) -> None:
    body = records(m6_run, 2)
    body[0]["session_key"], body[0]["driver_number"] = 3, 1

    with client_for(m6_run) as client:
        response = client.post("/predict", json={"records": body})

    assert response.status_code == 200
    assert response.json() == {
        "target": "next_lap_time",
        "model": {"uri": MODEL_URI, "run_id": "stub-run"},
        "predictions": [
            {"index": 0, "value": 80.0, "keys": {"session_key": 3, "driver_number": 1, "lap_number": body[0]["lap_number"]}},
            {"index": 1, "value": 81.0, "keys": {"lap_number": body[1]["lap_number"]}},
        ],
    }


def test_contract_violations_are_422_with_every_problem_listed(m6_run) -> None:
    bad = {**records(m6_run, 1)[0], "next_lap_time": 90.0, "compound": 7}

    with client_for(m6_run) as client:
        response = client.post("/predict", json={"records": [bad]})

    assert response.status_code == 422
    fields = {error["field"] for error in response.json()["detail"]}
    assert fields == {"next_lap_time", "compound"}


@pytest.mark.parametrize("payload", [{}, {"records": []}, {"records": [{}], "extra": 1}, {"records": "laps"}])
def test_malformed_bodies_are_422(m6_run, payload) -> None:
    with client_for(m6_run) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_too_many_records_are_422(m6_run) -> None:
    with client_for(m6_run, max_records=1) as client:
        response = client.post("/predict", json={"records": records(m6_run, 2)})

    assert response.status_code == 422
    assert "at most 1 records" in response.json()["detail"][0]["error"]


def test_invalid_model_output_is_500(m6_run) -> None:
    with client_for(m6_run, RecordingModel(output=[float("nan"), 1.0])) as client:
        response = client.post("/predict", json={"records": records(m6_run, 2)})

    assert response.status_code == 500
    assert "non-finite" in response.json()["detail"]


def test_a_service_that_failed_to_load_reports_why_and_serves_nothing() -> None:
    def broken() -> PredictionService:
        raise RuntimeError("set F1_SERVING_PREPROCESSING_URI and F1_SERVING_MODEL_URI")

    with TestClient(create_app(broken)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert "F1_SERVING_MODEL_URI" in health.json()["reason"]
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json={"records": [{}]}).status_code == 503


def test_the_default_app_needs_both_locations(monkeypatch) -> None:
    monkeypatch.delenv(app_module.MODEL_URI_ENV, raising=False)
    monkeypatch.setenv(app_module.PREPROCESSING_URI_ENV, "somewhere")

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert app_module.MODEL_URI_ENV in response.json()["reason"]
