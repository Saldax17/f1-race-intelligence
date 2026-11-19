"""FastAPI prototype for local inference (M8).

Run it with the locked ``training`` and ``serve`` extras installed::

    F1_SERVING_PREPROCESSING_URI=runs:/<M7_run_id>/m6 \
    F1_SERVING_MODEL_URI=models:/<registered-name>/<numeric-version> \
    uvicorn f1_race_intelligence.serving.app:app

The service loads once, at startup. ``/live`` reports process liveness;
``/health`` reports model readiness without returning internal load errors.
If loading fails, ``/predict`` refuses requests and a correlation ID connects
the generic HTTP response to the technical server log.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from f1_race_intelligence.serving.contract import ContractError
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import DEFAULT_MAX_RECORDS, PredictionError, PredictionService
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_URI_ENV = "F1_SERVING_MODEL_URI"
PREPROCESSING_URI_ENV = "F1_SERVING_PREPROCESSING_URI"
MAX_RECORDS_ENV = "F1_SERVING_MAX_RECORDS"


class PredictRequest(BaseModel):
    """Feature records, one per lap, keyed by M6 input feature name."""

    model_config = ConfigDict(extra="forbid")
    records: List[Dict[str, Any]] = Field(min_length=1)


def service_from_environment() -> PredictionService:
    """The default factory: configuration from environment variables only."""
    missing = [name for name in (PREPROCESSING_URI_ENV, MODEL_URI_ENV) if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"set {' and '.join(missing)}")
    return PredictionService.load(
        preprocessing_uri=os.environ[PREPROCESSING_URI_ENV].strip(),
        model_uri=os.environ[MODEL_URI_ENV].strip(),
        loader=MlflowPyfuncLoader(),
        max_records=int(os.getenv(MAX_RECORDS_ENV, "").strip() or DEFAULT_MAX_RECORDS),
    )


def create_app(service_factory: Optional[Callable[[], PredictionService]] = None) -> FastAPI:
    factory = service_factory or service_from_environment

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service, app.state.load_error_id = None, None
        try:
            app.state.service = factory()
        except Exception:  # reported by /health, never swallowed silently
            app.state.load_error_id = uuid.uuid4().hex
            logger.exception(
                "serving_load_failed error_id=%s",
                app.state.load_error_id,
                extra={"error_id": app.state.load_error_id},
            )
        yield

    app = FastAPI(
        title="F1 Race Intelligence - next lap time",
        version="0.1.0",
        description="Local prototype: M6 preprocessing plus a model registered by M7.",
        lifespan=lifespan,
    )

    def unavailable(request: Request) -> JSONResponse:
        error_id = request.app.state.load_error_id
        if error_id is None:
            error_id = uuid.uuid4().hex
            request.app.state.load_error_id = error_id
            logger.error("serving_not_ready", extra={"error_id": error_id})
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "Model service is not ready.",
                "error_id": error_id,
            },
        )

    @app.get("/live")
    def live():
        return {"status": "ok"}

    @app.get("/health")
    def health(request: Request):
        if request.app.state.service is None:
            return unavailable(request)
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info(request: Request):
        service = request.app.state.service
        if service is None:
            return unavailable(request)
        return service.describe()

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request):
        service: Optional[PredictionService] = request.app.state.service
        if service is None:
            return unavailable(request)
        try:
            values = service.predict(body.records)
        except ContractError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors})
        except PredictionError as exc:
            logger.error("serving_prediction_failed", extra={"error": str(exc)})
            return JSONResponse(status_code=500, content={"detail": f"invalid model output: {exc}"})
        contract = service.contract
        return {
            "target": contract.target,
            "model": {"uri": service.model.uri, "run_id": service.model.run_id},
            "predictions": [
                {"index": index, "value": value, "keys": contract.keys_of(record)}
                for index, (value, record) in enumerate(zip(values, body.records))
            ],
        }

    return app


app = create_app()
