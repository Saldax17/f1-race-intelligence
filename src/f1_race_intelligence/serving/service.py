"""Prediction service: M6's fitted preprocessing, then the registered model.

The service owns no transformation. A request is checked against the
contract, shaped like M5's output, and then passed through M6's own
``prepare_matrix`` and the ``preprocessor.joblib`` M6 fitted on the training
races. The model receives exactly the encoded matrix M6 writes to
``features.parquet``, which is what M7 trains on.

Three artifacts have to belong together: the feature schema, the fitted
preprocessor and the model. Serving a model with the preprocessing of a
different M6 run would silently shift every column, so the service checks
that they agree before it accepts a single request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from f1_race_intelligence.modeling import preprocessing
from f1_race_intelligence.modeling.runner import PREPROCESSOR_FILE
from f1_race_intelligence.serving.contract import FEATURE_SCHEMA_FILE, ArtifactError, ContractError, FeatureContract
from f1_race_intelligence.serving.model import LoadedModel, ModelLoader
from f1_race_intelligence.storage.store import store_from_uri
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RECORDS = 1000


class PredictionError(RuntimeError):
    """The model returned something that is not one finite value per record."""


@dataclass(frozen=True)
class PredictionService:
    contract: FeatureContract
    preprocessor: Any
    model: LoadedModel
    preprocessing_uri: str
    max_records: int = DEFAULT_MAX_RECORDS

    @classmethod
    def load(
        cls,
        *,
        preprocessing_uri: str,
        model_uri: str,
        loader: ModelLoader,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> "PredictionService":
        """Load and cross-check the M6 artifacts and the model.

        ``preprocessing_uri`` is the M6 run directory that holds
        ``preprocessor.joblib`` and ``feature_schema.json``
        (``.../modeling/preprocessing/runs/<run_id>``), local or ``s3://``.
        It is read through the project's storage layer, like every stage.
        The preprocessor is a pickle: only point this at M6 output you trust.
        """
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        backend = store_from_uri(preprocessing_uri).backend
        try:
            schema = backend.read_json(FEATURE_SCHEMA_FILE)
            fitted = backend.read_joblib(PREPROCESSOR_FILE)
        except FileNotFoundError as exc:
            raise ArtifactError(
                f"{preprocessing_uri!r} is not an M6 preprocessing run directory "
                f"(needs {PREPROCESSOR_FILE} and {FEATURE_SCHEMA_FILE}): {exc}"
            ) from exc

        contract = FeatureContract.from_schema(schema)
        _check_preprocessor_matches_schema(contract, fitted)
        model = loader.load(model_uri)
        _check_model_matches_schema(contract, model)

        logger.info(
            "serving_loaded",
            extra={
                "model_uri": model.uri,
                "model_run_id": model.run_id,
                "preprocessing_uri": preprocessing_uri,
                "encoded_features": len(contract.encoded_feature_names),
                "input_signature_verified": model.input_names is not None,
            },
        )
        return cls(contract, fitted, model, preprocessing_uri, max_records)

    def predict(self, records: Sequence[Mapping[str, Any]]) -> List[float]:
        """One prediction of the target per record, in request order."""
        if not records:
            raise ContractError([{"record": None, "field": "records", "error": "at least one record is required"}])
        if len(records) > self.max_records:
            raise ContractError(
                [{"record": None, "field": "records", "error": f"at most {self.max_records} records per request"}]
            )

        started = time.perf_counter()
        frame = self.contract.to_frame(records)
        encoded = self.encode(frame)
        values = _as_predictions(self.model.model.predict(encoded), len(records))
        logger.info(
            "serving_predicted",
            extra={"records": len(records), "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
        )
        return values

    def encode(self, frame) -> Any:
        """M6's transformation, with M6's code and M6's fitted transformer."""
        matrix = preprocessing.prepare_matrix(frame, self.contract.numeric, self.contract.categorical)
        return preprocessing.transform_to_frame(self.preprocessor, matrix)

    def describe(self) -> Dict[str, Any]:
        return {
            "model": {
                "uri": self.model.uri,
                "run_id": self.model.run_id,
                "flavors": list(self.model.flavors),
                "input_signature_verified": self.model.input_names is not None,
            },
            "preprocessing": {"uri": self.preprocessing_uri, **self.contract.describe()},
            "max_records": self.max_records,
        }


def _check_preprocessor_matches_schema(contract: FeatureContract, fitted: Any) -> None:
    try:
        consumed = [str(name) for name in fitted.feature_names_in_]
        produced = [str(name) for name in fitted.get_feature_names_out()]
    except AttributeError as exc:
        raise ArtifactError(f"{PREPROCESSOR_FILE} is not a fitted M6 preprocessor: {exc}") from exc
    if consumed != list(contract.features):
        raise ArtifactError(
            f"{PREPROCESSOR_FILE} was fitted on different input columns than {FEATURE_SCHEMA_FILE} lists: "
            "the two files come from different M6 runs"
        )
    if produced != list(contract.encoded_feature_names):
        raise ArtifactError(
            f"{PREPROCESSOR_FILE} produces different columns than {FEATURE_SCHEMA_FILE} lists: "
            "the two files come from different M6 runs"
        )


def _check_model_matches_schema(contract: FeatureContract, model: LoadedModel) -> None:
    if model.input_names is None:
        logger.warning(
            "serving_model_without_signature",
            extra={"model_uri": model.uri, "detail": "column agreement with M6 cannot be verified"},
        )
        return
    expected, declared = set(contract.encoded_feature_names), set(model.input_names)
    if expected != declared:
        missing, extra = sorted(expected - declared), sorted(declared - expected)
        raise ArtifactError(
            f"model {model.uri!r} expects different columns than this M6 run encodes "
            f"({len(missing)} missing from the model, {len(extra)} unknown to M6; "
            f"e.g. missing {missing[:3]}, unknown {extra[:3]}): it was trained on another M6 run"
        )


def _as_predictions(raw: Any, count: int) -> List[float]:
    try:
        values = np.asarray(raw, dtype="float64").reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"model output is not numeric: {exc}") from exc
    if values.size != count:
        raise PredictionError(f"model returned {values.size} value(s) for {count} record(s)")
    if not np.all(np.isfinite(values)):
        raise PredictionError("model returned a non-finite prediction")
    return [float(value) for value in values]
