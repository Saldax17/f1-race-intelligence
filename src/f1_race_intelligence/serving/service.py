"""Prediction service: M6's fitted preprocessing, then the registered model.

The service owns no transformation. A request is checked against the
contract, shaped like M5's output, and then passed through M6's own
``prepare_matrix`` and the ``preprocessor.joblib`` M6 fitted on the training
races. The model receives exactly the encoded matrix M6 writes to
``features.parquet``, which is what M7 trains on.

Three artifacts have to belong together: the feature schema, the fitted
preprocessor and the model. M7's MLflow run URI binds the first two to the
model's run ID and records their M6 hashes; the service checks that identity
and content before it accepts a single request.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np

from f1_race_intelligence.modeling.model_package import ModelPackageError, read_model_package
from f1_race_intelligence.modeling import preprocessing
from f1_race_intelligence.modeling.runner import PREPROCESSOR_FILE
from f1_race_intelligence.serving.contract import FEATURE_SCHEMA_FILE, ArtifactError, ContractError, FeatureContract
from f1_race_intelligence.serving.model import LoadedModel, MlflowPyfuncLoader, ModelLoader
from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.store import store_from_uri
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RECORDS = 1000
_MLFLOW_PREPROCESSING_URI = re.compile(r"^runs:/([^/]+)/m6/?$")


class PredictionError(RuntimeError):
    """The model returned something that is not one finite value per record."""


@dataclass(frozen=True)
class PredictionService:
    contract: FeatureContract
    preprocessor: Any
    model: LoadedModel
    preprocessing_uri: str
    max_records: int = DEFAULT_MAX_RECORDS
    m6_run_id: Optional[str] = None
    input_fingerprint: Optional[str] = None

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

        ``preprocessing_uri`` may be ``runs:/<M7_run_id>/m6``, which downloads
        the preprocessor and schema
        associated with the model, and verifies their M7 provenance hashes.
        A local or ``s3://`` M6 preprocessing run directory remains supported
        for the standalone prototype. The preprocessor is a pickle: only point
        this at artifacts you trust.
        """
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        if preprocessing_uri.startswith("runs:/") and not _is_exact_model_version(model_uri):
            raise ArtifactError(
                "M7 serving requires an exact registered-model URI "
                "models:/<registered_name>/<numeric_version>; aliases and latest are not accepted"
            )
        m7_run_id, artifact_directory, provenance = _resolve_m7_artifacts(preprocessing_uri)
        if isinstance(loader, MlflowPyfuncLoader) and m7_run_id is None:
            raise ArtifactError("MlflowPyfuncLoader requires preprocessing artifacts from the same M7 run")
        backend = store_from_uri(artifact_directory).backend
        try:
            schema_bytes = backend.read_bytes(FEATURE_SCHEMA_FILE)
            preprocessor_bytes = backend.read_bytes(PREPROCESSOR_FILE)
            if provenance is not None:
                _verify_m7_artifact_hash(provenance, FEATURE_SCHEMA_FILE, schema_bytes)
                _verify_m7_artifact_hash(provenance, PREPROCESSOR_FILE, preprocessor_bytes)
            schema = json.loads(schema_bytes.decode("utf-8"))
            fitted = joblib.load(io.BytesIO(preprocessor_bytes))
        except FileNotFoundError as exc:
            raise ArtifactError(
                f"{preprocessing_uri!r} is not an M6 preprocessing run directory "
                f"(needs {PREPROCESSOR_FILE} and {FEATURE_SCHEMA_FILE}): {exc}"
            ) from exc

        contract = FeatureContract.from_schema(schema)
        _check_preprocessor_matches_schema(contract, fitted)
        model = loader.load(model_uri)
        if m7_run_id is not None and model.run_id != m7_run_id:
            raise ArtifactError(
                f"preprocessor artifacts belong to M7 run {m7_run_id!r}, "
                f"but model {model_uri!r} belongs to {model.run_id!r}"
            )
        if m7_run_id is not None and model.input_names is None:
            raise ArtifactError("M7 serving requires a named model input signature")
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
        return cls(
            contract,
            fitted,
            model,
            preprocessing_uri,
            max_records,
            provenance.get("m6_run_id") if provenance is not None else None,
            provenance.get("input_fingerprint") if provenance is not None else None,
        )

    @classmethod
    def load_published_package(
        cls,
        *,
        backend: StorageBackend,
        prefix: str,
        loader: ModelLoader,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> "PredictionService":
        """Load a verified published package without consulting MLflow.

        The model is materialized to a temporary directory only while the
        loader reads the MLflow pyfunc bundle. Its in-memory model, M6
        preprocessor, and schema are then retained by the service.
        """
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        try:
            package = read_model_package(backend, prefix)
        except ModelPackageError as exc:
            raise ArtifactError(f"published model package failed verification: {exc}") from exc

        model_metadata = package.manifest["model"]
        provenance_metadata = package.manifest["provenance"]
        try:
            provenance = json.loads(package.artifacts["data_provenance.json"].decode("utf-8"))
            report = json.loads(package.artifacts["training_report.json"].decode("utf-8"))
            schema_bytes = package.artifacts[f"m6/{FEATURE_SCHEMA_FILE}"]
            schema = json.loads(schema_bytes.decode("utf-8"))
            preprocessor_bytes = package.artifacts[f"m6/{PREPROCESSOR_FILE}"]
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"published model package has invalid M6/M7 metadata: {exc}") from exc

        if (
            not isinstance(provenance, dict)
            or provenance.get("m6_run_id") != provenance_metadata["m6_run_id"]
            or provenance.get("input_fingerprint") != provenance_metadata["input_fingerprint"]
            or not isinstance(report, dict)
            or report.get("run_id") != model_metadata["m7_run_id"]
            or report.get("m6_run_id") != provenance_metadata["m6_run_id"]
            or report.get("input_fingerprint") != provenance_metadata["input_fingerprint"]
            or not isinstance(report.get("registered_model"), dict)
            or report["registered_model"].get("uri") != model_metadata["uri"]
        ):
            raise ArtifactError("published package provenance does not bind the M6 artifacts to this M7 model")
        _verify_m7_artifact_hash(provenance, FEATURE_SCHEMA_FILE, schema_bytes)
        _verify_m7_artifact_hash(provenance, PREPROCESSOR_FILE, preprocessor_bytes)

        contract = FeatureContract.from_schema(schema)
        if contract.target != model_metadata["target"]:
            raise ArtifactError("published package target differs from the M6 feature schema")
        try:
            fitted = joblib.load(io.BytesIO(preprocessor_bytes))
        except Exception as exc:
            raise ArtifactError(f"{PREPROCESSOR_FILE} in the published package could not be loaded: {exc}") from exc
        _check_preprocessor_matches_schema(contract, fitted)

        with tempfile.TemporaryDirectory(prefix="f1-model-package-") as directory:
            model_directory = Path(directory) / "model"
            for name, content in package.artifacts.items():
                if not name.startswith("model/"):
                    continue
                relative = Path(*name.split("/")[1:])
                destination = model_directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            loaded = loader.load(str(model_directory))

        if loaded.run_id != model_metadata["m7_run_id"]:
            raise ArtifactError(
                "published model artifact belongs to M7 run "
                f"{loaded.run_id!r}, not manifest run {model_metadata['m7_run_id']!r}"
            )
        if loaded.input_names is None:
            raise ArtifactError("published M7 model requires a named input signature")
        if list(loaded.input_names) != model_metadata["input_signature"]:
            raise ArtifactError("loaded model signature differs from the published package manifest")
        model = replace(loaded, uri=model_metadata["uri"])
        _check_model_matches_schema(contract, model)

        logger.info(
            "serving_loaded_published_package",
            extra={
                "package_uri": package.package_uri,
                "model_uri": model.uri,
                "model_run_id": model.run_id,
                "encoded_features": len(contract.encoded_feature_names),
                "input_signature_verified": True,
            },
        )
        return cls(
            contract,
            fitted,
            model,
            package.package_uri,
            max_records,
            provenance_metadata["m6_run_id"],
            provenance_metadata["input_fingerprint"],
        )

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
            "provenance": {
                "m7_run_id": self.model.run_id,
                "m6_run_id": self.m6_run_id,
                "input_fingerprint": self.input_fingerprint,
            },
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
    expected, declared = contract.encoded_feature_names, model.input_names
    if expected != declared:
        missing, extra = sorted(set(expected) - set(declared)), sorted(set(declared) - set(expected))
        order_differs = not missing and not extra
        raise ArtifactError(
            f"model {model.uri!r} has an input signature incompatible with this M6 run "
            f"({len(missing)} missing from the model, {len(extra)} unknown to M6, "
            f"order differs: {order_differs}; e.g. missing {missing[:3]}, "
            f"unknown {extra[:3]}): it was trained on another M6 run"
        )


def _resolve_m7_artifacts(preprocessing_uri: str):
    """Download the preprocessing artifacts and provenance from one M7 run."""
    if not preprocessing_uri.startswith("runs:/"):
        return None, preprocessing_uri, None
    match = _MLFLOW_PREPROCESSING_URI.fullmatch(preprocessing_uri)
    if match is None:
        raise ArtifactError("an MLflow preprocessing URI must have the form runs:/<M7_run_id>/m6")
    run_id = match.group(1)
    try:
        from mlflow.artifacts import download_artifacts

        artifact_directory = str(download_artifacts(artifact_uri=preprocessing_uri))
        provenance_path = download_artifacts(artifact_uri=f"runs:/{run_id}/data_provenance.json")
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactError(f"could not download M6 artifacts and provenance from M7 run {run_id!r}: {exc}") from exc
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("m6_run_id"), str)
        or not provenance["m6_run_id"]
        or not isinstance(provenance.get("input_fingerprint"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["input_fingerprint"])
    ):
        raise ArtifactError(f"M7 run {run_id!r} has no valid data_provenance.json")
    return run_id, artifact_directory, provenance


def _is_exact_model_version(uri: str) -> bool:
    if not uri.startswith("models:/"):
        return False
    registered_name, separator, version = uri[len("models:/") :].rpartition("/")
    return bool(separator and registered_name and version.isdecimal())


def _verify_m7_artifact_hash(provenance: Mapping[str, Any], artifact: str, content: bytes) -> None:
    hashes = provenance.get("verified_artifact_hashes")
    if not isinstance(hashes, list):
        raise ArtifactError("M7 provenance has no verified artifact hashes")
    declarations = [
        item
        for item in hashes
        if isinstance(item, dict) and item.get("artifact") == artifact and item.get("split") is None
    ]
    if len(declarations) != 1 or not isinstance(declarations[0].get("sha256"), str):
        raise ArtifactError(f"M7 provenance has no unique SHA-256 for {artifact}")
    expected = declarations[0]["sha256"]
    actual = hashlib.sha256(content).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise ArtifactError(f"{artifact} SHA-256 does not match M7 run provenance")


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
