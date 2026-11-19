"""Loading the model to serve, behind an MLflow-compatible interface.

The service depends on :class:`ModelLoader`, not on MLflow. The production
loader goes through ``mlflow.pyfunc``, MLflow's flavour-independent inference
interface, so whatever M7 logs (scikit-learn, boosting, a neural network)
and however it is addressed (``models:/<name>/<version>``,
``models:/<name>@<alias>``, ``runs:/<run_id>/<path>``) loads the same way.
The integrated M7 serving path further requires an exact numeric registry
version and artifacts from that model's M7 run.
MLflow is imported only when that loader runs: the rest of the package, and
its tests, work without it.

Loading a model deserialises code (pickle under most flavours). Only point
the service at a registry or run you trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Tuple


class ModelLoadError(RuntimeError):
    """The model could not be loaded."""


class PredictiveModel(Protocol):
    """What the service needs from a model: ``predict`` on a DataFrame."""

    def predict(self, data: Any) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class LoadedModel:
    """A model plus what the service reports and checks about it."""

    model: PredictiveModel
    uri: str
    run_id: Optional[str] = None
    flavors: Tuple[str, ...] = field(default_factory=tuple)
    # Column names from the model signature, when it has a named input
    # schema. The service refuses to start if they differ from M6's encoded
    # feature names; None means the model declared no named columns.
    input_names: Optional[Tuple[str, ...]] = None


class ModelLoader(Protocol):
    def load(self, uri: str) -> LoadedModel:  # pragma: no cover - protocol
        ...


class MlflowPyfuncLoader:
    """Load through ``mlflow.pyfunc.load_model``.

    The tracking server and registry come from MLflow's own configuration
    (``MLFLOW_TRACKING_URI`` and friends); nothing is configured here.
    """

    def load(self, uri: str) -> LoadedModel:
        try:
            from mlflow import pyfunc
        except ImportError as exc:
            raise ModelLoadError(
                'Loading a registered model needs the optional dependencies; install '
                '`python -m pip install -e ".[training,serve]"` from the project checkout'
            ) from exc

        try:
            model = pyfunc.load_model(uri)
        except Exception as exc:  # MLflow raises many types; the cause is kept
            raise ModelLoadError(f"MLflow could not load {uri!r}: {exc}") from exc

        metadata = getattr(model, "metadata", None)
        return LoadedModel(
            model=model,
            uri=uri,
            run_id=getattr(metadata, "run_id", None),
            flavors=tuple(sorted((getattr(metadata, "flavors", None) or {}).keys())),
            input_names=_signature_input_names(metadata),
        )


def _signature_input_names(metadata: Any) -> Optional[Tuple[str, ...]]:
    """Named input columns of an MLflow signature, or None if there are none."""
    get_schema = getattr(metadata, "get_input_schema", None)
    schema = get_schema() if callable(get_schema) else None
    if schema is None or not getattr(schema, "has_input_names", lambda: True)():
        return None
    names = list(schema.input_names())
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return tuple(names)
