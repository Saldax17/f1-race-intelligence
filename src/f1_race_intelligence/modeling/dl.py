"""Tabular deep learning for M7: a small PyTorch MLP over the M6 matrix.

The network is a plain feed-forward regressor trained on the encoded M6
feature matrix (``preprocessing.output_feature_names``). It exists to give
M7 a real deep-learning model (mini-batches, Huber/MSE loss, early stopping
on the validation split, per-epoch curves) without changing the M6 → M7 →
MLflow → M8 contract:

* the target is optionally standardised with statistics computed on the
  finite **training** targets only; validation is used for early stopping
  and never for scaling, and test is never seen here;
* the inverse target transformation lives inside the model, so
  :meth:`FittedTabularMLP.predict` and the registered ``mlflow.pyfunc``
  model always return ``next_lap_time`` in seconds;
* the registered pyfunc accepts the ``float64`` DataFrame M8 sends, selects
  the declared feature columns in order and returns a 1-D ``float64`` array.

PyTorch and MLflow are optional (the ``dl`` and ``training`` extras), so
neither is imported at module level: the M1–M6 runtime must be able to
import this module without them. The only exception is the guarded import
of ``mlflow.pyfunc.PythonModel`` below, which lets the pyfunc wrapper be a
top-level class that cloudpickle serialises by reference.
"""

from __future__ import annotations

import math
import random
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised implicitly by both CI environments
    from mlflow.pyfunc import PythonModel
except ImportError:  # the M1-M6 runtime has no MLflow

    class PythonModel:  # type: ignore[no-redef]
        """Stand-in base class so this module imports without MLflow."""


__all__ = [
    "MLPConfig",
    "MLPEpochMetrics",
    "FittedTabularMLP",
    "TabularMLPPyfuncModel",
    "train_tabular_mlp",
    "save_mlp_checkpoint",
    "load_mlp_checkpoint",
    "log_mlp_pyfunc_model",
]

TORCH_INSTALL_HINT = 'python -m pip install -e ".[training,dl]"'
CHECKPOINT_FILENAME = "mlp_checkpoint.pt"
CHECKPOINT_FORMAT_VERSION = 1
MIN_TARGET_STD = 1e-8
_EVAL_BATCH_SIZE = 8192
_CHECKPOINT_KEYS = (
    "format_version",
    "state_dict",
    "feature_names",
    "hidden_dims",
    "dropout",
    "y_mean",
    "y_std",
    "scale_target",
)


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters of the tabular MLP and its training loop."""

    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    loss: Literal["huber", "mse"] = "huber"
    scale_target: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class MLPEpochMetrics:
    """Mean losses of one epoch, in the (possibly scaled) training target units."""

    epoch: int
    train_loss: float
    val_loss: Optional[float]


@dataclass
class FittedTabularMLP:
    """A trained MLP: best weights, target scaling and the training curve."""

    feature_names: Tuple[str, ...]
    config: MLPConfig
    state_dict: Dict[str, Any]
    y_mean: float
    y_std: float
    best_epoch: int
    epochs_trained: int
    best_val_loss: Optional[float]
    total_params: int
    history: List[MLPEpochMetrics]
    _network: Any = field(default=None, init=False, repr=False, compare=False)

    def predict(self, features: Any) -> np.ndarray:
        """Predict in the original target scale.

        Accepts a DataFrame containing every name in ``feature_names`` (extra
        columns are ignored, order is taken from ``feature_names``) or a 2-D
        array whose columns already follow ``feature_names``. Returns a 1-D
        ``float64`` array of length ``N``.
        """
        torch = _require_torch()
        if self._network is None:
            network = _build_network(torch, len(self.feature_names), self.config.hidden_dims, self.config.dropout)
            network.load_state_dict(self.state_dict)
            network.eval()
            self._network = network
        matrix = _feature_matrix(features, self.feature_names, label="features")
        return _predict_original_scale(torch, self._network, matrix, self.y_mean, self.y_std)


class TabularMLPPyfuncModel(PythonModel):
    """``mlflow.pyfunc`` wrapper that serves a :class:`FittedTabularMLP` checkpoint.

    The instance carries no state when it is pickled; ``load_context`` rebuilds
    the network from the ``checkpoint`` artifact on load.
    """

    def __init__(self) -> None:
        self._network: Any = None
        self._feature_names: Tuple[str, ...] = ()
        self._y_mean = 0.0
        self._y_std = 1.0

    def load_context(self, context: Any) -> None:
        torch = _require_torch()
        checkpoint = load_mlp_checkpoint(Path(context.artifacts["checkpoint"]))
        feature_names = tuple(checkpoint["feature_names"])
        network = _build_network(torch, len(feature_names), tuple(checkpoint["hidden_dims"]), checkpoint["dropout"])
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        self._network = network
        self._feature_names = feature_names
        self._y_mean = float(checkpoint["y_mean"])
        self._y_std = float(checkpoint["y_std"])

    # MLflow 3 reads this ``model_input`` hint: ``pd.DataFrame`` keeps the input
    # untouched (no type-hint schema inference); the logged signature governs it.
    def predict(
        self, context: Any, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Return 1-D ``float64`` predictions in seconds for a feature DataFrame."""
        torch = _require_torch()
        if self._network is None:
            raise RuntimeError("the MLP pyfunc model has not been loaded; call load_context first")
        matrix = _feature_matrix(model_input, self._feature_names, label="model_input")
        return _predict_original_scale(torch, self._network, matrix, self._y_mean, self._y_std)


def train_tabular_mlp(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    *,
    feature_names: Sequence[str],
    config: MLPConfig,
) -> FittedTabularMLP:
    """Train the MLP on ``train`` with early stopping on the finite ``val`` targets.

    Rows of ``train`` without a finite target are dropped. The target scaling
    statistics come from the remaining training targets only. When ``val`` has
    no finite target, the best epoch is chosen on the training loss and
    ``best_val_loss`` is ``None``.
    """
    torch = _require_torch()
    config = _validated_config(config)
    names = _validated_feature_names(feature_names)

    train_matrix = _feature_matrix(train_x, names, label="train_x")
    train_targets = _target_vector(train_y, len(train_matrix), label="train_y")
    train_mask = np.isfinite(train_targets)
    if not np.any(train_mask):
        raise ValueError("train_y contains no finite targets")
    train_matrix = train_matrix[train_mask]
    train_targets = train_targets[train_mask]

    val_matrix_all = _feature_matrix(val_x, names, label="val_x", require_finite=False)
    val_targets_all = _target_vector(val_y, len(val_matrix_all), label="val_y")
    val_mask = np.isfinite(val_targets_all)
    val_matrix = val_matrix_all[val_mask]
    val_targets = val_targets_all[val_mask]
    if val_matrix.size and not np.isfinite(val_matrix).all():
        raise ValueError("val_x has missing or non-finite feature values in rows with a finite target")
    has_validation = len(val_targets) > 0

    # Anti-leakage: scaling statistics come from the training targets alone.
    if config.scale_target:
        y_mean = float(np.mean(train_targets))
        raw_std = float(np.std(train_targets))
        y_std = raw_std if raw_std >= MIN_TARGET_STD else 1.0
    else:
        y_mean, y_std = 0.0, 1.0

    seed = int(config.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        device = torch.device("cpu")
        network = _build_network(torch, len(names), config.hidden_dims, config.dropout).to(device)
        total_params = int(sum(parameter.numel() for parameter in network.parameters()))
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0) if config.loss == "huber" else torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        train_x_t = torch.from_numpy(np.ascontiguousarray(train_matrix, dtype=np.float32)).to(device)
        train_y_t = torch.from_numpy(
            np.ascontiguousarray((train_targets - y_mean) / y_std, dtype=np.float32)
        ).to(device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x_t, train_y_t),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        if has_validation:
            val_x_t = torch.from_numpy(np.ascontiguousarray(val_matrix, dtype=np.float32)).to(device)
            val_y_t = torch.from_numpy(
                np.ascontiguousarray((val_targets - y_mean) / y_std, dtype=np.float32)
            ).to(device)

        history: List[MLPEpochMetrics] = []
        best_monitored = math.inf
        best_epoch = 0
        best_state = _cpu_state_dict(network)
        epochs_without_improvement = 0
        epochs_trained = 0

        for epoch in range(1, config.max_epochs + 1):
            network.train()
            loss_sum = 0.0
            seen = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(network(batch_x).reshape(-1), batch_y)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * int(batch_y.shape[0])
                seen += int(batch_y.shape[0])
            train_loss = loss_sum / seen

            val_loss: Optional[float] = None
            if has_validation:
                val_loss = _evaluate_loss(torch, network, loss_fn, val_x_t, val_y_t)
            history.append(MLPEpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
            epochs_trained = epoch

            monitored = val_loss if val_loss is not None else train_loss
            if not math.isfinite(monitored):
                raise RuntimeError(f"MLP training diverged: non-finite loss at epoch {epoch}")
            if monitored < best_monitored - config.min_delta:
                best_monitored = monitored
                best_epoch = epoch
                best_state = _cpu_state_dict(network)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

        network.load_state_dict(best_state)
        network.eval()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)

    fitted = FittedTabularMLP(
        feature_names=names,
        config=config,
        state_dict=best_state,
        y_mean=y_mean,
        y_std=y_std,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        best_val_loss=float(best_monitored) if has_validation else None,
        total_params=total_params,
        history=history,
    )
    fitted._network = network
    return fitted


def save_mlp_checkpoint(fitted: FittedTabularMLP, path: Path) -> Path:
    """Write the weights and the metadata needed to rebuild the model.

    The file holds CPU tensors and plain Python values only, so it loads with
    ``torch.load(..., weights_only=True)``.
    """
    torch = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": {str(key): value.detach().cpu().clone() for key, value in fitted.state_dict.items()},
        "feature_names": [str(name) for name in fitted.feature_names],
        "hidden_dims": [int(dim) for dim in fitted.config.hidden_dims],
        "dropout": float(fitted.config.dropout),
        "y_mean": float(fitted.y_mean),
        "y_std": float(fitted.y_std),
        "scale_target": bool(fitted.config.scale_target),
    }
    torch.save(payload, target)
    return target


def load_mlp_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_mlp_checkpoint` onto the CPU."""
    torch = _require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"MLP checkpoint {path} is not a dictionary")
    missing = [key for key in _CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"MLP checkpoint {path} is missing keys: {missing}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported MLP checkpoint format version: {payload['format_version']!r}")
    return payload


def log_mlp_pyfunc_model(
    fitted: FittedTabularMLP,
    *,
    example: pd.DataFrame,
    registered_model_name: str,
    artifact_path: str = "model",
) -> None:
    """Log and register the MLP as an ``mlflow.pyfunc`` model in the active run.

    The signature is inferred from ``example`` and the model's own predictions,
    so the named input schema follows ``example``'s columns; pass the encoded
    feature columns in ``feature_names`` order.
    """
    torch = _require_torch()
    try:
        import cloudpickle
        import mlflow
        import mlflow.pyfunc
        from mlflow.models import infer_signature
    except ImportError as exc:
        raise RuntimeError(
            "logging the MLP requires the optional training dependencies; install "
            f"`{TORCH_INSTALL_HINT}` first"
        ) from exc

    if list(example.columns) != list(fitted.feature_names):
        raise ValueError("example columns must equal the fitted feature_names in the same order")
    signature = infer_signature(example, fitted.predict(example))
    pip_requirements = [
        f"mlflow=={mlflow.__version__}",
        f"torch=={torch.__version__.split('+', 1)[0]}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ]
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = save_mlp_checkpoint(fitted, Path(directory) / CHECKPOINT_FILENAME)
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=TabularMLPPyfuncModel(),
            artifacts={"checkpoint": str(checkpoint_path)},
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example,
            pip_requirements=pip_requirements,
        )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"the M7 MLP requires PyTorch (the optional `dl` extra); install `{TORCH_INSTALL_HINT}` first"
        ) from exc
    return torch


def _build_network(torch: Any, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> Any:
    """``Linear → ReLU (→ Dropout)`` per hidden layer, then ``Linear(…, 1)``."""
    layers: List[Any] = []
    previous = int(input_dim)
    for dim in hidden_dims:
        layers.append(torch.nn.Linear(previous, int(dim)))
        layers.append(torch.nn.ReLU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        previous = int(dim)
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _cpu_state_dict(network: Any) -> Dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def _evaluate_loss(torch: Any, network: Any, loss_fn: Any, features: Any, targets: Any) -> float:
    network.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, int(targets.shape[0]), _EVAL_BATCH_SIZE):
            batch_x = features[start:start + _EVAL_BATCH_SIZE]
            batch_y = targets[start:start + _EVAL_BATCH_SIZE]
            total += float(loss_fn(network(batch_x).reshape(-1), batch_y).item()) * int(batch_y.shape[0])
    return total / int(targets.shape[0])


def _predict_original_scale(torch: Any, network: Any, matrix: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    network.eval()
    inputs = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    with torch.no_grad():
        scaled = network(inputs).reshape(-1).cpu().numpy()
    return scaled.astype(np.float64) * float(y_std) + float(y_mean)


def _feature_matrix(
    features: Any,
    feature_names: Sequence[str],
    *,
    label: str,
    require_finite: bool = True,
) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"{label} is missing required feature columns: {missing}")
        selected = features.loc[:, list(feature_names)]
        try:
            matrix = selected.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} feature columns must be numeric") from exc
    else:
        try:
            matrix = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a DataFrame or a numeric 2-D array") from exc
        if matrix.ndim != 2:
            raise ValueError(f"{label} must be 2-D, got shape {matrix.shape}")
        if matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"{label} has {matrix.shape[1]} columns but the model expects {len(feature_names)}"
            )
    if require_finite and not np.isfinite(matrix).all():
        raise ValueError(f"{label} has missing or non-finite feature values")
    return matrix.astype(np.float32)


def _target_vector(values: Any, expected_rows: int, *, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if len(vector) != expected_rows:
        raise ValueError(f"{label} has {len(vector)} rows but its features have {expected_rows}")
    return vector


def _validated_feature_names(feature_names: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(feature_names, str):
        raise ValueError("feature_names must be a sequence of column names, not a string")
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    return names


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _validated_config(config: MLPConfig) -> MLPConfig:
    if not isinstance(config, MLPConfig):
        raise TypeError("config must be an MLPConfig")
    if isinstance(config.hidden_dims, (str, bytes)):
        raise ValueError("hidden_dims must be a sequence of positive integers")
    try:
        hidden_dims = tuple(config.hidden_dims)
    except TypeError as exc:
        raise ValueError("hidden_dims must be a sequence of positive integers") from exc
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    if any(not _is_int(dim) or dim <= 0 for dim in hidden_dims):
        raise ValueError(f"hidden_dims must be positive integers, got {hidden_dims!r}")
    checks = (
        (_is_real(config.dropout) and 0.0 <= config.dropout < 1.0, "dropout must be in [0.0, 1.0)"),
        (_is_real(config.learning_rate) and math.isfinite(config.learning_rate) and config.learning_rate > 0,
         "learning_rate must be a finite number > 0"),
        (_is_real(config.weight_decay) and math.isfinite(config.weight_decay) and config.weight_decay >= 0,
         "weight_decay must be a finite number >= 0"),
        (_is_int(config.batch_size) and config.batch_size >= 1, "batch_size must be an integer >= 1"),
        (_is_int(config.max_epochs) and config.max_epochs >= 1, "max_epochs must be an integer >= 1"),
        (_is_int(config.early_stopping_patience) and config.early_stopping_patience >= 1,
         "early_stopping_patience must be an integer >= 1"),
        (_is_real(config.min_delta) and math.isfinite(config.min_delta) and config.min_delta >= 0,
         "min_delta must be a finite number >= 0"),
        (config.loss in ("huber", "mse"), "loss must be 'huber' or 'mse'"),
        (isinstance(config.scale_target, (bool, np.bool_)), "scale_target must be a boolean"),
        (_is_int(config.random_seed), "random_seed must be an integer"),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(f"{message}; got {config!r}")
    return replace(
        config,
        hidden_dims=tuple(int(dim) for dim in hidden_dims),
        dropout=float(config.dropout),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        batch_size=int(config.batch_size),
        max_epochs=int(config.max_epochs),
        early_stopping_patience=int(config.early_stopping_patience),
        min_delta=float(config.min_delta),
        scale_target=bool(config.scale_target),
        random_seed=int(config.random_seed),
    )
