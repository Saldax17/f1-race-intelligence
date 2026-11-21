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

Training optionally reduces the learning rate on a validation-loss plateau
(``ReduceLROnPlateau``) and records the learning rate of every epoch. Two
post-hoc analyses complete the model: :func:`compute_permutation_importance`
(deterministic permutation feature importance on a held-out split, meant for
``validation``) and :func:`diagnose_generalization` (a transparent
train-vs-validation over/underfitting verdict that never reads ``test``).

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
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

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
    "PERMUTATION_COLLINEARITY_NOTE",
    "FeatureImportanceEntry",
    "PermutationImportanceResult",
    "compute_permutation_importance",
    "diagnose_generalization",
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
UNDERFITTING_TRAIN_RATIO_VS_BASELINE = 0.95
OVERFITTING_VAL_TO_TRAIN_MAE_RATIO = 1.25
_MIN_MAE_FOR_RATIO = 1e-12

PERMUTATION_COLLINEARITY_NOTE = (
    "Permutation importance shuffles one encoded column at a time. Correlated features "
    "(e.g. lap_duration, its rolling means and the sector durations) and the columns of a "
    "one-hot encoded category share signal: while one of them is shuffled the model can "
    "still read the same information from the others, so the importance of redundant "
    "columns may be split between them or underestimated individually. Read low values "
    "of correlated columns as 'redundant given the rest', not as 'irrelevant'."
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
    lr_scheduler: Literal["none", "reduce_on_plateau"] = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6


@dataclass(frozen=True)
class MLPEpochMetrics:
    """Mean losses of one epoch, in the (possibly scaled) training target units.

    ``learning_rate`` is the optimizer learning rate used during the epoch.
    """

    epoch: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: Optional[float] = None


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


@dataclass(frozen=True)
class FeatureImportanceEntry:
    """Increase of the error metrics (seconds) when one feature column is shuffled."""

    feature: str
    mae_increase_mean: float
    mae_increase_std: float
    rmse_increase_mean: float
    rmse_increase_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "mae_increase_mean": float(self.mae_increase_mean),
            "mae_increase_std": float(self.mae_increase_std),
            "rmse_increase_mean": float(self.rmse_increase_mean),
            "rmse_increase_std": float(self.rmse_increase_std),
        }


@dataclass(frozen=True)
class PermutationImportanceResult:
    """Permutation feature importance of a fitted MLP on one split.

    ``importances`` is sorted from the most to the least important feature
    (by mean MAE increase, then mean RMSE increase, then feature name).
    """

    split: str
    n_rows: int
    n_repeats: int
    random_seed: int
    baseline_mae: float
    baseline_rmse: float
    importances: List[FeatureImportanceEntry]
    collinearity_note: str = PERMUTATION_COLLINEARITY_NOTE

    def top(self, k: int = 10) -> List[FeatureImportanceEntry]:
        """Return the ``k`` most important features."""
        if not _is_int(k) or k < 0:
            raise ValueError(f"k must be an integer >= 0, got {k!r}")
        return list(self.importances[: int(k)])

    def to_dict(self, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        """JSON-serialisable view; ``top_k`` keeps only the most important features."""
        entries = self.importances if top_k is None else self.top(top_k)
        return {
            "method": "permutation_feature_importance",
            "split": self.split,
            "n_rows": int(self.n_rows),
            "n_repeats": int(self.n_repeats),
            "random_seed": int(self.random_seed),
            "baseline_mae": float(self.baseline_mae),
            "baseline_rmse": float(self.baseline_rmse),
            "n_features": len(self.importances),
            "importances": [entry.to_dict() for entry in entries],
            "collinearity_note": self.collinearity_note,
        }


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
        scheduler: Any = None
        if config.lr_scheduler == "reduce_on_plateau":
            # Same improvement rule as early stopping: an absolute ``min_delta`` drop.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience,
                threshold=config.min_delta,
                threshold_mode="abs",
                min_lr=config.lr_scheduler_min_lr,
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
            current_lr = float(optimizer.param_groups[0]["lr"])
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
            history.append(
                MLPEpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, learning_rate=current_lr)
            )
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
            if scheduler is not None:
                scheduler.step(monitored)

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


def compute_permutation_importance(
    fitted: FittedTabularMLP,
    features: Any,
    targets: Any,
    *,
    split: str = "validation",
    n_repeats: int = 5,
    random_seed: Optional[int] = None,
) -> PermutationImportanceResult:
    """Permutation feature importance of ``fitted`` on a held-out split.

    Each feature column is shuffled ``n_repeats`` times and the increase of MAE
    and RMSE (in seconds) over the unshuffled baseline is averaged. Rows
    without a finite target are dropped. Shuffling uses an isolated
    ``numpy.random.Generator`` seeded with ``random_seed`` (the model's seed by
    default), so the result is deterministic and the global RNG state is left
    untouched; the caller's ``features`` are never modified. Compute it on
    ``validation`` (never ``test``, which would leak evaluation data into model
    analysis). See :data:`PERMUTATION_COLLINEARITY_NOTE` for how to read it.
    """
    if not isinstance(fitted, FittedTabularMLP):
        raise TypeError("fitted must be a FittedTabularMLP")
    if not _is_int(n_repeats) or n_repeats < 1:
        raise ValueError(f"n_repeats must be an integer >= 1, got {n_repeats!r}")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if random_seed is not None and not _is_int(random_seed):
        raise ValueError(f"random_seed must be an integer or None, got {random_seed!r}")
    seed = int(fitted.config.random_seed if random_seed is None else random_seed)

    # ``_feature_matrix`` returns a new array, so the caller's data is never mutated.
    matrix_all = _feature_matrix(features, fitted.feature_names, label="features", require_finite=False)
    target_all = _target_vector(targets, len(matrix_all), label="targets")
    mask = np.isfinite(target_all)
    if not np.any(mask):
        raise ValueError("targets contain no finite values")
    valid_matrix = np.array(matrix_all[mask], dtype=np.float32, copy=True)
    valid_targets = target_all[mask]
    if not np.isfinite(valid_matrix).all():
        raise ValueError("features have missing or non-finite values in rows with a finite target")
    n_rows = int(len(valid_targets))

    baseline_mae, baseline_rmse = _mae_rmse(fitted.predict(valid_matrix), valid_targets)
    rng = np.random.default_rng(seed)
    entries: List[FeatureImportanceEntry] = []
    for column, name in enumerate(fitted.feature_names):
        original = valid_matrix[:, column].copy()
        mae_increases: List[float] = []
        rmse_increases: List[float] = []
        try:
            for _ in range(int(n_repeats)):
                valid_matrix[:, column] = original[rng.permutation(n_rows)]
                perm_mae, perm_rmse = _mae_rmse(fitted.predict(valid_matrix), valid_targets)
                mae_increases.append(perm_mae - baseline_mae)
                rmse_increases.append(perm_rmse - baseline_rmse)
        finally:
            valid_matrix[:, column] = original
        entries.append(
            FeatureImportanceEntry(
                feature=str(name),
                mae_increase_mean=float(np.mean(mae_increases)),
                mae_increase_std=float(np.std(mae_increases, ddof=0)),
                rmse_increase_mean=float(np.mean(rmse_increases)),
                rmse_increase_std=float(np.std(rmse_increases, ddof=0)),
            )
        )
    entries.sort(key=lambda entry: (-entry.mae_increase_mean, -entry.rmse_increase_mean, entry.feature))
    return PermutationImportanceResult(
        split=split,
        n_rows=n_rows,
        n_repeats=int(n_repeats),
        random_seed=seed,
        baseline_mae=baseline_mae,
        baseline_rmse=baseline_rmse,
        importances=entries,
    )


def diagnose_generalization(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    model_key: str = "mlp",
    baseline_key: str = "reference_mean",
    epochs_trained: int,
    best_epoch: int,
    max_epochs: int,
) -> Dict[str, Any]:
    """Transparent over/underfitting verdict from the ``train`` and ``validation`` metrics.

    ``metrics`` follows the M7 ``metrics.json`` layout
    (``metrics[split][model_key]["mae"]``). Only ``train`` and ``validation``
    are read; ``test`` never takes part, so the verdict cannot be tuned on it.

    * ``underfitting``: the model does not beat the mean baseline on
      validation, or its train MAE is within 95 % of the baseline's;
    * ``overfitting_risk``: validation MAE is at least 1.25x the train MAE;
    * ``balanced``: neither of the above;
    * ``insufficient_metrics``: train or validation MAE is missing.

    Pure Python/NumPy: it imports neither PyTorch nor MLflow.
    """
    train = metrics.get("train", {}) or {}
    validation = metrics.get("validation", {}) or {}

    train_mae = _metric_value(train, model_key, "mae")
    validation_mae = _metric_value(validation, model_key, "mae")
    train_rmse = _metric_value(train, model_key, "rmse")
    validation_rmse = _metric_value(validation, model_key, "rmse")
    baseline_train_mae = _metric_value(train, baseline_key, "mae")
    baseline_validation_mae = _metric_value(validation, baseline_key, "mae")

    val_minus_train_mae = (
        validation_mae - train_mae if train_mae is not None and validation_mae is not None else None
    )
    val_to_train_mae_ratio = (
        validation_mae / train_mae
        if train_mae is not None and validation_mae is not None and train_mae > _MIN_MAE_FOR_RATIO
        else None
    )
    val_vs_baseline_mae_delta = (
        validation_mae - baseline_validation_mae
        if validation_mae is not None and baseline_validation_mae is not None
        else None
    )

    if train_mae is None or validation_mae is None:
        status = "insufficient_metrics"
        missing = [label for label, value in (("train", train_mae), ("validation", validation_mae)) if value is None]
        explanation = (
            f"Cannot diagnose generalization: {model_key!r} MAE is missing for {' and '.join(missing)}."
        )
    elif (baseline_validation_mae is not None and validation_mae >= baseline_validation_mae) or (
        baseline_train_mae is not None
        and train_mae >= baseline_train_mae * UNDERFITTING_TRAIN_RATIO_VS_BASELINE
    ):
        status = "underfitting"
        reasons = []
        if baseline_validation_mae is not None and validation_mae >= baseline_validation_mae:
            reasons.append(
                f"validation MAE {validation_mae:.4f} does not beat the {baseline_key!r} baseline "
                f"({baseline_validation_mae:.4f})"
            )
        if baseline_train_mae is not None and train_mae >= baseline_train_mae * UNDERFITTING_TRAIN_RATIO_VS_BASELINE:
            reasons.append(
                f"train MAE {train_mae:.4f} is >= {UNDERFITTING_TRAIN_RATIO_VS_BASELINE:.2f}x the "
                f"{baseline_key!r} train MAE ({baseline_train_mae:.4f})"
            )
        explanation = "Underfitting: " + "; ".join(reasons) + "."
    elif val_to_train_mae_ratio is not None and val_to_train_mae_ratio >= OVERFITTING_VAL_TO_TRAIN_MAE_RATIO:
        status = "overfitting_risk"
        explanation = (
            f"Overfitting risk: validation MAE {validation_mae:.4f} is {val_to_train_mae_ratio:.2f}x the "
            f"train MAE {train_mae:.4f} (threshold {OVERFITTING_VAL_TO_TRAIN_MAE_RATIO:.2f}x)."
        )
    else:
        status = "balanced"
        ratio_text = f"{val_to_train_mae_ratio:.2f}x" if val_to_train_mae_ratio is not None else "n/a"
        baseline_text = (
            f" and beats the {baseline_key!r} baseline by {-val_vs_baseline_mae_delta:.4f}"
            if val_vs_baseline_mae_delta is not None
            else ""
        )
        explanation = (
            f"Balanced: validation MAE {validation_mae:.4f} is {ratio_text} the train MAE {train_mae:.4f} "
            f"(below {OVERFITTING_VAL_TO_TRAIN_MAE_RATIO:.2f}x){baseline_text}."
        )

    early_stopped = bool(epochs_trained < max_epochs)
    explanation += (
        f" Training ran {epochs_trained}/{max_epochs} epochs (best epoch {best_epoch}"
        f"{', early stopped' if early_stopped else ''})."
    )
    return {
        "status": status,
        "model_key": model_key,
        "baseline_key": baseline_key,
        "train_mae": train_mae,
        "validation_mae": validation_mae,
        "train_rmse": train_rmse,
        "validation_rmse": validation_rmse,
        "baseline_train_mae": baseline_train_mae,
        "baseline_validation_mae": baseline_validation_mae,
        "val_minus_train_mae": val_minus_train_mae,
        "val_to_train_mae_ratio": val_to_train_mae_ratio,
        "val_vs_baseline_mae_delta": val_vs_baseline_mae_delta,
        "early_stopped": early_stopped,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "thresholds": {
            "underfitting_train_ratio_vs_baseline": UNDERFITTING_TRAIN_RATIO_VS_BASELINE,
            "overfitting_val_to_train_mae_ratio": OVERFITTING_VAL_TO_TRAIN_MAE_RATIO,
        },
        "explanation": explanation,
        "evaluated_splits": ["train", "validation"],
    }


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
    registered_model_name: Optional[str] = None,
    artifact_path: str = "model",
) -> None:
    """Log the MLP as an ``mlflow.pyfunc`` model in the active run.

    With a non-empty ``registered_model_name`` the model is also registered as
    a new version in the Model Registry; with ``None`` it is only logged under
    ``runs:/<run_id>/<artifact_path>`` (useful for experimental comparisons
    that must not add candidates to the registry).

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

    if registered_model_name is not None and (
        not isinstance(registered_model_name, str) or not registered_model_name.strip()
    ):
        raise ValueError("registered_model_name must be a non-empty string or None")
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


def _mae_rmse(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def _metric_value(split_metrics: Any, model_key: str, metric: str) -> Optional[float]:
    """``split_metrics[model_key][metric]`` as a finite float, or ``None``."""
    if not isinstance(split_metrics, Mapping):
        return None
    model_metrics = split_metrics.get(model_key)
    if not isinstance(model_metrics, Mapping):
        return None
    value = model_metrics.get(metric)
    if not _is_real(value) or not math.isfinite(float(value)):
        return None
    return float(value)


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
        (isinstance(config.lr_scheduler, str) and config.lr_scheduler in ("none", "reduce_on_plateau"),
         "lr_scheduler must be 'none' or 'reduce_on_plateau'"),
        (_is_real(config.lr_scheduler_factor) and math.isfinite(config.lr_scheduler_factor)
         and 0.0 < config.lr_scheduler_factor < 1.0,
         "lr_scheduler_factor must be a finite number in (0.0, 1.0)"),
        (_is_int(config.lr_scheduler_patience) and config.lr_scheduler_patience >= 1,
         "lr_scheduler_patience must be an integer >= 1"),
        (_is_real(config.lr_scheduler_min_lr) and math.isfinite(config.lr_scheduler_min_lr)
         and config.lr_scheduler_min_lr >= 0.0,
         "lr_scheduler_min_lr must be a finite number >= 0"),
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
        lr_scheduler_factor=float(config.lr_scheduler_factor),
        lr_scheduler_patience=int(config.lr_scheduler_patience),
        lr_scheduler_min_lr=float(config.lr_scheduler_min_lr),
    )
