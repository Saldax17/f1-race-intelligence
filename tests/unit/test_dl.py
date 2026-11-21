"""Unit tests for the M7 PyTorch tabular MLP (``modeling/dl.py``).

Tests that need PyTorch or MLflow skip when the optional extras are absent
(the ``base-runtime`` CI job); the tests that simulate a missing PyTorch run
everywhere.
"""

import builtins
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence.modeling.dl import (
    PERMUTATION_COLLINEARITY_NOTE,
    FeatureImportanceEntry,
    FittedTabularMLP,
    MLPConfig,
    MLPEpochMetrics,
    PermutationImportanceResult,
    TabularMLPPyfuncModel,
    compute_permutation_importance,
    diagnose_generalization,
    load_mlp_checkpoint,
    log_mlp_pyfunc_model,
    save_mlp_checkpoint,
    train_tabular_mlp,
)

FEATURES = ("num__tyre_age", "num__speed_mean", "num__lap_time_prev_1", "cat__compound_SOFT")
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FAST = MLPConfig(hidden_dims=(16, 8), max_epochs=5, batch_size=16, early_stopping_patience=3)


def _frame(rows: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, len(FEATURES)))
    values[:, 3] = (values[:, 3] > 0).astype(float)
    frame = pd.DataFrame(values, columns=list(FEATURES)).astype("float64")
    target = 90.0 + 1.5 * values[:, 0] - 0.8 * values[:, 1] + 0.5 * values[:, 2] + rng.normal(scale=0.05, size=rows)
    return frame, target


@pytest.fixture
def splits() -> dict:
    train_x, train_y = _frame(160, seed=1)
    val_x, val_y = _frame(48, seed=2)
    test_x, test_y = _frame(32, seed=3)
    return {"train": (train_x, train_y), "val": (val_x, val_y), "test": (test_x, test_y)}


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="the M7 MLP tests need the optional `dl` extra")


def _train(splits: dict, config: MLPConfig = FAST, **overrides) -> FittedTabularMLP:
    train_x, train_y = overrides.pop("train", splits["train"])
    val_x, val_y = overrides.pop("val", splits["val"])
    assert not overrides
    return train_tabular_mlp(train_x, train_y, val_x, val_y, feature_names=FEATURES, config=config)


@pytest.fixture
def mlflow_tracking(tmp_path: Path, torch):
    mlflow = pytest.importorskip("mlflow", reason="the pyfunc tests need the optional `training` extra")
    from mlflow.tracking import MlflowClient

    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow' / 'tracking.db').as_posix()}"
    (tmp_path / "mlflow").mkdir()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    client.create_experiment("m7-dl-unit", artifact_location=(tmp_path / "mlflow" / "artifacts").as_uri())
    mlflow.set_experiment("m7-dl-unit")
    try:
        yield mlflow, client
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


# --------------------------------------------------------------------------- #
# Reproducibility, losses and config validation
# --------------------------------------------------------------------------- #


def test_same_seed_reproduces_losses_and_predictions_exactly(torch, splits) -> None:
    config = MLPConfig(hidden_dims=(16, 8), dropout=0.2, max_epochs=6, batch_size=16, random_seed=7)
    first = _train(splits, config)
    second = _train(splits, config)
    test_x = splits["test"][0]

    assert first.history == second.history
    assert (first.best_epoch, first.epochs_trained, first.best_val_loss) == (
        second.best_epoch, second.epochs_trained, second.best_val_loss
    )
    assert np.array_equal(first.predict(test_x), second.predict(test_x))
    for key, tensor in first.state_dict.items():
        assert torch.equal(tensor, second.state_dict[key])

    other_seed = _train(splits, MLPConfig(hidden_dims=(16, 8), dropout=0.2, max_epochs=6, batch_size=16, random_seed=8))
    assert not np.array_equal(first.predict(test_x), other_seed.predict(test_x))


def test_training_restores_global_determinism_flag(torch, splits) -> None:
    before = torch.are_deterministic_algorithms_enabled()
    _train(splits)
    assert torch.are_deterministic_algorithms_enabled() == before


@pytest.mark.parametrize("loss", ["huber", "mse"])
def test_supported_losses_train_and_predict_finite_seconds(torch, splits, loss: str) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=(16,), max_epochs=4, batch_size=32, loss=loss))
    predictions = fitted.predict(splits["test"][0])

    assert fitted.config.loss == loss
    assert predictions.dtype == np.float64 and predictions.shape == (32,)
    assert np.isfinite(predictions).all()
    assert all(isinstance(rec, MLPEpochMetrics) and math.isfinite(rec.train_loss) for rec in fitted.history)
    assert [rec.epoch for rec in fitted.history] == list(range(1, fitted.epochs_trained + 1))


def test_huber_and_mse_are_different_objectives(torch, splits) -> None:
    huber = _train(splits, MLPConfig(hidden_dims=(16,), max_epochs=3, loss="huber", scale_target=False))
    mse = _train(splits, MLPConfig(hidden_dims=(16,), max_epochs=3, loss="mse", scale_target=False))
    assert huber.history[0].train_loss != mse.history[0].train_loss


def test_architecture_and_parameter_count(torch, splits) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=[16, 8], dropout=0.0, max_epochs=1))
    # 4->16, 16->8, 8->1 linear layers; a list of dims is normalised to a tuple.
    assert fitted.config.hidden_dims == (16, 8)
    assert fitted.total_params == (4 * 16 + 16) + (16 * 8 + 8) + (8 * 1 + 1)
    assert fitted.feature_names == FEATURES
    assert all(tensor.device.type == "cpu" for tensor in fitted.state_dict.values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_dims": ()},
        {"hidden_dims": (0,)},
        {"hidden_dims": (16, -4)},
        {"hidden_dims": (8.5,)},
        {"hidden_dims": (True,)},
        {"hidden_dims": "64"},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1e-4},
        {"batch_size": 0},
        {"max_epochs": 0},
        {"early_stopping_patience": 0},
        {"min_delta": -0.1},
        {"loss": "mae"},
        {"scale_target": "yes"},
        {"random_seed": 1.5},
        {"lr_scheduler": "cosine"},
        {"lr_scheduler": None},
        {"lr_scheduler_factor": 0.0},
        {"lr_scheduler_factor": 1.0},
        {"lr_scheduler_factor": -0.5},
        {"lr_scheduler_factor": float("nan")},
        {"lr_scheduler_factor": float("inf")},
        {"lr_scheduler_factor": True},
        {"lr_scheduler_factor": "0.5"},
        {"lr_scheduler_patience": 0},
        {"lr_scheduler_patience": -1},
        {"lr_scheduler_patience": 2.0},
        {"lr_scheduler_patience": True},
        {"lr_scheduler_min_lr": -1e-6},
        {"lr_scheduler_min_lr": float("nan")},
        {"lr_scheduler_min_lr": float("inf")},
        {"lr_scheduler_min_lr": False},
    ],
)
def test_invalid_hyperparameters_are_rejected(torch, splits, overrides: dict) -> None:
    with pytest.raises(ValueError):
        _train(splits, MLPConfig(**overrides))


def test_invalid_inputs_are_rejected(torch, splits) -> None:
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]

    with pytest.raises(ValueError, match="feature_names"):
        train_tabular_mlp(train_x, train_y, val_x, val_y, feature_names=[], config=FAST)
    with pytest.raises(ValueError, match="missing required feature columns"):
        train_tabular_mlp(train_x.drop(columns=[FEATURES[0]]), train_y, val_x, val_y, feature_names=FEATURES, config=FAST)
    with pytest.raises(ValueError, match="rows"):
        train_tabular_mlp(train_x, train_y[:-1], val_x, val_y, feature_names=FEATURES, config=FAST)
    with pytest.raises(ValueError, match="no finite targets"):
        train_tabular_mlp(train_x, np.full(len(train_y), np.nan), val_x, val_y, feature_names=FEATURES, config=FAST)
    broken = train_x.copy()
    broken.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        train_tabular_mlp(broken, train_y, val_x, val_y, feature_names=FEATURES, config=FAST)


# --------------------------------------------------------------------------- #
# Target scaling: training targets only
# --------------------------------------------------------------------------- #


def test_target_scaling_uses_finite_train_targets_only(torch, splits) -> None:
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    train_y = train_y.copy()
    train_y[:5] = np.nan  # rows without a target are dropped, never imputed
    config = MLPConfig(hidden_dims=(8,), max_epochs=2)

    normal = _train(splits, config, train=(train_x, train_y), val=(val_x, val_y))
    extreme = _train(splits, config, train=(train_x, train_y), val=(val_x, val_y * 1e6 + 1e9))

    finite_train = train_y[np.isfinite(train_y)]
    assert normal.y_mean == pytest.approx(float(np.mean(finite_train)), rel=0, abs=1e-12)
    assert normal.y_std == pytest.approx(float(np.std(finite_train)), rel=0, abs=1e-12)
    assert (extreme.y_mean, extreme.y_std) == (normal.y_mean, normal.y_std)
    # Validation only drives monitoring: the extreme targets change the loss, not the scaling.
    assert extreme.best_val_loss != normal.best_val_loss


def test_constant_target_does_not_divide_by_zero(torch, splits) -> None:
    train_x, train_y = splits["train"]
    fitted = _train(splits, MLPConfig(hidden_dims=(8,), max_epochs=3), train=(train_x, np.full(len(train_y), 91.25)))

    assert fitted.y_mean == pytest.approx(91.25)
    assert fitted.y_std == 1.0
    predictions = fitted.predict(splits["test"][0])
    assert np.isfinite(predictions).all()
    assert all(math.isfinite(rec.train_loss) for rec in fitted.history)


def test_scale_target_false_keeps_identity_scaling(torch, splits) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=(8,), max_epochs=2, scale_target=False))
    assert (fitted.y_mean, fitted.y_std) == (0.0, 1.0)


def test_predict_inverts_target_scaling_to_seconds(torch, splits) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=(32, 16), max_epochs=60, early_stopping_patience=60, batch_size=16))
    test_x, test_y = splits["test"]
    predictions = fitted.predict(test_x)
    # Back in seconds around 90 s, not in standardised units around 0.
    assert abs(float(np.mean(predictions)) - float(np.mean(test_y))) < 1.0
    assert float(np.sqrt(np.mean((predictions - test_y) ** 2))) < float(np.std(test_y))


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #


def test_early_stopping_stops_and_restores_best_checkpoint(torch, splits) -> None:
    train_x, train_y = splits["train"]
    val_x, _ = splits["val"]
    noise_val_y = np.random.default_rng(99).normal(loc=95.0, scale=5.0, size=len(val_x))  # unlearnable
    config = MLPConfig(
        hidden_dims=(32, 16), dropout=0.0, max_epochs=300, batch_size=16, early_stopping_patience=4, min_delta=0.0
    )
    fitted = _train(splits, config, val=(val_x, noise_val_y))

    assert fitted.epochs_trained < config.max_epochs
    assert 1 <= fitted.best_epoch <= fitted.epochs_trained
    assert fitted.epochs_trained - fitted.best_epoch == config.early_stopping_patience
    val_losses = [rec.val_loss for rec in fitted.history]
    assert fitted.best_val_loss == min(val_losses) == val_losses[fitted.best_epoch - 1]

    # The restored weights reproduce the best validation loss, not the last epoch's.
    scaled_pred = (fitted.predict(val_x) - fitted.y_mean) / fitted.y_std
    scaled_true = (noise_val_y - fitted.y_mean) / fitted.y_std
    restored_loss = float(
        torch.nn.SmoothL1Loss(beta=1.0)(
            torch.tensor(scaled_pred, dtype=torch.float32), torch.tensor(scaled_true, dtype=torch.float32)
        )
    )
    assert restored_loss == pytest.approx(fitted.best_val_loss, rel=1e-4)
    assert restored_loss < val_losses[-1]


def test_validation_without_finite_targets_falls_back_to_train_loss(torch, splits) -> None:
    val_x, val_y = splits["val"]
    fitted = _train(splits, MLPConfig(hidden_dims=(8,), max_epochs=4), val=(val_x, np.full(len(val_y), np.nan)))

    assert fitted.best_val_loss is None
    assert all(rec.val_loss is None for rec in fitted.history)
    train_losses = [rec.train_loss for rec in fitted.history]
    assert 1 <= fitted.best_epoch <= fitted.epochs_trained
    assert train_losses[fitted.best_epoch - 1] == min(train_losses)
    assert np.isfinite(fitted.predict(splits["test"][0])).all()


def test_validation_rows_without_target_are_ignored(torch, splits) -> None:
    val_x, val_y = splits["val"]
    val_x, val_y = val_x.copy(), val_y.copy()
    val_y[:10] = np.nan
    val_x.iloc[:10, 0] = np.nan  # features of untargeted rows are never read
    fitted = _train(splits, MLPConfig(hidden_dims=(8,), max_epochs=2), val=(val_x, val_y))
    assert fitted.best_val_loss is not None and math.isfinite(fitted.best_val_loss)


# --------------------------------------------------------------------------- #
# Missing PyTorch / MLflow
# --------------------------------------------------------------------------- #


def _block_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_torch_raises_clear_install_hint(monkeypatch, splits, tmp_path: Path) -> None:
    _block_torch(monkeypatch)
    hint = re.escape('python -m pip install -e ".[training,dl]"')
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]

    with pytest.raises(RuntimeError, match=hint):
        train_tabular_mlp(train_x, train_y, val_x, val_y, feature_names=FEATURES, config=MLPConfig())
    with pytest.raises(RuntimeError, match=hint):
        load_mlp_checkpoint(tmp_path / "missing.pt")

    fitted = FittedTabularMLP(
        feature_names=FEATURES, config=MLPConfig(), state_dict={}, y_mean=0.0, y_std=1.0,
        best_epoch=1, epochs_trained=1, best_val_loss=None, total_params=0, history=[],
    )
    with pytest.raises(RuntimeError, match=hint):
        fitted.predict(train_x)
    with pytest.raises(RuntimeError, match=hint):
        save_mlp_checkpoint(fitted, tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match=hint):
        TabularMLPPyfuncModel().load_context(SimpleNamespace(artifacts={"checkpoint": str(tmp_path / "x.pt")}))
    with pytest.raises(RuntimeError, match=hint):
        log_mlp_pyfunc_model(fitted, example=train_x.head(2), registered_model_name="unused")
    with pytest.raises(RuntimeError, match=hint):
        compute_permutation_importance(fitted, train_x, splits["train"][1])


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_ROOT), env.get("PYTHONPATH")]))
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def test_module_imports_without_torch_or_mlflow() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['mlflow'] = None\n"
        "import numpy as np, pandas as pd\n"
        "import f1_race_intelligence.modeling.dl as dl\n"
        "assert sorted(dl.__all__) == sorted(['MLPConfig', 'MLPEpochMetrics', 'FittedTabularMLP',\n"
        "    'TabularMLPPyfuncModel', 'train_tabular_mlp', 'save_mlp_checkpoint',\n"
        "    'load_mlp_checkpoint', 'log_mlp_pyfunc_model', 'PERMUTATION_COLLINEARITY_NOTE',\n"
        "    'FeatureImportanceEntry', 'PermutationImportanceResult',\n"
        "    'compute_permutation_importance', 'diagnose_generalization'])\n"
        "assert all(hasattr(dl, name) for name in dl.__all__)\n"
        "dl.TabularMLPPyfuncModel()\n"
        "frame = pd.DataFrame({'a': [1.0, 2.0]})\n"
        "try:\n"
        "    dl.train_tabular_mlp(frame, np.ones(2), frame, np.ones(2), feature_names=['a'], config=dl.MLPConfig())\n"
        "except RuntimeError as exc:\n"
        "    assert '.[training,dl]' in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('expected RuntimeError')\n"
        "print('lazy-ok')\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout


def test_module_import_does_not_load_torch() -> None:
    result = _run_isolated(
        "import sys\nimport f1_race_intelligence.modeling.dl\nassert 'torch' not in sys.modules\nprint('no-torch')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "no-torch" in result.stdout


# --------------------------------------------------------------------------- #
# Checkpoint and pyfunc contract
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trip_with_weights_only(torch, splits, tmp_path: Path) -> None:
    fitted = _train(splits)
    path = save_mlp_checkpoint(fitted, tmp_path / "nested" / "mlp_checkpoint.pt")
    assert path == tmp_path / "nested" / "mlp_checkpoint.pt" and path.is_file()

    raw = torch.load(path, map_location="cpu", weights_only=True)  # plain tensors and primitives only
    loaded = load_mlp_checkpoint(path)
    assert set(raw) == set(loaded)
    assert loaded["feature_names"] == list(FEATURES)
    assert loaded["hidden_dims"] == [16, 8]
    assert loaded["dropout"] == pytest.approx(0.1)
    assert (loaded["y_mean"], loaded["y_std"], loaded["scale_target"]) == (fitted.y_mean, fitted.y_std, True)
    assert set(loaded["state_dict"]) == set(fitted.state_dict)
    for key, tensor in loaded["state_dict"].items():
        assert tensor.device.type == "cpu"
        assert torch.equal(tensor, fitted.state_dict[key])

    model = TabularMLPPyfuncModel()
    model.load_context(SimpleNamespace(artifacts={"checkpoint": str(path)}))
    test_x = splits["test"][0]
    np.testing.assert_array_equal(model.predict(None, test_x), fitted.predict(test_x))


def test_checkpoint_rejects_foreign_payload(torch, tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_mlp_checkpoint(path)


def test_pyfunc_wrapper_selects_columns_in_order_and_returns_float64(torch, splits, tmp_path: Path) -> None:
    fitted = _train(splits)
    model = TabularMLPPyfuncModel()
    model.load_context(SimpleNamespace(artifacts={"checkpoint": str(save_mlp_checkpoint(fitted, tmp_path / "c.pt"))}))
    test_x = splits["test"][0]
    expected = fitted.predict(test_x)

    shuffled = test_x.loc[:, list(reversed(FEATURES))].assign(key_session=1.0, extra=-5.0)
    predictions = model.predict(None, shuffled, params=None)
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64 and predictions.shape == (len(test_x),)
    np.testing.assert_array_equal(predictions, expected)

    with pytest.raises(ValueError, match="missing required feature columns"):
        model.predict(None, test_x.drop(columns=[FEATURES[2]]))
    with pytest.raises(RuntimeError, match="load_context"):
        TabularMLPPyfuncModel().predict(None, test_x)


def test_log_mlp_pyfunc_model_round_trip_honours_m8_contract(mlflow_tracking, splits, tmp_path: Path) -> None:
    mlflow, client = mlflow_tracking
    fitted = _train(splits)
    train_x = splits["train"][0]
    test_x = splits["test"][0].astype("float64")

    with mlflow.start_run() as run:
        log_mlp_pyfunc_model(fitted, example=train_x.head(5), registered_model_name="f1-next-lap-mlp-unit")
        run_id = run.info.run_id

    versions = [v for v in client.search_model_versions("name='f1-next-lap-mlp-unit'") if v.run_id == run_id]
    assert len(versions) == 1
    loaded = mlflow.pyfunc.load_model(f"models:/f1-next-lap-mlp-unit/{versions[0].version}")

    assert loaded.metadata.run_id == run_id
    assert loaded.metadata.get_input_schema().input_names() == list(FEATURES)
    predictions = loaded.predict(test_x)
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64 and predictions.shape == (len(test_x),)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, fitted.predict(test_x), rtol=1e-6, atol=1e-6)

    model_dir = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model", dst_path=str(tmp_path / "download")
    ))
    assert (model_dir / "MLmodel").is_file()
    assert (model_dir / "artifacts" / "mlp_checkpoint.pt").is_file()
    entries = list(model_dir.rglob("*"))
    assert not [path for path in entries if path.is_symlink()]
    for path in entries:
        if path.is_file():
            for part in path.relative_to(model_dir).parts:
                assert PACKAGE_FILE_PATTERN.fullmatch(part), part

    with pytest.raises(Exception):
        loaded.predict(test_x.drop(columns=[FEATURES[1]]))
    with pytest.raises(ValueError, match="missing required feature columns"):
        loaded.unwrap_python_model().predict(None, test_x.drop(columns=[FEATURES[1]]))


def test_log_mlp_pyfunc_model_requires_ordered_example(mlflow_tracking, splits) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=(4,), max_epochs=1))
    reordered = splits["train"][0].loc[:, list(reversed(FEATURES))].head(3)
    with pytest.raises(ValueError, match="same order"):
        log_mlp_pyfunc_model(fitted, example=reordered, registered_model_name="unused")


# --------------------------------------------------------------------------- #
# Learning-rate scheduler
# --------------------------------------------------------------------------- #


def _plateau_config(**overrides) -> MLPConfig:
    base = dict(
        hidden_dims=(32, 16), dropout=0.0, learning_rate=1e-2, max_epochs=60, batch_size=16,
        early_stopping_patience=60, min_delta=0.0, lr_scheduler_factor=0.5, lr_scheduler_patience=2,
        lr_scheduler_min_lr=1e-3,
    )
    base.update(overrides)
    return MLPConfig(**base)


def test_epoch_metrics_learning_rate_is_optional_for_backward_compatibility() -> None:
    record = MLPEpochMetrics(epoch=1, train_loss=0.5, val_loss=None)
    assert record.learning_rate is None
    assert MLPConfig().lr_scheduler == "reduce_on_plateau"
    assert (MLPConfig().lr_scheduler_factor, MLPConfig().lr_scheduler_patience, MLPConfig().lr_scheduler_min_lr) == (
        0.5, 5, 1e-6
    )


def test_reduce_on_plateau_lowers_lr_after_patience_and_respects_min_lr(torch, splits) -> None:
    val_x, _ = splits["val"]
    noise_val_y = np.random.default_rng(99).normal(loc=95.0, scale=5.0, size=len(val_x))  # unlearnable
    config = _plateau_config(lr_scheduler="reduce_on_plateau")
    fitted = _train(splits, config, val=(val_x, noise_val_y))

    rates = [rec.learning_rate for rec in fitted.history]
    assert all(isinstance(rate, float) for rate in rates)
    assert rates[0] == pytest.approx(config.learning_rate)
    assert rates == sorted(rates, reverse=True)  # never increases
    assert min(rates) == pytest.approx(config.lr_scheduler_min_lr)
    assert all(rate >= config.lr_scheduler_min_lr * (1 - 1e-9) for rate in rates)

    # Replay the ReduceLROnPlateau rule (abs threshold = min_delta) on the recorded validation losses.
    lr, best, bad = config.learning_rate, math.inf, 0
    for record in fitted.history:
        assert record.learning_rate == pytest.approx(lr, rel=1e-9)
        if record.val_loss < best - config.min_delta:
            best, bad = record.val_loss, 0
        else:
            bad += 1
        if bad > config.lr_scheduler_patience:
            lr, bad = max(lr * config.lr_scheduler_factor, config.lr_scheduler_min_lr), 0


def test_lr_scheduler_none_keeps_learning_rate_constant(torch, splits) -> None:
    val_x, _ = splits["val"]
    noise_val_y = np.random.default_rng(99).normal(loc=95.0, scale=5.0, size=len(val_x))
    config = _plateau_config(lr_scheduler="none", max_epochs=20)
    fitted = _train(splits, config, val=(val_x, noise_val_y))

    assert fitted.epochs_trained == 20
    assert [rec.learning_rate for rec in fitted.history] == [config.learning_rate] * 20


def test_lr_scheduler_training_is_reproducible_and_restores_determinism_flag(torch, splits) -> None:
    before = torch.are_deterministic_algorithms_enabled()
    config = _plateau_config(max_epochs=15)
    first = _train(splits, config)
    second = _train(splits, config)
    assert first.history == second.history
    assert torch.are_deterministic_algorithms_enabled() == before


# --------------------------------------------------------------------------- #
# Permutation feature importance
# --------------------------------------------------------------------------- #

PFI_CONFIG = MLPConfig(hidden_dims=(32, 16), max_epochs=60, early_stopping_patience=60, batch_size=16)


def test_permutation_importance_ranks_informative_feature_above_noise(torch, splits) -> None:
    fitted = _train(splits, PFI_CONFIG)
    val_x, val_y = splits["val"]
    result = compute_permutation_importance(fitted, val_x, val_y, n_repeats=5)

    assert isinstance(result, PermutationImportanceResult)
    assert all(isinstance(entry, FeatureImportanceEntry) for entry in result.importances)
    assert (result.split, result.n_rows, result.n_repeats, result.random_seed) == ("validation", 48, 5, 42)
    assert sorted(entry.feature for entry in result.importances) == sorted(FEATURES)
    by_name = {entry.feature: entry for entry in result.importances}
    # The target is 1.5 * tyre_age - 0.8 * speed_mean + 0.5 * lap_time_prev_1; compound is pure noise.
    assert result.importances[0].feature == "num__tyre_age"
    assert by_name["num__tyre_age"].mae_increase_mean > 0.5
    assert by_name["num__tyre_age"].mae_increase_mean > 10 * abs(by_name["cat__compound_SOFT"].mae_increase_mean)
    assert by_name["num__tyre_age"].rmse_increase_mean > 10 * abs(by_name["cat__compound_SOFT"].rmse_increase_mean)
    keys = [(-e.mae_increase_mean, -e.rmse_increase_mean, e.feature) for e in result.importances]
    assert keys == sorted(keys)
    assert all(entry.mae_increase_std >= 0.0 and entry.rmse_increase_std >= 0.0 for entry in result.importances)

    predictions = fitted.predict(val_x)
    assert result.baseline_mae == pytest.approx(float(np.mean(np.abs(predictions - val_y))), rel=1e-12)
    assert result.baseline_rmse == pytest.approx(float(np.sqrt(np.mean((predictions - val_y) ** 2))), rel=1e-12)


def test_permutation_importance_is_deterministic_and_seed_dependent(torch, splits) -> None:
    fitted = _train(splits)
    val_x, val_y = splits["val"]
    first = compute_permutation_importance(fitted, val_x, val_y, n_repeats=3, random_seed=11)
    second = compute_permutation_importance(fitted, val_x, val_y, n_repeats=3, random_seed=11)
    other = compute_permutation_importance(fitted, val_x, val_y, n_repeats=3, random_seed=12)

    assert first == second
    assert first.random_seed == 11
    assert first.importances != other.importances
    assert compute_permutation_importance(fitted, val_x, val_y, n_repeats=3).random_seed == fitted.config.random_seed


def test_permutation_importance_does_not_mutate_inputs_or_global_rng(torch, splits) -> None:
    fitted = _train(splits)
    val_x, val_y = splits["val"]
    frame, targets = val_x.copy(), val_y.copy()
    array = val_x.to_numpy().copy()

    np.random.seed(123)
    np_state = np.random.get_state()
    import random as std_random

    std_random.seed(321)
    py_state = std_random.getstate()
    torch_state = torch.get_rng_state()

    from_frame = compute_permutation_importance(fitted, frame, targets, n_repeats=2)
    from_array = compute_permutation_importance(fitted, array, targets, n_repeats=2)

    pd.testing.assert_frame_equal(frame, val_x)
    np.testing.assert_array_equal(targets, val_y)
    np.testing.assert_array_equal(array, val_x.to_numpy())
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert std_random.getstate() == py_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert from_frame == from_array


def test_permutation_importance_drops_non_finite_targets(torch, splits) -> None:
    fitted = _train(splits)
    val_x, val_y = splits["val"]
    holes_x, holes_y = val_x.copy(), val_y.copy()
    holes_y[:8] = np.nan
    holes_y[8] = np.inf
    holes_x.iloc[:9, 0] = np.nan  # features of untargeted rows are never read

    with_holes = compute_permutation_importance(fitted, holes_x, holes_y, n_repeats=2)
    clean = compute_permutation_importance(fitted, val_x.iloc[9:], val_y[9:], n_repeats=2)
    assert with_holes.n_rows == len(val_y) - 9
    assert with_holes == clean

    with pytest.raises(ValueError, match="targets contain no finite values"):
        compute_permutation_importance(fitted, val_x, np.full(len(val_y), np.nan))
    broken = val_x.copy()
    broken.iloc[20, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_permutation_importance(fitted, broken, val_y)


@pytest.mark.parametrize("kwargs", [{"n_repeats": 0}, {"n_repeats": True}, {"n_repeats": 1.5}, {"split": ""},
                                    {"split": "  "}, {"split": None}, {"random_seed": 1.5}])
def test_permutation_importance_rejects_invalid_arguments(torch, splits, kwargs: dict) -> None:
    fitted = _train(splits, MLPConfig(hidden_dims=(4,), max_epochs=1))
    val_x, val_y = splits["val"]
    with pytest.raises(ValueError):
        compute_permutation_importance(fitted, val_x, val_y, **kwargs)


def test_permutation_importance_to_dict_top_k_and_collinearity_note(torch, splits) -> None:
    import json

    fitted = _train(splits)
    val_x, val_y = splits["val"]
    result = compute_permutation_importance(fitted, val_x, val_y, n_repeats=2)

    assert result.collinearity_note == PERMUTATION_COLLINEARITY_NOTE
    assert "one-hot" in PERMUTATION_COLLINEARITY_NOTE and "lap_duration" in PERMUTATION_COLLINEARITY_NOTE
    assert result.top(2) == result.importances[:2]
    assert result.top() == result.importances  # fewer than 10 features

    full = result.to_dict()
    top2 = result.to_dict(top_k=2)
    assert json.loads(json.dumps(top2)) == top2
    assert full["n_features"] == top2["n_features"] == len(FEATURES)
    assert len(full["importances"]) == len(FEATURES)
    assert top2["importances"] == [entry.to_dict() for entry in result.importances[:2]]
    assert top2["collinearity_note"] == PERMUTATION_COLLINEARITY_NOTE
    assert (top2["split"], top2["n_rows"], top2["n_repeats"]) == ("validation", 48, 2)
    assert set(top2["importances"][0]) == {
        "feature", "mae_increase_mean", "mae_increase_std", "rmse_increase_mean", "rmse_increase_std"
    }
    with pytest.raises(ValueError):
        result.top(-1)


# --------------------------------------------------------------------------- #
# Generalization diagnosis (train vs validation only)
# --------------------------------------------------------------------------- #


def _metrics(train_mae, val_mae, base_train=3.0, base_val=3.0, test=(1.0, 3.0)) -> dict:
    return {
        "train": {"mlp": {"mae": train_mae, "rmse": train_mae * 1.3}, "reference_mean": {"mae": base_train}},
        "validation": {"mlp": {"mae": val_mae, "rmse": val_mae * 1.3}, "reference_mean": {"mae": base_val}},
        "test": {"mlp": {"mae": test[0], "rmse": test[0]}, "reference_mean": {"mae": test[1]}},
    }


def _diagnose(metrics, **overrides) -> dict:
    kwargs = dict(epochs_trained=40, best_epoch=30, max_epochs=100)
    kwargs.update(overrides)
    return diagnose_generalization(metrics, **kwargs)


def test_diagnose_balanced() -> None:
    result = _diagnose(_metrics(1.0, 1.1))
    assert result["status"] == "balanced"
    assert result["train_mae"] == 1.0 and result["validation_mae"] == 1.1
    assert result["train_rmse"] == pytest.approx(1.3) and result["validation_rmse"] == pytest.approx(1.43)
    assert (result["baseline_train_mae"], result["baseline_validation_mae"]) == (3.0, 3.0)
    assert result["val_minus_train_mae"] == pytest.approx(0.1)
    assert result["val_to_train_mae_ratio"] == pytest.approx(1.1)
    assert result["val_vs_baseline_mae_delta"] == pytest.approx(-1.9)
    assert result["early_stopped"] is True
    assert (result["epochs_trained"], result["best_epoch"], result["max_epochs"]) == (40, 30, 100)
    assert result["thresholds"] == {
        "underfitting_train_ratio_vs_baseline": 0.95, "overfitting_val_to_train_mae_ratio": 1.25
    }
    assert result["evaluated_splits"] == ["train", "validation"]
    assert "Balanced" in result["explanation"] and "1.10x" in result["explanation"]


def test_diagnose_overfitting_risk() -> None:
    result = _diagnose(_metrics(1.0, 1.25), epochs_trained=100, best_epoch=100)
    assert result["status"] == "overfitting_risk"
    assert result["val_to_train_mae_ratio"] == pytest.approx(1.25)
    assert result["early_stopped"] is False
    assert "Overfitting" in result["explanation"]


@pytest.mark.parametrize(
    ("train_mae", "val_mae", "base_train", "base_val"),
    [
        (2.0, 3.1, 3.0, 3.0),  # validation does not beat the mean baseline
        (2.9, 2.95, 3.0, 3.0),  # train MAE is >= 95 % of the baseline's train MAE
        (1.0, 2.0, 3.0, 2.0),  # underfitting wins over a high val/train ratio
    ],
)
def test_diagnose_underfitting(train_mae, val_mae, base_train, base_val) -> None:
    result = _diagnose(_metrics(train_mae, val_mae, base_train, base_val))
    assert result["status"] == "underfitting"
    assert "Underfitting" in result["explanation"]


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"train": {"mlp": {"mae": 1.0}}},
        {"train": {"mlp": {"mae": 1.0}}, "validation": {"mlp": {"mae": float("nan")}}},
        {"train": {"mlp": {"mae": None}}, "validation": {"mlp": {"mae": 1.0}}},
        {"train": {"other": {"mae": 1.0}}, "validation": {"other": {"mae": 1.0}}},
    ],
)
def test_diagnose_insufficient_metrics(metrics) -> None:
    result = _diagnose(metrics)
    assert result["status"] == "insufficient_metrics"
    assert result["val_to_train_mae_ratio"] is None
    assert result["evaluated_splits"] == ["train", "validation"]


def test_diagnose_zero_train_mae_has_no_ratio_and_custom_keys() -> None:
    metrics = {
        "train": {"net": {"mae": 0.0}, "dummy": {"mae": 3.0}},
        "validation": {"net": {"mae": 0.5}, "dummy": {"mae": 3.0}},
    }
    result = _diagnose(metrics, model_key="net", baseline_key="dummy")
    assert result["val_to_train_mae_ratio"] is None
    assert result["status"] == "balanced"
    assert (result["model_key"], result["baseline_key"]) == ("net", "dummy")


class _NoTestMetrics(dict):
    """Metrics mapping that fails the test if the ``test`` split is ever read."""

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("diagnose_generalization must never read the test split")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "test":
            raise AssertionError("diagnose_generalization must never read the test split")
        return super().get(key, default)


@pytest.mark.parametrize(("train_mae", "val_mae"), [(1.0, 1.1), (1.0, 1.6), (2.0, 3.1)])
def test_diagnose_ignores_test_split_entirely(train_mae, val_mae) -> None:
    normal = _diagnose(_metrics(train_mae, val_mae))
    absurd = _diagnose(_metrics(train_mae, val_mae, test=(1e9, 1e9)))
    no_test = _metrics(train_mae, val_mae)
    del no_test["test"]

    assert absurd == normal
    assert _diagnose(no_test) == normal
    assert _diagnose(_NoTestMetrics(_metrics(train_mae, val_mae, test=(1e9, -1e9)))) == normal
    assert "test" not in normal["evaluated_splits"]


def test_diagnose_generalization_runs_without_torch_or_mlflow() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['mlflow'] = None\n"
        "import f1_race_intelligence.modeling.dl as dl\n"
        "metrics = {'train': {'mlp': {'mae': 1.0}, 'reference_mean': {'mae': 3.0}},\n"
        "           'validation': {'mlp': {'mae': 1.1}, 'reference_mean': {'mae': 3.0}}}\n"
        "result = dl.diagnose_generalization(metrics, epochs_trained=5, best_epoch=3, max_epochs=10)\n"
        "assert result['status'] == 'balanced', result\n"
        "assert dl.diagnose_generalization({}, epochs_trained=1, best_epoch=1, max_epochs=10)['status'] "
        "== 'insufficient_metrics'\n"
        "assert sys.modules['torch'] is None and sys.modules['mlflow'] is None\n"
        "assert 'Permutation' in dl.PERMUTATION_COLLINEARITY_NOTE\n"
        "print('diagnose-ok')\n"
    )
    result = _run_isolated(code)
    assert result.returncode == 0, result.stderr
    assert "diagnose-ok" in result.stdout


# --------------------------------------------------------------------------- #
# Logging without registering
# --------------------------------------------------------------------------- #


def test_log_mlp_pyfunc_model_without_registry_logs_run_artifact_only(mlflow_tracking, splits) -> None:
    mlflow, client = mlflow_tracking
    fitted = _train(splits)
    test_x = splits["test"][0].astype("float64")

    with mlflow.start_run() as run:
        log_mlp_pyfunc_model(fitted, example=splits["train"][0].head(5), registered_model_name=None)
        run_id = run.info.run_id
    with mlflow.start_run():
        log_mlp_pyfunc_model(fitted, example=splits["train"][0].head(5))  # the default is None too

    assert client.search_registered_models() == []
    assert list(client.search_model_versions()) == []
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert loaded.metadata.run_id == run_id
    assert loaded.metadata.get_input_schema().input_names() == list(FEATURES)
    np.testing.assert_allclose(loaded.predict(test_x), fitted.predict(test_x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["", "   ", 7])
def test_log_mlp_pyfunc_model_rejects_blank_registered_name(mlflow_tracking, splits, name) -> None:
    mlflow, _ = mlflow_tracking
    fitted = _train(splits, MLPConfig(hidden_dims=(4,), max_epochs=1))
    with mlflow.start_run(), pytest.raises(ValueError, match="registered_model_name"):
        log_mlp_pyfunc_model(fitted, example=splits["train"][0].head(3), registered_model_name=name)
