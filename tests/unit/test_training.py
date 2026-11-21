"""M7 tests using synthetic M5-shaped rows and locally generated M6 artifacts."""

import hashlib
import json
from pathlib import Path
from re import escape as re_escape
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling.runner import ModelingRunner
from f1_race_intelligence.modeling.training import TrainingRunner, regression_metrics

RACE_START = pd.Timestamp("2023-03-05T15:00:00Z")


def _write_synthetic_m5(root: Path) -> None:
    for session, year, day_offset in ((11, 2023, 0), (12, 2024, 365), (13, 2025, 730)):
        rows: List[Dict[str, Any]] = []
        start = RACE_START + pd.Timedelta(days=day_offset)
        for driver in (1, 44):
            for lap in range(1, 5):
                rows.append(
                    {
                        "year": year,
                        "meeting_key": 1000 + session,
                        "session_key": session,
                        "driver_number": driver,
                        "lap_number": lap,
                        "session_date": start,
                        "lap_duration": 90.0 + lap + driver * 0.1,
                        "tyre_age": lap,
                        "position_at_lap_start": 1,
                        "speed_mean": 200.0 + lap + session,
                        "gap_to_leader_s": None if lap == 1 else float(lap),
                        "lap_time_prev_1": None if lap == 1 else 90.0 + lap - 1,
                        "compound": "SOFT",
                        "team_name": "Red Bull",
                        "driver_acronym": f"D{driver}",
                        "is_pit_out_lap": False,
                        "pit_in_lap": False,
                        "is_lapped": False,
                        "next_lap_time": 91.0 + lap + driver * 0.1 if lap < 4 else None,
                        "has_target": lap < 4,
                    }
                )
        directory = root / str(year)
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(directory / f"session_{session}.parquet", index=False)


@pytest.fixture
def synthetic_m6(tmp_path: Path) -> tuple:
    feature_root = tmp_path / "lap_features"
    _write_synthetic_m5(feature_root)
    m6_settings = AppSettings(
        modeling={
            "input_path": str(feature_root),
            "output_path": str(tmp_path / "modeling"),
            "report_path": str(tmp_path / "modeling" / "reports"),
            "train_fraction": 1 / 3,
            "validation_fraction": 1 / 3,
            "min_sessions_per_split": 1,
        }
    )
    m6 = ModelingRunner(m6_settings).run()
    assert m6.aborted is None
    report_paths = list((tmp_path / "modeling" / "reports").glob("modeling_report_*.json"))
    assert len(report_paths) == 1
    report_path = report_paths[0]
    m6_payload = json.loads(report_path.read_text(encoding="utf-8"))
    training_settings = AppSettings(
        modeling=m6_settings.modeling,
        training={
            "m6_report_path": str(report_path),
            "tracking_db_path": str(tmp_path / "mlflow" / "tracking.db"),
            "artifact_path": str(tmp_path / "mlflow" / "artifacts"),
            "output_path": str(tmp_path / "training-reports"),
            "experiment_name": "synthetic-m7-test",
            "registered_model_name": "synthetic-next-lap-test",
        },
    )
    return training_settings, report_path, m6_payload


def test_regression_metrics_handle_missing_zero_and_undefined_values() -> None:
    metrics = regression_metrics(
        [100.0, 0.0, float("nan"), 50.0],
        [110.0, 5.0, 20.0, float("inf")],
    )

    assert metrics["valid_rows"] == 2
    assert metrics["excluded_rows"] == 2
    assert metrics["zero_target_rows"] == 1
    assert metrics["mae"] == pytest.approx(7.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(62.5))
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.975)
    assert regression_metrics([0.0, float("nan")], [2.0, 1.0])["mape"] is None
    assert regression_metrics([4.0], [3.0])["r2"] is None


def test_training_refuses_an_invalid_m6_report(synthetic_m6) -> None:
    settings, m6_report_path, m6_payload = synthetic_m6
    m6_payload["verdict"] = "INVALID"
    m6_report_path.write_text(json.dumps(m6_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not trainable"):
        TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    assert not Path(settings.training.tracking_db_path).exists()


def test_training_is_reproducible_tracked_registered_and_recoverable(synthetic_m6) -> None:
    pytest.importorskip("mlflow", reason="install the optional training extra to run MLflow integration tests")
    settings, m6_report_path, m6_payload = synthetic_m6
    original_report_hash = hashlib.sha256(m6_report_path.read_bytes()).hexdigest()

    first = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()
    second = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    assert first.metrics == second.metrics
    assert first.input_fingerprint == m6_payload["source"]["input_fingerprint"]
    assert first.m6_run_id == m6_payload["run_id"]
    assert first.model_version != second.model_version
    assert hashlib.sha256(m6_report_path.read_bytes()).hexdigest() == original_report_hash
    assert first.metrics["test"]["linear_regression"]["mae"] is not None
    assert first.metrics["test"]["reference_mean"]["mae"] is not None

    from mlflow.sklearn import load_model
    from mlflow.tracking import MlflowClient
    import mlflow

    tracking_uri = f"sqlite:///{(Path(settings.training.tracking_db_path).resolve()).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    restored = load_model(first.model_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    run = client.get_run(first.run_id)
    assert run.data.tags["input_fingerprint"] == first.input_fingerprint
    assert run.data.tags["m6_run_id"] == first.m6_run_id
    assert run.data.metrics["test.linear_regression.mae"] == pytest.approx(
        first.metrics["test"]["linear_regression"]["mae"]
    )

    test_artifact = next(
        item
        for item in m6_payload["artifact_files"]
        if item["split"] == "test" and item["artifact"] == "features.parquet"
    )
    matrix = pd.read_parquet(test_artifact["path"])
    columns = m6_payload["preprocessing"]["output_feature_names"]
    recovered_prediction = restored.predict(matrix.loc[:, columns].head(1))
    assert np.isfinite(recovered_prediction).all()

    saved_report = json.loads(Path(first.report_path).read_text(encoding="utf-8"))
    assert saved_report["registered_model"]["recovery_check"] == "passed"
    assert saved_report["input_fingerprint"] == first.input_fingerprint


def test_training_explains_when_optional_extra_is_missing(synthetic_m6, monkeypatch) -> None:
    import builtins

    settings, m6_report_path, _ = synthetic_m6
    real_import = builtins.__import__

    def without_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlflow)
    with pytest.raises(RuntimeError, match=r"\[training\]"):
        TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()


# -- M7 deep learning (model_type="mlp") -----------------------------------------

TORCH_HINT = 'python -m pip install -e ".[training,dl]"'
MLP_REGISTERED_NAME = "synthetic-next-lap-mlp-test"
FAST_MLP = {
    "registered_model_name": MLP_REGISTERED_NAME,
    "hidden_dims": [8, 4],
    "dropout": 0.0,
    "learning_rate": 0.01,
    "batch_size": 4,
    "max_epochs": 6,
    "early_stopping_patience": 3,
}
MODEL_FAMILIES = {"mlp", "linear_regression", "reference_mean"}


def _with_training(settings: AppSettings, **training: Any) -> AppSettings:
    data = settings.model_dump()
    data["training"].update(training)
    return AppSettings(**data)


def _block_imports(monkeypatch, *modules: str) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if any(name == module or name.startswith(f"{module}.") for module in modules):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def _train_targets(m6_payload: Dict[str, Any]) -> np.ndarray:
    train_artifact = next(
        item
        for item in m6_payload["artifact_files"]
        if item["split"] == "train" and item["artifact"] == "features.parquet"
    )
    targets = pd.read_parquet(train_artifact["path"])["next_lap_time"].to_numpy(dtype="float64")
    return targets[np.isfinite(targets)]


def test_training_runner_rejects_an_unknown_model_type(synthetic_m6) -> None:
    settings, m6_report_path, _ = synthetic_m6

    with pytest.raises(ValueError, match="Unsupported M7 model_type 'random_forest'"):
        TrainingRunner(settings, m6_report_path=str(m6_report_path), model_type="random_forest")


def test_mlp_training_is_reproducible_tracked_registered_and_recoverable(synthetic_m6) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    pytest.importorskip("mlflow", reason="install the optional training extra")
    base_settings, m6_report_path, m6_payload = synthetic_m6
    settings = _with_training(base_settings, model_type="mlp", mlp=FAST_MLP)
    original_report_hash = hashlib.sha256(m6_report_path.read_bytes()).hexdigest()

    first = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()
    second = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    # reproducible: same seed, same data -> identical metrics for every model family
    assert first.metrics == second.metrics
    assert first.model_type == second.model_type == "mlp"
    assert first.model_name == second.model_name == MLP_REGISTERED_NAME
    assert first.model_uri == f"models:/{MLP_REGISTERED_NAME}/{first.model_version}"
    assert first.model_version != second.model_version
    assert first.input_fingerprint == m6_payload["source"]["input_fingerprint"]
    assert first.m6_run_id == m6_payload["run_id"]
    assert first.summary()["model_type"] == "mlp"
    assert hashlib.sha256(m6_report_path.read_bytes()).hexdigest() == original_report_hash

    # comparative baselines are scored on every split in the same run
    for split in ("train", "validation", "test"):
        assert set(first.metrics[split]) == MODEL_FAMILIES
        for family in MODEL_FAMILIES:
            assert first.metrics[split][family]["mae"] is not None

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{(Path(settings.training.tracking_db_path).resolve()).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    run = client.get_run(first.run_id)
    saved_report = json.loads(Path(first.report_path).read_text(encoding="utf-8"))
    mlp_training = saved_report["mlp_training"]

    assert run.info.run_name.startswith("m7-mlp-")
    assert run.data.tags["model_type"] == "mlp"
    assert run.data.tags["input_fingerprint"] == first.input_fingerprint
    assert run.data.tags["m6_run_id"] == first.m6_run_id
    params = run.data.params
    assert params["model_type"] == "mlp"
    assert params["estimator"] == "f1_race_intelligence.modeling.dl.TabularMLP"
    assert json.loads(params["mlp_hidden_dims"]) == FAST_MLP["hidden_dims"]
    assert int(params["mlp_batch_size"]) == FAST_MLP["batch_size"]
    assert int(params["mlp_max_epochs"]) == FAST_MLP["max_epochs"]
    assert params["mlp_loss"] == "huber"
    assert params["mlp_scale_target"] == "True"
    assert int(params["mlp_best_epoch"]) == mlp_training["best_epoch"]
    assert int(params["mlp_epochs_trained"]) == mlp_training["epochs_trained"]
    assert int(params["mlp_total_params"]) == mlp_training["total_params"] > 0

    # per-epoch learning curves, one point per trained epoch
    epochs = mlp_training["epochs_trained"]
    assert 1 <= mlp_training["best_epoch"] <= epochs <= FAST_MLP["max_epochs"]
    train_curve = client.get_metric_history(first.run_id, "train.mlp.loss")
    val_curve = client.get_metric_history(first.run_id, "validation.mlp.loss")
    assert sorted(point.step for point in train_curve) == list(range(1, epochs + 1))
    assert sorted(point.step for point in val_curve) == list(range(1, epochs + 1))
    assert all(np.isfinite(point.value) for point in train_curve + val_curve)
    best_val = min(point.value for point in val_curve)
    assert mlp_training["best_val_loss"] == pytest.approx(best_val)

    # final per-split metrics for the MLP and both baselines
    for split in ("train", "validation", "test"):
        for family in MODEL_FAMILIES:
            assert run.data.metrics[f"{split}.{family}.mae"] == pytest.approx(first.metrics[split][family]["mae"])

    artifacts = {item.path for item in client.list_artifacts(first.run_id)}
    # (MLflow 3 stores the model itself as a LoggedModel, outside this listing)
    assert {
        "m6",
        "training_config.json",
        "data_provenance.json",
        "metrics.json",
        "training_report.json",
        "training_history.json",
    } <= artifacts
    history = mlflow.artifacts.load_dict(f"runs:/{first.run_id}/training_history.json")
    assert [entry["epoch"] for entry in history] == list(range(1, epochs + 1))

    # report and config snapshot keep the M7 schema and add the MLP section
    assert saved_report["model_type"] == "mlp"
    assert saved_report["config"]["model_type"] == "mlp"
    assert saved_report["config"]["registered_model_name"] == MLP_REGISTERED_NAME
    assert saved_report["config"]["mlp"]["hidden_dims"] == FAST_MLP["hidden_dims"]
    assert saved_report["config"]["mlp"]["random_seed"] == settings.modeling.random_seed
    assert saved_report["registered_model"]["recovery_check"] == "passed"
    assert saved_report["registered_model"]["uri"] == first.model_uri
    assert saved_report["features"] == m6_payload["preprocessing"]["output_feature_names"]

    # anti-leakage: target scaling statistics come from the finite train targets only
    train_targets = _train_targets(m6_payload)
    assert mlp_training["y_mean"] == pytest.approx(float(np.mean(train_targets)))
    assert mlp_training["y_std"] == pytest.approx(float(np.std(train_targets)))

    # the registered model is a pyfunc (not sklearn) answering in seconds
    model = mlflow.pyfunc.load_model(first.model_uri)
    assert model.metadata.run_id == first.run_id
    assert "sklearn" not in model.metadata.flavors
    columns = m6_payload["preprocessing"]["output_feature_names"]
    assert list(model.metadata.get_input_schema().input_names()) == columns
    test_artifact = next(
        item
        for item in m6_payload["artifact_files"]
        if item["split"] == "test" and item["artifact"] == "features.parquet"
    )
    matrix = pd.read_parquet(test_artifact["path"]).loc[:, columns].astype("float64")
    predictions = np.asarray(model.predict(matrix))
    assert predictions.dtype == np.float64
    assert predictions.shape == (len(matrix),)
    assert np.isfinite(predictions).all()
    assert np.all((predictions > 80.0) & (predictions < 110.0))


def test_mlp_training_explains_when_torch_is_missing(synthetic_m6, monkeypatch) -> None:
    # Runs in every environment, including base-runtime without torch or MLflow.
    base_settings, m6_report_path, _ = synthetic_m6
    settings = _with_training(base_settings, model_type="mlp", mlp=FAST_MLP)
    _block_imports(monkeypatch, "torch")

    with pytest.raises(RuntimeError, match=re_escape(TORCH_HINT)) as excinfo:
        TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    assert "PyTorch" in str(excinfo.value)
    # training fails before any MLflow tracking state is created
    assert not Path(settings.training.tracking_db_path).exists()
    assert not Path(settings.training.output_path).exists()


def test_model_type_argument_explains_when_torch_is_missing(synthetic_m6, monkeypatch) -> None:
    base_settings, m6_report_path, _ = synthetic_m6
    _block_imports(monkeypatch, "torch")

    with pytest.raises(RuntimeError, match=re_escape(TORCH_HINT)):
        TrainingRunner(base_settings, m6_report_path=str(m6_report_path), model_type="mlp").run()


def test_linear_path_never_needs_torch_even_without_any_optional_extra(synthetic_m6, monkeypatch) -> None:
    # Runs everywhere: with torch and MLflow both missing, the linear baseline
    # reaches the MLflow requirement (the training extra), never a torch error.
    settings, m6_report_path, _ = synthetic_m6
    _block_imports(monkeypatch, "torch", "mlflow")

    with pytest.raises(RuntimeError, match=r"\[training\]") as excinfo:
        TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    message = str(excinfo.value)
    assert ".[training]" in message
    assert "torch" not in message.lower() and "dl]" not in message


def test_linear_regression_still_trains_and_registers_when_torch_is_missing(synthetic_m6, monkeypatch) -> None:
    pytest.importorskip("mlflow", reason="install the optional training extra")
    settings, m6_report_path, m6_payload = synthetic_m6
    _block_imports(monkeypatch, "torch")

    result = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    assert result.model_type == "linear_regression"
    assert result.model_name == settings.training.registered_model_name
    assert set(result.metrics["test"]) == {"linear_regression", "reference_mean"}
    assert result.metrics["test"]["linear_regression"]["mae"] is not None
    saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert saved_report["model_type"] == "linear_regression"
    assert saved_report["config"]["estimator"] == "sklearn.linear_model.LinearRegression"
    assert "mlp" not in saved_report["config"]
    assert "mlp_training" not in saved_report
    assert saved_report["registered_model"]["recovery_check"] == "passed"
    assert result.m6_run_id == m6_payload["run_id"]


def test_mlp_run_reports_the_same_linear_baseline_as_a_linear_run(synthetic_m6) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    pytest.importorskip("mlflow", reason="install the optional training extra")
    base_settings, m6_report_path, _ = synthetic_m6
    mlp_settings = _with_training(base_settings, model_type="mlp", mlp=FAST_MLP)

    linear = TrainingRunner(base_settings, m6_report_path=str(m6_report_path)).run()
    mlp = TrainingRunner(mlp_settings, m6_report_path=str(m6_report_path)).run()

    assert linear.model_name != mlp.model_name
    for split in ("train", "validation", "test"):
        assert mlp.metrics[split]["linear_regression"] == linear.metrics[split]["linear_regression"]
        assert mlp.metrics[split]["reference_mean"] == linear.metrics[split]["reference_mean"]
        assert "mlp" not in linear.metrics[split]


# -- PR #25 refinements: PFI, LR scheduler, generalization, register_model=False --


@pytest.fixture
def mlflow_uris():
    """Yield MLflow and restore its global tracking/registry URIs afterwards."""
    mlflow = pytest.importorskip("mlflow", reason="install the optional training extra")
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


def _client_for(settings: AppSettings, mlflow):
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{(Path(settings.training.tracking_db_path).resolve()).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _top_features(interpretability: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ranked PFI entries of the report (``top_features`` or the ``importances`` list)."""
    entries = interpretability.get("top_features", interpretability.get("importances"))
    assert isinstance(entries, list)
    return entries


def test_mlp_training_logs_permutation_importance_lr_schedule_and_generalization(
    synthetic_m6, mlflow_uris
) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    mlflow = mlflow_uris
    base_settings, m6_report_path, m6_payload = synthetic_m6
    settings = _with_training(
        base_settings,
        model_type="mlp",
        mlp={
            **FAST_MLP,
            "lr_scheduler": "reduce_on_plateau",
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 1,
            "permutation_importance_repeats": 3,
            "permutation_importance_top_k": 4,
        },
    )

    result = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    client = _client_for(settings, mlflow)
    artifacts = {item.path for item in client.list_artifacts(result.run_id)}
    assert "permutation_importance.json" in artifacts
    pfi = mlflow.artifacts.load_dict(f"runs:/{result.run_id}/permutation_importance.json")
    features = m6_payload["preprocessing"]["output_feature_names"]
    assert pfi["split"] == "validation"
    assert pfi["n_repeats"] == 3
    # the logged artifact keeps every feature; the report keeps only the top-k
    assert pfi["n_features"] == len(features) == len(pfi["importances"])
    assert {entry["feature"] for entry in pfi["importances"]} == set(features)

    saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    interpretability = saved_report["interpretability"]
    assert interpretability["method"] == "permutation_feature_importance"
    assert interpretability["split"] == "validation"
    top = _top_features(interpretability)
    assert 0 < len(top) <= 4
    assert all(entry["feature"] in features for entry in top)
    assert all(np.isfinite(entry["mae_increase_mean"]) for entry in top)
    mae_increases = [entry["mae_increase_mean"] for entry in top]
    assert mae_increases == sorted(mae_increases, reverse=True)
    assert isinstance(interpretability["collinearity_note"], str) and interpretability["collinearity_note"]
    assert not any(name.startswith("key_") or name == "next_lap_time" for name in (e["feature"] for e in top))

    diagnosis = saved_report["generalization_diagnosis"]
    assert diagnosis["evaluated_splits"] == ["train", "validation"]
    assert diagnosis["status"] in {"underfitting", "overfitting_risk", "balanced", "insufficient_metrics"}
    assert diagnosis["train_mae"] == pytest.approx(result.metrics["train"]["mlp"]["mae"])
    assert diagnosis["validation_mae"] == pytest.approx(result.metrics["validation"]["mlp"]["mae"])
    assert diagnosis["epochs_trained"] == saved_report["mlp_training"]["epochs_trained"]
    assert isinstance(diagnosis["explanation"], str) and diagnosis["explanation"]

    # learning rate schedule: one train.mlp.lr point per epoch and in the history artifact
    epochs = saved_report["mlp_training"]["epochs_trained"]
    lr_curve = sorted(client.get_metric_history(result.run_id, "train.mlp.lr"), key=lambda point: point.step)
    assert [point.step for point in lr_curve] == list(range(1, epochs + 1))
    assert lr_curve[0].value == pytest.approx(FAST_MLP["learning_rate"])
    assert all(later.value <= earlier.value for earlier, later in zip(lr_curve, lr_curve[1:]))
    history = mlflow.artifacts.load_dict(f"runs:/{result.run_id}/training_history.json")
    assert all("learning_rate" in entry for entry in history)
    assert [entry["learning_rate"] for entry in history] == pytest.approx([point.value for point in lr_curve])
    assert saved_report["mlp_training"]["final_learning_rate"] == pytest.approx(history[-1]["learning_rate"])

    run = client.get_run(result.run_id)
    assert run.data.params["mlp_lr_scheduler"] == "reduce_on_plateau"
    assert saved_report["config"]["mlp"]["lr_scheduler"] == "reduce_on_plateau"
    assert saved_report["config"]["mlp"]["compute_permutation_importance"] is True
    assert saved_report["config"]["mlp"]["permutation_importance_top_k"] == 4


def test_mlp_training_can_disable_permutation_importance(synthetic_m6, mlflow_uris) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    mlflow = mlflow_uris
    base_settings, m6_report_path, _ = synthetic_m6
    settings = _with_training(
        base_settings,
        model_type="mlp",
        mlp={**FAST_MLP, "compute_permutation_importance": False, "lr_scheduler": "none"},
    )

    result = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert saved_report["interpretability"]["status"] == "disabled"
    assert "importances" not in saved_report["interpretability"]
    assert saved_report["generalization_diagnosis"]["evaluated_splits"] == ["train", "validation"]
    client = _client_for(settings, mlflow)
    artifacts = {item.path for item in client.list_artifacts(result.run_id)}
    assert "permutation_importance.json" not in artifacts
    assert "training_report.json" in artifacts
    # without a scheduler the learning rate stays constant
    lr_curve = client.get_metric_history(result.run_id, "train.mlp.lr")
    assert lr_curve and all(point.value == pytest.approx(FAST_MLP["learning_rate"]) for point in lr_curve)


def test_mlp_training_without_registration_keeps_the_registry_empty(synthetic_m6, mlflow_uris) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    mlflow = mlflow_uris
    base_settings, m6_report_path, _ = synthetic_m6
    settings = _with_training(base_settings, model_type="mlp", mlp=FAST_MLP)

    result = TrainingRunner(
        settings,
        m6_report_path=str(m6_report_path),
        register_model=False,
        extra_tags={"experiment_label": "unit-test"},
        run_name_prefix="m7-unregistered",
    ).run()

    assert result.model_version is None
    assert result.model_uri == f"runs:/{result.run_id}/model"
    saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    registered = saved_report["registered_model"]
    assert registered["recovery_check"] == "passed"
    assert registered["registered"] is False
    assert registered["version"] is None
    assert registered["uri"] == result.model_uri

    client = _client_for(settings, mlflow)
    assert client.search_model_versions(f"name='{MLP_REGISTERED_NAME}'") == []
    assert client.search_registered_models() == []
    run = client.get_run(result.run_id)
    assert run.info.run_name.startswith("m7-unregistered-")
    assert run.data.tags["experiment_label"] == "unit-test"
    assert run.data.tags["stage"] == "m7-training"

    # the unregistered run model is still loadable and answers in seconds
    model = mlflow.pyfunc.load_model(result.model_uri)
    assert model.metadata.run_id == result.run_id
    assert list(model.metadata.get_input_schema().input_names()) == saved_report["features"]


def test_linear_regression_report_has_no_dl_sections_and_needs_no_torch(
    synthetic_m6, mlflow_uris, monkeypatch
) -> None:
    mlflow = mlflow_uris
    settings, m6_report_path, _ = synthetic_m6
    _block_imports(monkeypatch, "torch")

    result = TrainingRunner(settings, m6_report_path=str(m6_report_path)).run()

    saved_report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert saved_report["model_type"] == "linear_regression"
    assert result.model_uri == f"models:/{settings.training.registered_model_name}/{result.model_version}"
    for key in ("interpretability", "generalization_diagnosis", "mlp_training"):
        assert key not in saved_report
    client = _client_for(settings, mlflow)
    artifacts = {item.path for item in client.list_artifacts(result.run_id)}
    assert "permutation_importance.json" not in artifacts
    assert "training_history.json" not in artifacts
    assert client.get_metric_history(result.run_id, "train.mlp.lr") == []
    assert len(client.search_model_versions(f"name='{settings.training.registered_model_name}'")) == 1
