"""M7 controlled MLP comparison (``MLPComparisonRunner`` / ``--stage m7-compare``).

Synthetic and offline: M5-shaped rows go through the real M6 runner, then the
comparison trains every candidate MLP on the same verified M6 report. Some
tests replace ``TrainingRunner`` with a fake so the selection policy
(``validation.mlp.mae`` only, never ``test``) is checked without PyTorch or
MLflow. The numbers only prove that the technical chain holds; they are not
academic results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
import yaml

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling import training as training_module
from f1_race_intelligence.modeling.experiments import (
    DEFAULT_MLP_EXPERIMENTS,
    MLPComparisonRunner,
    MLPExperimentSpec,
)
from serving_support import M6Run, build_m6_run, to_records

DEFAULT_NAMES = ["mlp_small", "mlp_default", "mlp_deeper_reg", "mlp_mse"]
COMPARE_REGISTERED_NAME = "synthetic-m7-compare-winner"
# Small and fast: the experiment specs override hidden_dims/dropout/weight_decay/loss.
FAST_MLP: Dict[str, Any] = {
    "learning_rate": 0.01,
    "batch_size": 8,
    "max_epochs": 8,
    "early_stopping_patience": 3,
    "lr_scheduler": "reduce_on_plateau",
    "lr_scheduler_patience": 2,
    "permutation_importance_repeats": 2,
    "permutation_importance_top_k": 3,
}
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch) -> None:
    for name in ("F1_ENVIRONMENT", "F1_STORAGE_BACKEND", "F1_STORAGE_BUCKET", "F1_STORAGE_LAYOUT", "F1_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def m6_run(tmp_path_factory) -> M6Run:
    return build_m6_run(tmp_path_factory.mktemp("m7-compare-m6"))


@pytest.fixture
def mlflow_uris() -> Iterator[Any]:
    """Yield MLflow and restore its global tracking/registry URIs afterwards."""
    mlflow = pytest.importorskip("mlflow", reason="install the optional training extra")
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


def _report_path(m6: M6Run) -> Path:
    (report_path,) = (m6.output_dir / "reports").glob("modeling_report_*.json")
    return report_path


def _training_section(root: Path, report_path: Path) -> Dict[str, Any]:
    return {
        "m6_report_path": str(report_path),
        "tracking_db_path": str(root / "mlflow" / "tracking.db"),
        "artifact_path": str(root / "mlflow" / "artifacts"),
        "output_path": str(root / "training-reports"),
        "experiment_name": "synthetic-m7-compare",
        "model_type": "mlp",
        "mlp": {**FAST_MLP, "registered_model_name": COMPARE_REGISTERED_NAME},
    }


def _settings(root: Path, report_path: Path) -> AppSettings:
    return AppSettings(training=_training_section(root, report_path))


def _client(settings: AppSettings, mlflow: Any):
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{Path(settings.training.tracking_db_path).resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _metrics(val_mae: float, val_rmse: float, test_mae: float, test_rmse: float) -> Dict[str, Any]:
    def family(mae: float, rmse: float) -> Dict[str, Any]:
        return {"mae": mae, "rmse": rmse, "r2": 0.5, "mape": 1.0, "valid_rows": 40}

    return {
        "train": {"mlp": family(0.5, 0.6), "reference_mean": family(3.0, 3.5), "linear_regression": family(0.7, 0.8)},
        "validation": {
            "mlp": family(val_mae, val_rmse),
            "reference_mean": family(3.0, 3.5),
            "linear_regression": family(0.9, 1.0),
        },
        "test": {
            "mlp": family(test_mae, test_rmse),
            "reference_mean": family(3.0, 3.5),
            "linear_regression": family(0.9, 1.0),
        },
    }


class FakeTrainingRunner:
    """Stand-in for ``TrainingRunner`` returning scripted metrics per experiment label."""

    scripted: Dict[str, Dict[str, Any]] = {}
    calls: List[Dict[str, Any]] = []

    def __init__(
        self,
        settings: AppSettings,
        *,
        m6_report_path: Optional[str] = None,
        model_type: Optional[str] = None,
        register_model: bool = True,
        extra_tags: Optional[Dict[str, str]] = None,
        run_name_prefix: Optional[str] = None,
    ) -> None:
        self._settings = settings
        self._label = (extra_tags or {})["experiment_label"]
        self._register = register_model
        FakeTrainingRunner.calls.append(
            {
                "label": self._label,
                "m6_report_path": m6_report_path,
                "model_type": model_type,
                "register_model": register_model,
                "extra_tags": dict(extra_tags or {}),
                "run_name_prefix": run_name_prefix,
                "hidden_dims": list(settings.training.mlp.hidden_dims),
                "loss": settings.training.mlp.loss,
            }
        )

    def run(self) -> training_module.TrainingResult:
        index = len(FakeTrainingRunner.calls)
        run_id = f"run-{index}-{self._label}"
        output_dir = Path(self._settings.training.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"training_report_{run_id}.json"
        metrics = FakeTrainingRunner.scripted[self._label]
        report = {
            "config": {"mlp": self._settings.training.mlp.model_dump()},
            "mlp_training": {"best_epoch": 2, "epochs_trained": 4, "total_params": 10, "final_learning_rate": 0.01},
            "generalization_diagnosis": {"status": "balanced", "evaluated_splits": ["train", "validation"]},
            "interpretability": {
                "method": "permutation_feature_importance",
                "split": "validation",
                "importances": [
                    {
                        "feature": "num__lap_number",
                        "mae_increase_mean": 0.3,
                        "mae_increase_std": 0.1,
                        "rmse_increase_mean": 0.4,
                        "rmse_increase_std": 0.1,
                    }
                ],
                "collinearity_note": "lags are correlated",
            },
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        name = self._settings.training.effective_registered_model_name("mlp")
        version = "1" if self._register else None
        return training_module.TrainingResult(
            run_id=run_id,
            m6_run_id="m6-run",
            input_fingerprint="sha256:" + "0" * 64,
            model_name=name,
            model_version=version,
            model_uri=f"models:/{name}/1" if self._register else f"runs:/{run_id}/model",
            metrics=metrics,
            report_path=str(report_path),
            source_commit=None,
            model_type="mlp",
        )


@pytest.fixture
def fake_training(monkeypatch) -> type:
    FakeTrainingRunner.scripted = {}
    FakeTrainingRunner.calls = []
    monkeypatch.setattr(training_module, "TrainingRunner", FakeTrainingRunner)
    return FakeTrainingRunner


# --------------------------------------------------------------------------- #
# Without PyTorch: lazy imports, input validation and the selection policy
# --------------------------------------------------------------------------- #


def test_experiments_module_imports_without_torch_or_mlflow() -> None:
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch', 'mlflow'}:\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from f1_race_intelligence.modeling import experiments\n"
        "assert [spec.name for spec in experiments.DEFAULT_MLP_EXPERIMENTS] == "
        f"{DEFAULT_NAMES!r}\n"
        "assert experiments.SELECTION_METRIC == 'validation.mlp.mae'\n"
        "assert 'torch' not in sys.modules and 'mlflow' not in sys.modules\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC_ROOT), os.environ.get("PYTHONPATH", "")])}
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_default_experiments_are_the_four_documented_configurations() -> None:
    assert [spec.name for spec in DEFAULT_MLP_EXPERIMENTS] == DEFAULT_NAMES
    assert all(spec.description for spec in DEFAULT_MLP_EXPERIMENTS)
    assert {spec.mlp_overrides["loss"] for spec in DEFAULT_MLP_EXPERIMENTS} == {"huber", "mse"}


def test_comparison_rejects_an_invalid_m6_report_before_importing_torch(tmp_path, monkeypatch) -> None:
    import builtins

    m6 = build_m6_run(tmp_path / "m6")
    report_path = _report_path(m6)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["verdict"] = "INVALID"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    settings = _settings(tmp_path, report_path)

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "mlflow"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="not trainable"):
        MLPComparisonRunner(settings, m6_report_path=str(report_path)).run()

    assert not Path(settings.training.tracking_db_path).exists()
    assert not (Path(settings.training.output_path) / "comparisons").exists()


def test_comparison_rejects_missing_remote_or_non_local_inputs(tmp_path) -> None:
    settings = _settings(tmp_path, tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="M6 report not found"):
        MLPComparisonRunner(settings, m6_report_path=str(tmp_path / "missing.json")).run()
    with pytest.raises(ValueError, match="local M6 report only"):
        MLPComparisonRunner(settings, m6_report_path="s3://bucket/report.json").run()
    s3_settings = AppSettings(storage={"backend": "s3", "bucket": "b"}, training=settings.training.model_dump())
    with pytest.raises(ValueError, match="local storage only"):
        MLPComparisonRunner(s3_settings, m6_report_path=str(tmp_path / "missing.json")).run()


def test_comparison_rejects_empty_or_duplicate_experiment_specs(tmp_path) -> None:
    settings = _settings(tmp_path, tmp_path / "report.json")
    spec = MLPExperimentSpec(name="a", description="a", mlp_overrides={})

    with pytest.raises(ValueError, match="at least one"):
        MLPComparisonRunner(settings, experiments=[])
    with pytest.raises(ValueError, match="duplicate"):
        MLPComparisonRunner(settings, experiments=[spec, spec])
    with pytest.raises(TypeError, match="MLPExperimentSpec"):
        MLPComparisonRunner(settings, experiments=[{"name": "a"}])


def test_winner_is_selected_on_validation_mae_never_on_test(m6_run, tmp_path, fake_training) -> None:
    report_path = _report_path(m6_run)
    specs = [
        MLPExperimentSpec("test_favourite", "best on test, worse on validation", {"hidden_dims": [8]}),
        MLPExperimentSpec("validation_winner", "best on validation, worse on test", {"hidden_dims": [16, 8]}),
        MLPExperimentSpec("middle", "in between", {"hidden_dims": [4], "loss": "mse"}),
    ]
    fake_training.scripted = {
        "test_favourite": _metrics(val_mae=2.0, val_rmse=2.5, test_mae=0.01, test_rmse=0.02),
        "validation_winner": _metrics(val_mae=1.0, val_rmse=1.5, test_mae=9.0, test_rmse=9.5),
        "middle": _metrics(val_mae=1.5, val_rmse=1.8, test_mae=5.0, test_rmse=5.5),
    }

    result = MLPComparisonRunner(
        _settings(tmp_path, report_path), m6_report_path=str(report_path), experiments=specs
    ).run()

    assert result.best_experiment == "validation_winner"
    assert result.selection_metric == "validation.mlp.mae"
    assert [cand["name"] for cand in result.candidates] == ["validation_winner", "middle", "test_favourite"]
    assert [cand["rank"] for cand in result.candidates] == [1, 2, 3]
    assert result.registered_winner is None
    # every candidate trained once, unregistered, on the same report with its own overrides
    assert [call["label"] for call in fake_training.calls] == ["test_favourite", "validation_winner", "middle"]
    assert all(call["register_model"] is False for call in fake_training.calls)
    assert all(call["model_type"] == "mlp" for call in fake_training.calls)
    assert {call["m6_report_path"] for call in fake_training.calls} == {str(report_path.resolve())}
    assert [call["hidden_dims"] for call in fake_training.calls] == [[8], [16, 8], [4]]
    assert len({call["extra_tags"]["comparison_id"] for call in fake_training.calls}) == 1

    payload = json.loads(Path(result.json_report_path).read_text(encoding="utf-8"))
    assert payload["best_experiment"] == "validation_winner"
    assert payload["selection_policy"]["selection_metric"] == "validation.mlp.mae"
    assert payload["selection_policy"]["tie_breaker"] == "validation.mlp.rmse"
    assert "never used for model selection" in payload["selection_policy"]["anti_snooping"]
    markdown = Path(result.markdown_report_path).read_text(encoding="utf-8")
    assert "Anti-Snooping Policy" in markdown
    assert "**validation_winner** (selected)" in markdown


def test_validation_rmse_breaks_ties_and_test_still_never_decides(m6_run, tmp_path, fake_training) -> None:
    report_path = _report_path(m6_run)
    specs = [
        MLPExperimentSpec("tie_worse_rmse", "same MAE, worse RMSE, best test", {"hidden_dims": [8]}),
        MLPExperimentSpec("tie_better_rmse", "same MAE, better RMSE, worst test", {"hidden_dims": [8]}),
    ]
    fake_training.scripted = {
        "tie_worse_rmse": _metrics(val_mae=1.0, val_rmse=2.0, test_mae=0.01, test_rmse=0.01),
        "tie_better_rmse": _metrics(val_mae=1.0, val_rmse=1.2, test_mae=9.0, test_rmse=9.0),
    }

    result = MLPComparisonRunner(
        _settings(tmp_path, report_path), m6_report_path=str(report_path), experiments=specs
    ).run()

    assert result.best_experiment == "tie_better_rmse"


def test_register_best_retrains_and_registers_only_the_validation_winner(m6_run, tmp_path, fake_training) -> None:
    report_path = _report_path(m6_run)
    fake_training.scripted = {
        "mlp_small": _metrics(val_mae=1.4, val_rmse=1.6, test_mae=0.1, test_rmse=0.2),
        "mlp_default": _metrics(val_mae=1.3, val_rmse=1.5, test_mae=3.0, test_rmse=3.2),
        "mlp_deeper_reg": _metrics(val_mae=1.1, val_rmse=1.4, test_mae=4.0, test_rmse=4.1),
        "mlp_mse": _metrics(val_mae=1.2, val_rmse=1.3, test_mae=0.05, test_rmse=0.06),
    }

    result = MLPComparisonRunner(
        _settings(tmp_path, report_path), m6_report_path=str(report_path), register_best=True
    ).run()

    assert result.best_experiment == "mlp_deeper_reg"
    registrations = [call for call in fake_training.calls if call["register_model"]]
    assert len(fake_training.calls) == len(DEFAULT_NAMES) + 1
    assert len(registrations) == 1
    (winner_call,) = registrations
    assert winner_call["label"] == "mlp_deeper_reg"
    assert winner_call["hidden_dims"] == [128, 64, 32]
    assert winner_call["extra_tags"]["comparison_winner"] == "true"
    assert result.registered_winner is not None
    assert result.registered_winner["experiment"] == "mlp_deeper_reg"
    assert result.registered_winner["model_name"] == COMPARE_REGISTERED_NAME
    assert result.summary()["registered_winner"]["model_version"] == "1"


# --------------------------------------------------------------------------- #
# Real PyTorch + MLflow sweeps
# --------------------------------------------------------------------------- #


def test_real_comparison_runs_four_configs_without_touching_the_registry(m6_run, tmp_path, mlflow_uris) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    mlflow = mlflow_uris
    report_path = _report_path(m6_run)
    m6_report = json.loads(report_path.read_text(encoding="utf-8"))
    fingerprint = m6_report["source"]["input_fingerprint"]
    settings = _settings(tmp_path, report_path)

    result = MLPComparisonRunner(settings, m6_report_path=str(report_path), register_best=False).run()

    assert result.input_fingerprint == fingerprint
    assert result.m6_run_id == m6_report["run_id"]
    assert sorted(cand["name"] for cand in result.candidates) == sorted(DEFAULT_NAMES)
    assert [cand["rank"] for cand in result.candidates] == [1, 2, 3, 4]
    assert result.registered_winner is None
    assert all(cand["model_version"] is None for cand in result.candidates)
    assert all(cand["model_uri"] == f"runs:/{cand['run_id']}/model" for cand in result.candidates)
    val_maes = [cand["metrics"]["validation"]["mlp"]["mae"] for cand in result.candidates]
    assert val_maes == sorted(val_maes)
    assert result.best_experiment == result.candidates[0]["name"]

    client = _client(settings, mlflow)
    assert client.search_model_versions(f"name='{COMPARE_REGISTERED_NAME}'") == []
    assert client.search_registered_models() == []
    comparison_ids = set()
    for cand in result.candidates:
        run = client.get_run(cand["run_id"])
        assert run.data.tags["input_fingerprint"] == fingerprint
        assert run.data.tags["experiment_label"] == cand["name"]
        assert run.info.run_name.startswith(f"m7-cmp-{cand['name']}-")
        comparison_ids.add(run.data.tags["comparison_id"])
        saved = json.loads(Path(cand["report_path"]).read_text(encoding="utf-8"))
        assert saved["input_fingerprint"] == fingerprint
        assert saved["registered_model"]["registered"] is False
        assert saved["generalization_diagnosis"]["evaluated_splits"] == ["train", "validation"]
        assert saved["interpretability"]["split"] == "validation"
    assert comparison_ids == {result.comparison_id}

    json_path = Path(result.json_report_path)
    md_path = Path(result.markdown_report_path)
    assert json_path.name == f"mlp_comparison_{result.comparison_id}.json"
    assert md_path.name == f"mlp_comparison_{result.comparison_id}.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["m6_provenance"]["input_fingerprint"] == fingerprint
    assert set(payload["m6_provenance"]["m6_splits"]) == {"train", "validation", "test"}
    assert all(payload["m6_provenance"]["m6_splits"][split]["sessions"] >= 1 for split in ("train", "validation", "test"))
    assert payload["registered_winner"] is None
    assert [cand["name"] for cand in payload["candidates"]] == [cand["name"] for cand in result.candidates]
    assert set(payload["baselines"]["validation"]) == {"reference_mean", "linear_regression"}

    markdown = md_path.read_text(encoding="utf-8")
    assert "session(s)" in markdown
    assert "## 2. Comparative Metrics" in markdown
    assert "| Model / Config | Train MAE |" in markdown
    assert "Baseline: `DummyRegressor(mean)`" in markdown and "Baseline: `LinearRegression`" in markdown
    assert "## 3. Overfitting / Underfitting Diagnosis (`train` vs `validation`)" in markdown
    assert "## 4. Interpretability — Permutation Feature Importance" in markdown
    assert "| Rank | Encoded Feature | MAE Increase" in markdown
    winner_interp = result.candidates[0]["interpretability"]
    top = winner_interp.get("top_features", winner_interp.get("importances"))
    assert 0 < len(top) <= FAST_MLP["permutation_importance_top_k"]
    assert all(f"`{entry['feature']}`" in markdown for entry in top)
    assert "No model was registered" in markdown
    for name in DEFAULT_NAMES:
        assert f"MLP: `{name}`" in markdown or f"MLP: **`{name}`** (winner)" in markdown
        diagnosis = next(cand for cand in result.candidates if cand["name"] == name)["generalization_diagnosis"]
        assert f"| `{name}` | `{diagnosis['status']}` |" in markdown


def test_cli_m7_compare_registers_only_the_winner_which_m8_serves(
    m6_run, tmp_path, mlflow_uris, monkeypatch, capsys
) -> None:
    pytest.importorskip("torch", reason="install the optional dl extra")
    pytest.importorskip("fastapi", reason="install the optional serve extra")
    from fastapi.testclient import TestClient
    from sklearn.compose import ColumnTransformer

    from f1_race_intelligence import cli
    from f1_race_intelligence.modeling.model_package import publish_model
    from f1_race_intelligence.serving.app import create_app
    from f1_race_intelligence.serving.model import MlflowPyfuncLoader
    from f1_race_intelligence.serving.service import PredictionService
    from f1_race_intelligence.storage.backends.local import LocalStorageBackend

    mlflow = mlflow_uris
    report_path = _report_path(m6_run)
    training = _training_section(tmp_path, report_path)
    training["mlp"].pop("registered_model_name")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"training": training}), encoding="utf-8")
    cli_name = "cli-m7-compare-winner"

    code = cli.main(
        [
            "--stage",
            "m7-compare",
            "--m6-report",
            str(report_path),
            "--register-best",
            "--registered-model-name",
            cli_name,
            "--config",
            str(config_file),
        ]
    )

    assert code == cli.EXIT_OK
    assert cli_name in capsys.readouterr().err
    (json_report,) = (tmp_path / "training-reports" / "comparisons").glob("mlp_comparison_*.json")
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    winner = payload["registered_winner"]
    assert winner is not None
    assert winner["experiment"] == payload["best_experiment"] == payload["candidates"][0]["name"]
    assert winner["model_name"] == cli_name
    assert winner["model_uri"] == f"models:/{cli_name}/{winner['model_version']}"
    assert (tmp_path / "training-reports" / "comparisons" / f"{json_report.stem}.md").is_file()

    settings = AppSettings(training=training)
    client = _client(settings, mlflow)
    versions = client.search_model_versions(f"name='{cli_name}'")
    assert len(versions) == 1
    (version,) = versions
    assert version.run_id == winner["run_id"]
    assert [model.name for model in client.search_registered_models()] == [cli_name]
    run = client.get_run(winner["run_id"])
    assert run.data.tags["comparison_winner"] == "true"
    assert run.data.tags["experiment_label"] == winner["experiment"]
    assert run.data.tags["comparison_id"] == payload["comparison_id"]
    assert winner["run_id"] not in {cand["run_id"] for cand in payload["candidates"]}

    # the registered winner is publishable and served by M8 without refitting M6
    def forbidden_fit(*args, **kwargs):
        pytest.fail("M8 must only transform with the M6 preprocessor; it must never fit it")

    monkeypatch.setattr(ColumnTransformer, "fit", forbidden_fit)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden_fit)
    backend = LocalStorageBackend(tmp_path / "model-store")
    package = publish_model(winner["model_uri"], backend)
    assert package.manifest["model"]["m7_run_id"] == winner["run_id"]
    assert package.manifest["provenance"]["input_fingerprint"] == payload["m6_provenance"]["input_fingerprint"]
    service = PredictionService.load_published_package(
        backend=backend, prefix=package.prefix, loader=MlflowPyfuncLoader()
    )
    rows = m6_run.split_file("test", "dataset.parquet").head(3)
    records = to_records(rows, list(m6_run.numeric) + list(m6_run.categorical))
    direct = service.predict(records)
    with TestClient(create_app(lambda: service)) as http:
        response = http.post("/predict", json={"records": records})

    assert response.status_code == 200
    values = [item["value"] for item in response.json()["predictions"]]
    assert len(values) == len(records)
    assert all(50.0 < value < 150.0 for value in values)
    assert values == pytest.approx(list(direct), rel=1e-9, abs=1e-9)
