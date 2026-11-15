"""M7 tests using synthetic M5-shaped rows and locally generated M6 artifacts."""

import hashlib
import json
from pathlib import Path
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
