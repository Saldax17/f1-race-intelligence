"""Demostración local de la cadena MLOps M1-M8 con datos sintéticos (sin red).

Usa el código real del proyecto y un cliente OpenF1 *simulado* (``SeasonClient``,
de ``tests/unit/test_backfill.py``); no llama a OpenF1 ni a AWS. Secuencia:

    plan (F1_EXTRACTION_YEARS) -> M2..M5 por temporada -> M6 --require-complete-history
    (diferido con histórico incompleto, ejecutado con el completo) -> M7 (CLI)
    -> MLflow (SQLite local) -> publicación -> M8 (/live, /health, /model-info, /predict)

``test_integral_flow`` usa el baseline lineal por defecto. ``test_integral_dl_flow``
recorre la misma cadena con Deep Learning (``--stage m7 --model-type mlp``: MLP
tabular PyTorch CPU-only registrado como ``mlflow.pyfunc``), publica el paquete
inmutable en almacenamiento local y en un S3 simulado en memoria, y lo sirve con
M8 sin reajustar el preprocesador. Se omite si PyTorch (extra ``dl``) no está
instalado. Las métricas sintéticas solo verifican la cadena técnica; no son
resultados académicos.

Ejecución (desde la raíz del repositorio; ver DEMO_LOCAL.md):

    python -m pytest scripts/demo_local_mlops.py -s -p no:cacheprovider -p no:warnings

Todo se escribe en directorios temporales de pytest; el repositorio no cambia.
"""
import hashlib
import json
import sys
from pathlib import Path

# Los helpers de la demo (cliente simulado, fábricas de datos) viven en tests/unit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "unit"))

import mlflow
import pytest
import yaml
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from test_backfill import (  # noqa: F401 - config e isolated son fixtures de pytest
    SEASONS,
    SeasonClient,
    cli,
    config,
    history,
    isolated,
    load_settings,
    m6,
    modeling_reports,
    run_phase,
)
from f1_race_intelligence.modeling.model_package import PUBLISHED_FILE, publish_model, read_model_package
from f1_race_intelligence.modeling.verification import verify_report
from f1_race_intelligence.serving.app import create_app
from f1_race_intelligence.serving.model import MlflowPyfuncLoader
from f1_race_intelligence.serving.service import PredictionService
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
from s3_fakes import FakeS3Client

NAME = "integral-demo-model"
DL_NAME = "integral-dl-demo-model"


def say(tag, value):
    print(f"[DEMO] {tag}: {value}")


def test_integral_flow(config, tmp_path, monkeypatch, capsys):
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["training"] = {
        "tracking_db_path": str(tmp_path / "mlflow" / "tracking.db"),
        "artifact_path": str(tmp_path / "mlflow" / "artifacts"),
        "output_path": str(tmp_path / "training-reports"),
        "experiment_name": "integral-demo",
        "registered_model_name": NAME,
    }
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    client = SeasonClient()

    # 1-2. input preparation/validation and M2-M5, phase by phase; 4. M6 deferred then run
    exit_codes = {}
    for year in SEASONS:
        plan, results = run_phase(config, tmp_path, client, str(year), monkeypatch)
        assert all(codes == [0, 0, 0, 0] for codes in results.values())
        before = len(modeling_reports(tmp_path))
        exit_codes[year] = m6(config, "--require-complete-history")
        say(f"phase {year}", f"sessions planned={len(plan['sessions'])}; m2..m5 exit codes={sorted(set(map(tuple, results.values())))}; m6 exit={exit_codes[year]}; m6 reports so far={len(modeling_reports(tmp_path))} (before {before})")
    assert exit_codes == {2023: 3, 2024: 3, 2025: 0}
    state = history(config)
    say("history", {y: (s["status"], s["sessions"]) for y, s in state["seasons"].items()})

    # 3. M6 dataset + integrity
    (report_path,) = modeling_reports(tmp_path)
    m6_report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = verify_report(str(report_path), storage=load_settings(str(config)).storage)
    say("M6", f"run_id={m6_report['run_id']} verdict={m6_report['verdict']} sessions={m6_report['source']['sessions']} fingerprint={m6_report['source']['input_fingerprint'][:23]}... verify_report ok={integrity['ok']} artifacts={len(m6_report['artifact_files'])}")
    assert integrity["ok"] and m6_report["verdict"] in {"VALID", "VALID_WITH_WARNINGS"}

    # 5. M7 through the real CLI
    code = cli.main(["--stage", "m7", "--m6-report", str(report_path), "--config", str(config)])
    say("M7 exit", code)
    assert code == 0
    reports = sorted((tmp_path / "training-reports").rglob("*.json"))
    training = json.loads(next(r for r in reports if json.loads(r.read_text())["run_id"]).read_text(encoding="utf-8"))
    say("M7 report", {k: training.get(k) for k in ("run_id", "m6_run_id", "input_fingerprint", "target")} | {"registered_model": training.get("registered_model")})
    say("M7 metrics", json.dumps(training.get("metrics", training.get("models", "n/a")))[:400])

    # 6. MLflow: registry + artifacts retrieved
    uri = f"sqlite:///{(tmp_path / 'mlflow' / 'tracking.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mc = MlflowClient()
    versions = mc.search_model_versions(f"name='{NAME}'")
    say("MLflow registry", [(v.name, v.version, v.status, v.run_id) for v in versions])
    mv = mc.get_model_version(NAME, "1")
    run = mc.get_run(mv.run_id)
    say("MLflow run", f"status={run.info.status} metrics={dict(list(run.data.metrics.items())[:4])}")
    artifacts = sorted(a.path for a in mc.list_artifacts(mv.run_id))
    say("MLflow artifacts", artifacts)
    assert mv.run_id == training["run_id"] and run.info.status == "FINISHED"
    local_pre = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{mv.run_id}/m6/preprocessor.joblib")
    pre_hash = hashlib.sha256(open(local_pre, "rb").read()).hexdigest()

    # 7. Publication
    store = LocalStorageBackend(tmp_path / "model-store")
    package = publish_model(f"models:/{NAME}/1", store)
    again = publish_model(f"models:/{NAME}/1", store)
    manifest = package.manifest
    say("package", f"prefix={package.prefix} files={sorted(manifest['files'])}")
    say("manifest model", manifest["model"])
    say("manifest provenance", {k: manifest["provenance"][k] for k in ("m6_run_id", "input_fingerprint", "data_version")})
    say("PUBLISHED marker", store.read_bytes(f"{package.prefix}/{PUBLISHED_FILE}").decode().strip())
    assert again.manifest_sha256 == package.manifest_sha256  # idempotent republication
    assert read_model_package(store, package.prefix).manifest_sha256 == package.manifest_sha256
    assert manifest["files"]["m6/preprocessor.joblib"]["sha256"] == pre_hash

    # 8. M8 consumes the published package; no refit
    def forbidden(*a, **k):
        pytest.fail("serving must not refit the preprocessor")
    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    service = PredictionService.load_published_package(backend=store, prefix=package.prefix, loader=MlflowPyfuncLoader())
    import pandas as pd
    test_file = next((tmp_path / "modeling" / "test" / "runs" / m6_report["run_id"]).glob("dataset.parquet"))
    rows = pd.read_parquet(test_file).head(3)
    from serving_support import to_records
    records = to_records(rows, list(service.contract.numeric) + list(service.contract.categorical))
    with TestClient(create_app(lambda: service)) as http:
        live, health, info = http.get("/live"), http.get("/health"), http.get("/model-info")
        pred = http.post("/predict", json={"records": records})
    say("GET /live", (live.status_code, live.json()))
    say("GET /health", (health.status_code, health.json()))
    info_json = info.json()
    say("GET /model-info", {"model": info_json["model"], "preprocessing": info_json["preprocessing"], "provenance": info_json["provenance"]})
    say("POST /predict", (pred.status_code, [round(p["value"], 3) for p in pred.json()["predictions"]], pred.json().get("target")))
    assert live.status_code == health.status_code == info.status_code == pred.status_code == 200

    # 6. traceability chain M6 -> M7 -> MLflow -> package -> M8
    chain = {
        "M6 run_id": m6_report["run_id"],
        "M7 report m6_run_id": training["m6_run_id"],
        "package m6_run_id": manifest["provenance"]["m6_run_id"],
        "M8 provenance m6_run_id": info_json["provenance"]["m6_run_id"],
        "M6 fingerprint": m6_report["source"]["input_fingerprint"],
        "M7 fingerprint": training["input_fingerprint"],
        "package fingerprint": manifest["provenance"]["input_fingerprint"],
        "M8 fingerprint": info_json["provenance"]["input_fingerprint"],
        "M7 run_id": training["run_id"],
        "MLflow version run_id": mv.run_id,
        "package m7_run_id": manifest["model"]["m7_run_id"],
        "M8 model run_id": info_json["model"]["run_id"],
        "package model uri": manifest["model"]["uri"],
        "M8 model uri": info_json["model"]["uri"],
    }
    for k, v in chain.items():
        say("chain", f"{k} = {v}")
    assert len({chain[k] for k in chain if "m6_run_id" in k or k == "M6 run_id"}) == 1
    assert len({chain[k] for k in chain if "fingerprint" in k}) == 1
    assert len({chain[k] for k in chain if "run_id" in k and "m6" not in k and "M6" not in k}) == 1
    assert chain["package model uri"] == chain["M8 model uri"] == f"models:/{NAME}/1"
    assert info_json["preprocessing"]["uri"] == package.package_uri  # served preprocessor comes from the package


def test_integral_dl_flow(config, tmp_path, monkeypatch, capsys):
    """M1-M8 with the M7 deep learning model (PyTorch MLP) and full traceability."""
    pytest.importorskip("torch", reason="install the optional dl extra to run the deep learning demo")
    import numpy as np
    import pandas as pd
    from serving_support import to_records

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["training"] = {
        "tracking_db_path": str(tmp_path / "mlflow" / "tracking.db"),
        "artifact_path": str(tmp_path / "mlflow" / "artifacts"),
        "output_path": str(tmp_path / "training-reports"),
        "experiment_name": "integral-dl-demo",
        # small, fast network for the synthetic demo (seed comes from modeling.random_seed)
        "mlp": {
            "hidden_dims": [32, 16],
            "dropout": 0.0,
            "learning_rate": 0.01,
            "batch_size": 8,
            "max_epochs": 40,
            "early_stopping_patience": 5,
        },
    }
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    client = SeasonClient()

    # 1-4. M1 plan, M2-M5 per season, M6 deferred until the history is complete
    exit_codes = {}
    for year in SEASONS:
        plan, results = run_phase(config, tmp_path, client, str(year), monkeypatch)
        assert all(codes == [0, 0, 0, 0] for codes in results.values())
        exit_codes[year] = m6(config, "--require-complete-history")
        say(f"DL phase {year}", f"sessions planned={len(plan['sessions'])}; m2..m5 exit codes={sorted(set(map(tuple, results.values())))}; m6 exit={exit_codes[year]}")
    assert exit_codes == {2023: 3, 2024: 3, 2025: 0}

    (report_path,) = modeling_reports(tmp_path)
    m6_report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = verify_report(str(report_path), storage=load_settings(str(config)).storage)
    say("DL M6", f"run_id={m6_report['run_id']} verdict={m6_report['verdict']} encoded features={len(m6_report['preprocessing']['output_feature_names'])} verify_report ok={integrity['ok']}")
    assert integrity["ok"] and m6_report["verdict"] in {"VALID", "VALID_WITH_WARNINGS"}

    # 5. M7 deep learning through the real CLI
    argv = ["--stage", "m7", "--model-type", "mlp", "--registered-model-name", DL_NAME,
            "--m6-report", str(report_path), "--config", str(config)]
    code = cli.main(argv)
    say("DL M7 command", "f1-pipeline " + " ".join({str(report_path): "<m6-report>", str(config): "<config>"}.get(a, a) for a in argv))
    say("DL M7 exit", code)
    assert code == 0
    (training_file,) = sorted((tmp_path / "training-reports").rglob("training_report_*.json"))
    training = json.loads(training_file.read_text(encoding="utf-8"))
    assert training["model_type"] == "mlp" and training["registered_model"]["name"] == DL_NAME
    mlp_training = training["mlp_training"]
    say("DL M7 report", {k: training.get(k) for k in ("run_id", "m6_run_id", "model_type", "target")} | {"registered_model": {k: training["registered_model"][k] for k in ("name", "version", "uri", "recovery_check")}})
    say("DL M7 MLP", {"hidden_dims": training["config"]["mlp"]["hidden_dims"], "loss": training["config"]["mlp"]["loss"]} | mlp_training)
    comparison = {split: {family: round(values["mae"], 3) for family, values in training["metrics"][split].items()} for split in ("train", "validation", "test")}
    say("DL M7 MAE (s) mlp vs baselines", comparison)
    assert all(set(row) == {"mlp", "linear_regression", "reference_mean"} for row in comparison.values())
    # interpretability (permutation feature importance on validation only) and train-vs-validation diagnosis
    interpretability = training["interpretability"]
    ranked = interpretability.get("top_features", interpretability.get("importances")) or []
    top3 = [(entry["feature"], round(entry["mae_increase_mean"], 4)) for entry in ranked[:3]]
    diagnosis = training["generalization_diagnosis"]
    say("DL M7 PFI top-3 (validation, MAE increase s)", top3)
    say("DL M7 generalization", {"status": diagnosis["status"], "evaluated_splits": diagnosis["evaluated_splits"], "val_to_train_mae_ratio": diagnosis["val_to_train_mae_ratio"]})
    assert interpretability["split"] == "validation" and interpretability["collinearity_note"]
    assert 0 < len(top3) <= 3 and all(name in m6_report["preprocessing"]["output_feature_names"] for name, _ in top3)
    assert diagnosis["evaluated_splits"] == ["train", "validation"]
    assert diagnosis["status"] in {"underfitting", "overfitting_risk", "balanced", "insufficient_metrics"}
    # anti-leakage: target scaling comes from the finite train targets only
    train_targets = pd.read_parquet(next(item["path"] for item in m6_report["artifact_files"] if item["split"] == "train" and item["artifact"] == "features.parquet"))["next_lap_time"].dropna()
    assert mlp_training["y_mean"] == pytest.approx(float(train_targets.mean()))

    # 6. MLflow: pyfunc registration, per-epoch curves, artifacts
    uri = f"sqlite:///{(tmp_path / 'mlflow' / 'tracking.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mc = MlflowClient()
    (mv,) = mc.search_model_versions(f"name='{DL_NAME}'")
    run = mc.get_run(mv.run_id)
    train_curve = sorted(mc.get_metric_history(mv.run_id, "train.mlp.loss"), key=lambda m: m.step)
    val_curve = sorted(mc.get_metric_history(mv.run_id, "validation.mlp.loss"), key=lambda m: m.step)
    say("DL MLflow registry", [(mv.name, mv.version, mv.status, mv.run_id)])
    say("DL MLflow run", f"name={run.info.run_name} status={run.info.status} tags.model_type={run.data.tags['model_type']} params.estimator={run.data.params['estimator']}")
    say("DL MLflow curves", f"epochs={len(train_curve)} best_epoch={mlp_training['best_epoch']} train.mlp.loss {train_curve[0].value:.4f}->{train_curve[-1].value:.4f} validation.mlp.loss {val_curve[0].value:.4f}->{min(m.value for m in val_curve):.4f} (best)")
    artifacts = sorted(a.path for a in mc.list_artifacts(mv.run_id))
    say("DL MLflow artifacts", artifacts)
    pyfunc = mlflow.pyfunc.load_model(f"models:/{DL_NAME}/{mv.version}")
    say("DL MLflow model", f"flavors={sorted(pyfunc.metadata.flavors)} run_id={pyfunc.metadata.run_id} signature inputs={len(pyfunc.metadata.get_input_schema().input_names())}")
    assert mv.run_id == training["run_id"] == pyfunc.metadata.run_id and run.info.status == "FINISHED"
    assert run.info.run_name.startswith("m7-mlp-") and run.data.tags["model_type"] == "mlp"
    assert len(train_curve) == len(val_curve) == mlp_training["epochs_trained"]
    assert "training_history.json" in artifacts
    assert "sklearn" not in pyfunc.metadata.flavors
    assert list(pyfunc.metadata.get_input_schema().input_names()) == m6_report["preprocessing"]["output_feature_names"]

    # 7. Immutable publication: local store and simulated S3 hold the same package
    model_uri = f"models:/{DL_NAME}/{mv.version}"
    store = LocalStorageBackend(tmp_path / "model-store")
    s3 = S3StorageBackend("dl-demo-models", client=FakeS3Client())
    package = publish_model(model_uri, store)
    s3_package = publish_model(model_uri, s3)
    again = publish_model(model_uri, store)
    manifest = package.manifest
    checkpoint = [name for name in manifest["files"] if name.endswith("mlp_checkpoint.pt")]
    say("DL package", f"prefix={package.prefix} files={len(manifest['files'])} checkpoint={checkpoint}")
    say("DL package local == simulated S3", package.manifest_sha256 == s3_package.manifest_sha256 and package.artifacts == s3_package.artifacts)
    say("DL PUBLISHED marker", store.read_bytes(f"{package.prefix}/{PUBLISHED_FILE}").decode().strip())
    assert checkpoint and package.manifest == s3_package.manifest and package.artifacts == s3_package.artifacts
    assert again.manifest_sha256 == package.manifest_sha256  # idempotent republication
    assert read_model_package(s3, s3_package.prefix).manifest_sha256 == package.manifest_sha256
    assert not any(path.is_symlink() for path in (tmp_path / "model-store").rglob("*"))

    # 8. M8 serves the published package; the M6 preprocessor is never refitted
    def forbidden(*a, **k):
        pytest.fail("serving must not refit the preprocessor")
    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", forbidden)
    service = PredictionService.load_published_package(backend=store, prefix=package.prefix, loader=MlflowPyfuncLoader())
    rows = pd.read_parquet(next((tmp_path / "modeling" / "test" / "runs" / m6_report["run_id"]).glob("dataset.parquet"))).head(3)
    records = to_records(rows, list(service.contract.numeric) + list(service.contract.categorical))
    with TestClient(create_app(lambda: service)) as http:
        live, health, info = http.get("/live"), http.get("/health"), http.get("/model-info")
        pred = http.post("/predict", json={"records": records})
    info_json = info.json()
    values = [p["value"] for p in pred.json()["predictions"]]
    actual = rows["next_lap_time"].astype(float).tolist()
    say("DL GET /live", (live.status_code, live.json()))
    say("DL GET /health", (health.status_code, health.json()))
    say("DL GET /model-info", {"model": info_json["model"], "provenance": info_json["provenance"]})
    say("DL POST /predict", (pred.status_code, [round(v, 3) for v in values], pred.json().get("target")))
    say("DL actual next_lap_time (s)", [round(v, 3) for v in actual])
    assert live.status_code == health.status_code == info.status_code == pred.status_code == 200
    # seconds in the original target scale: standardised outputs (~0) would sit ~y_mean/y_std (~38) std away
    assert all(v > 50.0 and abs(v - mlp_training["y_mean"]) < 10 * mlp_training["y_std"] for v in values)

    # M8 from the matched M7 run (runs:/<m7_run_id>/m6) gives the same answers
    from_run = PredictionService.load(preprocessing_uri=f"runs:/{mv.run_id}/m6", model_uri=model_uri, loader=MlflowPyfuncLoader())
    np.testing.assert_allclose(from_run.predict(records), values, rtol=1e-6, atol=1e-6)
    say("DL M8 runs:/<m7_run_id>/m6 parity", True)

    # traceability chain M6 -> M7 (MLP) -> MLflow -> package -> M8
    chain = {
        "M6 run_id": m6_report["run_id"],
        "M7 report m6_run_id": training["m6_run_id"],
        "package m6_run_id": manifest["provenance"]["m6_run_id"],
        "M8 provenance m6_run_id": info_json["provenance"]["m6_run_id"],
        "M6 fingerprint": m6_report["source"]["input_fingerprint"],
        "M7 fingerprint": training["input_fingerprint"],
        "package fingerprint": manifest["provenance"]["input_fingerprint"],
        "M8 fingerprint": info_json["provenance"]["input_fingerprint"],
        "M7 run_id": training["run_id"],
        "MLflow version run_id": mv.run_id,
        "package m7_run_id": manifest["model"]["m7_run_id"],
        "M8 model run_id": info_json["model"]["run_id"],
        "package model uri": manifest["model"]["uri"],
        "M8 model uri": info_json["model"]["uri"],
    }
    for k, v in chain.items():
        say("DL chain", f"{k} = {v}")
    assert len({chain[k] for k in chain if "m6_run_id" in k or k == "M6 run_id"}) == 1
    assert len({chain[k] for k in chain if "fingerprint" in k}) == 1
    assert len({chain[k] for k in chain if "run_id" in k and "m6" not in k and "M6" not in k}) == 1
    assert chain["package model uri"] == chain["M8 model uri"] == model_uri
    assert info_json["preprocessing"]["uri"] == package.package_uri
