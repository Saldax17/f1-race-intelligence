"""Serving prototype (M8): contract, artifact checks and parity with M6.

These tests need no FastAPI and no MLflow. The M6 artifacts are produced by
the real M6 runner from synthetic races; the models are recording stubs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence.modeling.runner import PREPROCESSOR_FILE
from f1_race_intelligence.serving.contract import FEATURE_SCHEMA_FILE, ArtifactError, ContractError, FeatureContract
from f1_race_intelligence.serving.model import MlflowPyfuncLoader, ModelLoadError
from f1_race_intelligence.serving.service import PredictionError, PredictionService
from serving_support import M6Run, RecordingModel, StubLoader, build_m6_run, to_records

MODEL_URI = "models:/example/1"  # opaque to the service; the stub loader ignores it


@pytest.fixture(scope="module")
def m6_run(tmp_path_factory) -> M6Run:
    return build_m6_run(tmp_path_factory.mktemp("m6"))


@pytest.fixture
def schema(m6_run) -> dict:
    return json.loads((m6_run.preprocessing_dir / FEATURE_SCHEMA_FILE).read_text(encoding="utf-8"))


def load(m6_run, model=None, input_names="encoded", **kwargs) -> PredictionService:
    schema = json.loads((m6_run.preprocessing_dir / FEATURE_SCHEMA_FILE).read_text(encoding="utf-8"))
    names = schema["encoded_feature_names"] if input_names == "encoded" else input_names
    return PredictionService.load(
        preprocessing_uri=str(m6_run.preprocessing_dir),
        model_uri=MODEL_URI,
        loader=StubLoader(model or RecordingModel(), input_names=names),
        **kwargs,
    )


def valid_records(m6_run, count=2):
    test_rows = m6_run.split_file("test", "dataset.parquet").head(count)
    return to_records(test_rows, m6_run.numeric + m6_run.categorical)


# -- the contract comes from M6's artifacts --------------------------------


def test_the_service_reads_the_files_m6_writes(m6_run) -> None:
    # Pins the names against a real M6 run, not against our own constants.
    assert {path.name for path in m6_run.preprocessing_dir.iterdir()} == {PREPROCESSOR_FILE, FEATURE_SCHEMA_FILE}


def test_the_contract_is_whatever_the_m6_schema_says(m6_run, schema) -> None:
    contract = load(m6_run).contract

    assert contract.numeric == m6_run.numeric
    assert contract.categorical == m6_run.categorical
    assert list(contract.encoded_feature_names) == schema["encoded_feature_names"]
    assert contract.target == schema["target"] == "next_lap_time"
    assert set(contract.traceability) == set(schema["traceability"])


def test_a_schema_that_is_not_from_m6_is_rejected() -> None:
    with pytest.raises(ArtifactError):
        FeatureContract.from_schema({"encoded_feature_names": ["a"], "target": "y"})
    with pytest.raises(ArtifactError, match="both numeric and categorical"):
        FeatureContract.from_schema(
            {"input_features": {"numeric": ["a"], "categorical": ["a"]}, "encoded_feature_names": ["a"], "target": "y"}
        )


def test_a_directory_without_m6_artifacts_is_rejected(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="not an M6 preprocessing run"):
        PredictionService.load(preprocessing_uri=str(tmp_path), model_uri=MODEL_URI, loader=StubLoader(RecordingModel()))


# -- M6's transformation, not a copy of it ----------------------------------


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_the_model_receives_exactly_the_matrix_m6_wrote(m6_run, schema, split) -> None:
    """Untransformed rows in, M6's own encoded features.parquet out, bit for bit."""
    model = RecordingModel()
    service = load(m6_run, model)
    dataset = m6_run.split_file(split, "dataset.parquet")

    service.predict(to_records(dataset, m6_run.numeric + m6_run.categorical))

    expected = m6_run.split_file(split, "features.parquet")[schema["encoded_feature_names"]]
    received = model.inputs[-1]
    assert list(received.columns) == schema["encoded_feature_names"]
    pd.testing.assert_frame_equal(received.reset_index(drop=True), expected.reset_index(drop=True))


def test_predictions_come_back_in_request_order(m6_run) -> None:
    assert load(m6_run).predict(valid_records(m6_run, 3)) == [80.0, 81.0, 82.0]


def test_traceability_fields_are_accepted_but_never_reach_the_model(m6_run) -> None:
    model = RecordingModel()
    service = load(m6_run, model)
    record = {**valid_records(m6_run, 1)[0], "session_key": 9999, "driver_number": 44, "year": 2025}

    service.predict([record])

    assert {"session_key", "driver_number", "year"}.isdisjoint(model.inputs[-1].columns)
    assert service.contract.keys_of(record)["session_key"] == 9999


def test_null_features_are_imputed_by_m6_and_unseen_categories_do_not_fail(m6_run) -> None:
    model = RecordingModel()
    service = load(m6_run, model)
    record = {name: None for name in m6_run.numeric + m6_run.categorical}
    record["compound"] = "HYPERSOFT"  # never seen in training: encoded as zeros by M6

    service.predict([record])

    received = model.inputs[-1]
    assert not received.isna().any().any()
    assert (received.filter(like="compound_") == 0).all().all()


# -- request validation -----------------------------------------------------


def errors_for(service, records):
    with pytest.raises(ContractError) as caught:
        service.predict(records)
    return {(error["record"], error["field"]): error["error"] for error in caught.value.errors}


def test_a_missing_feature_is_an_error_not_a_silent_null(m6_run) -> None:
    service = load(m6_run)
    record = valid_records(m6_run, 1)[0]
    del record["tyre_age"]

    assert "missing feature" in errors_for(service, [record])[(0, "tyre_age")]


def test_unknown_and_leaky_fields_are_rejected(m6_run) -> None:
    service = load(m6_run)
    record = {**valid_records(m6_run, 1)[0], "tyre_agee": 3, "next_lap_time": 91.0, "pit_duration": 22.0}

    errors = errors_for(service, [record])

    assert "unknown field" in errors[(0, "tyre_agee")]
    assert "must never reach the model" in errors[(0, "next_lap_time")]
    assert "must never reach the model" in errors[(0, "pit_duration")]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("lap_duration", "91.2", "numeric feature"),
        ("lap_duration", float("nan"), "numeric feature"),
        ("lap_duration", float("inf"), "numeric feature"),
        ("compound", 3, "categorical feature"),
        ("compound", True, "categorical feature"),
    ],
)
def test_values_of_the_wrong_kind_are_rejected(m6_run, field, value, message) -> None:
    service = load(m6_run)
    record = {**valid_records(m6_run, 1)[0], field: value}

    assert message in errors_for(service, [record])[(0, field)]


def test_every_problem_is_reported_in_one_answer(m6_run) -> None:
    service = load(m6_run)
    good, bad = valid_records(m6_run, 2)
    bad = {**bad, "lap_duration": "x", "compound": 1}

    errors = errors_for(service, [good, bad, "not an object"])

    assert set(errors) == {(1, "lap_duration"), (1, "compound"), (2, None)}


def test_empty_and_oversized_batches_are_rejected(m6_run) -> None:
    service = load(m6_run, max_records=2)

    assert "at least one record" in errors_for(service, [])[(None, "records")]
    assert "at most 2 records" in errors_for(service, valid_records(m6_run, 3))[(None, "records")]


# -- the three artifacts must belong together -------------------------------


def copy_run(m6_run, tmp_path, edit) -> Path:
    target = tmp_path / "run"
    target.mkdir()
    (target / PREPROCESSOR_FILE).write_bytes((m6_run.preprocessing_dir / PREPROCESSOR_FILE).read_bytes())
    schema = json.loads((m6_run.preprocessing_dir / FEATURE_SCHEMA_FILE).read_text(encoding="utf-8"))
    edit(schema)
    (target / FEATURE_SCHEMA_FILE).write_text(json.dumps(schema), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "edit",
    [
        lambda schema: schema["encoded_feature_names"].reverse(),
        lambda schema: schema["input_features"]["numeric"].pop(),
    ],
    ids=["encoded-columns-differ", "input-columns-differ"],
)
def test_a_schema_from_another_m6_run_is_refused(m6_run, tmp_path, edit) -> None:
    directory = copy_run(m6_run, tmp_path, edit)

    with pytest.raises(ArtifactError, match="different M6 runs"):
        PredictionService.load(preprocessing_uri=str(directory), model_uri=MODEL_URI, loader=StubLoader(RecordingModel()))


def test_a_model_trained_on_other_columns_is_refused(m6_run, schema) -> None:
    names = list(schema["encoded_feature_names"])[:-1] + ["column_from_another_run"]

    with pytest.raises(ArtifactError, match="trained on another M6 run"):
        load(m6_run, input_names=names)


def test_a_model_without_a_signature_loads_but_says_so(m6_run) -> None:
    service = load(m6_run, input_names=None)

    assert service.describe()["model"]["input_signature_verified"] is False


def test_signature_order_does_not_matter(m6_run, schema) -> None:
    service = load(m6_run, input_names=list(reversed(schema["encoded_feature_names"])))

    assert service.describe()["model"]["input_signature_verified"] is True


# -- model output -------------------------------------------------------------


@pytest.mark.parametrize(
    "output, message",
    [
        (np.array([1.0]), "1 value"),
        (np.array([1.0, np.nan]), "non-finite"),
        (np.array(["a", "b"]), "not numeric"),
    ],
)
def test_unusable_model_output_is_an_error(m6_run, output, message) -> None:
    service = load(m6_run, RecordingModel(output=output))

    with pytest.raises(PredictionError, match=message):
        service.predict(valid_records(m6_run, 2))


def test_a_column_vector_or_frame_output_is_accepted(m6_run) -> None:
    for output in (np.array([[1.5], [2.5]]), pd.DataFrame({"prediction": [1.5, 2.5]})):
        assert load(m6_run, RecordingModel(output=output)).predict(valid_records(m6_run, 2)) == [1.5, 2.5]


# -- the MLflow loader ----------------------------------------------------------


class FakeSchema:
    def __init__(self, names):
        self._names = names

    def has_input_names(self):
        return self._names is not None

    def input_names(self):
        return list(self._names or [])


def fake_mlflow(monkeypatch, *, names=("a", "b"), error=None):
    calls = []
    metadata = types.SimpleNamespace(
        run_id="abc123",
        flavors={"python_function": {}, "sklearn": {}},
        get_input_schema=lambda: FakeSchema(names),
    )

    def load_model(uri):
        calls.append(uri)
        if error:
            raise error
        return types.SimpleNamespace(metadata=metadata, predict=lambda data: data)

    module = types.ModuleType("mlflow")
    module.pyfunc = types.SimpleNamespace(load_model=load_model)
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return calls


def test_the_mlflow_loader_uses_pyfunc_and_reads_the_signature(monkeypatch) -> None:
    calls = fake_mlflow(monkeypatch)

    loaded = MlflowPyfuncLoader().load("models:/example@champion")

    assert calls == ["models:/example@champion"]
    assert (loaded.run_id, loaded.flavors, loaded.input_names) == ("abc123", ("python_function", "sklearn"), ("a", "b"))


def test_a_signature_without_column_names_is_not_trusted(monkeypatch) -> None:
    fake_mlflow(monkeypatch, names=None)

    assert MlflowPyfuncLoader().load("runs:/abc123/model").input_names is None


def test_mlflow_failures_become_model_load_errors(monkeypatch) -> None:
    fake_mlflow(monkeypatch, error=OSError("registry unreachable"))

    with pytest.raises(ModelLoadError, match="registry unreachable"):
        MlflowPyfuncLoader().load("models:/example/1")


def test_without_mlflow_the_error_says_how_to_get_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes the import fail

    with pytest.raises(ModelLoadError, match="pip install mlflow"):
        MlflowPyfuncLoader().load("models:/example/1")
