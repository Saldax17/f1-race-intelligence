from pathlib import Path

import pytest
from pydantic import ValidationError

from f1_race_intelligence.config.settings import (
    DEFAULT_LINEAR_REGISTERED_MODEL_NAME,
    DEFAULT_MLP_REGISTERED_MODEL_NAME,
    MLPSettings,
    TrainingSettings,
    load_settings,
)


def test_load_settings_reads_yaml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
openf1:
  base_url: "https://example.test/v1"
  timeout:
    connect: 2
    read: 9
  rate_limit:
    requests_per_second: 5
    requests_per_minute: 50
  retry:
    max_attempts: 4
    backoff_factor: 2
logging:
  level: "DEBUG"
  json_format: false
""",
        encoding="utf-8",
    )

    settings = load_settings(config_file)

    assert settings.openf1.base_url == "https://example.test/v1"
    assert settings.openf1.timeout.connect == 2
    assert settings.openf1.timeout.read == 9
    assert settings.openf1.rate_limit.requests_per_second == 5
    assert settings.openf1.rate_limit.requests_per_minute == 50
    assert settings.openf1.retry.max_attempts == 4
    assert settings.openf1.retry.backoff_factor == 2
    assert settings.logging.level == "DEBUG"
    assert settings.logging.json_format is False


def test_load_settings_uses_defaults_for_an_empty_file(tmp_path: Path) -> None:
    # A missing file is an error (test_settings_environments.py); an empty one
    # is the explicit way to ask for every default.
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")

    settings = load_settings(empty_file)

    assert settings.openf1.base_url == "https://api.openf1.org/v1"
    assert settings.openf1.rate_limit.requests_per_second == 3
    assert settings.openf1.rate_limit.requests_per_minute == 30


def test_load_settings_reads_historical_extraction_section(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
historical_extraction:
  years: [2023, 2024]
  session_types: ["Race", "Sprint"]
  endpoints: ["laps", "weather"]
  output_path: "custom/raw"
  manifest_path: "custom/manifests"
  overwrite: true
""",
        encoding="utf-8",
    )

    extraction = load_settings(config_file).historical_extraction

    assert extraction.years == [2023, 2024]
    assert extraction.session_types == ["Race", "Sprint"]
    assert extraction.endpoints == ["laps", "weather"]
    assert extraction.output_path == "custom/raw"
    assert extraction.manifest_path == "custom/manifests"
    assert extraction.overwrite is True


def test_historical_extraction_defaults_to_races_when_section_is_missing(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("openf1:\n  base_url: https://api.openf1.org/v1\n", encoding="utf-8")

    extraction = load_settings(config_file).historical_extraction

    assert extraction.years == []
    assert extraction.session_types == ["Race"]
    assert extraction.output_path == "data/raw"
    assert extraction.overwrite is False


def test_historical_extraction_env_overrides(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("historical_extraction:\n  years: [2023]\n", encoding="utf-8")

    monkeypatch.setenv("F1_EXTRACTION_YEARS", "2024,2025")
    monkeypatch.setenv("F1_EXTRACTION_OVERWRITE", "true")

    extraction = load_settings(config_file).historical_extraction

    assert extraction.years == [2024, 2025]
    assert extraction.overwrite is True


def test_load_settings_applies_env_overrides(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("openf1:\n  base_url: 'https://example.test/v1'\n", encoding="utf-8")

    monkeypatch.setenv("OPENF1_RATE_LIMIT_RPS", "7")
    monkeypatch.setenv("OPENF1_RETRY_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    settings = load_settings(config_file)

    assert settings.openf1.base_url == "https://example.test/v1"
    assert settings.openf1.rate_limit.requests_per_second == 7
    assert settings.openf1.retry.max_attempts == 9
    assert settings.logging.level == "WARNING"


# -- M7 deep learning (MLP) settings -------------------------------------------

REPOSITORY_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_mlp_settings_defaults_match_the_design() -> None:
    mlp = MLPSettings()

    assert mlp.registered_model_name is None
    assert mlp.hidden_dims == [64, 32]
    assert mlp.dropout == 0.1
    assert mlp.learning_rate == 1e-3
    assert mlp.weight_decay == 1e-4
    assert mlp.batch_size == 64
    assert mlp.max_epochs == 100
    assert mlp.early_stopping_patience == 10
    assert mlp.min_delta == 1e-4
    assert mlp.loss == "huber"
    assert mlp.scale_target is True


def test_training_settings_default_to_the_unchanged_linear_baseline() -> None:
    training = TrainingSettings()

    assert training.model_type == "linear_regression"
    assert training.registered_model_name == DEFAULT_LINEAR_REGISTERED_MODEL_NAME == "f1-next-lap-linear-regression"
    assert DEFAULT_MLP_REGISTERED_MODEL_NAME == "f1-next-lap-mlp"
    assert isinstance(training.mlp, MLPSettings)
    assert training.effective_registered_model_name() == DEFAULT_LINEAR_REGISTERED_MODEL_NAME


def test_repository_config_keeps_linear_default_and_declares_mlp_section(monkeypatch) -> None:
    for name in ("F1_ENVIRONMENT", "F1_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)

    training = load_settings(REPOSITORY_CONFIG).training

    assert training.model_type == "linear_regression"
    assert training.registered_model_name == DEFAULT_LINEAR_REGISTERED_MODEL_NAME
    assert training.mlp == MLPSettings()
    assert training.effective_registered_model_name("mlp") == DEFAULT_MLP_REGISTERED_MODEL_NAME


def test_load_settings_reads_the_training_mlp_section(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
training:
  model_type: mlp
  mlp:
    registered_model_name: custom-mlp
    hidden_dims: [32, 16, 8]
    dropout: 0.0
    learning_rate: 0.005
    weight_decay: 0.0
    batch_size: 128
    max_epochs: 7
    early_stopping_patience: 2
    min_delta: 0.0
    loss: mse
    scale_target: false
""",
        encoding="utf-8",
    )

    training = load_settings(config_file).training

    assert training.model_type == "mlp"
    assert training.mlp.hidden_dims == [32, 16, 8]
    assert training.mlp.dropout == 0.0
    assert training.mlp.learning_rate == 0.005
    assert training.mlp.weight_decay == 0.0
    assert training.mlp.batch_size == 128
    assert training.mlp.max_epochs == 7
    assert training.mlp.early_stopping_patience == 2
    assert training.mlp.min_delta == 0.0
    assert training.mlp.loss == "mse"
    assert training.mlp.scale_target is False
    assert training.effective_registered_model_name() == "custom-mlp"


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_dims": []},
        {"hidden_dims": [64, 0]},
        {"hidden_dims": [-8]},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-3},
        {"learning_rate": float("inf")},
        {"weight_decay": -1e-4},
        {"weight_decay": float("nan")},
        {"batch_size": 0},
        {"max_epochs": 0},
        {"early_stopping_patience": 0},
        {"min_delta": -1e-4},
        {"min_delta": float("inf")},
        {"loss": "mae"},
        {"registered_model_name": ""},
        {"registered_model_name": "   "},
    ],
    ids=lambda overrides: "-".join(f"{key}={value!r}" for key, value in overrides.items()),
)
def test_mlp_settings_reject_invalid_hyperparameters(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        MLPSettings(**overrides)


def test_invalid_mlp_settings_in_yaml_fail_settings_loading(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("training:\n  mlp:\n    dropout: 1.5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="dropout"):
        load_settings(config_file)


@pytest.mark.parametrize("value", ["", "  "])
def test_training_settings_reject_a_blank_registered_model_name(value: str) -> None:
    with pytest.raises(ValidationError, match="registered_model_name"):
        TrainingSettings(registered_model_name=value)


def test_training_settings_reject_an_unknown_model_type() -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(model_type="random_forest")


@pytest.mark.parametrize(
    ("training", "model_type", "expected"),
    [
        # linear_regression always uses training.registered_model_name
        ({}, "linear_regression", DEFAULT_LINEAR_REGISTERED_MODEL_NAME),
        ({"registered_model_name": "lin"}, "linear_regression", "lin"),
        ({"mlp": {"registered_model_name": "only-mlp"}}, "linear_regression", DEFAULT_LINEAR_REGISTERED_MODEL_NAME),
        # mlp: 1) its own name, 2) an overridden top-level name, 3) the MLP default
        ({"mlp": {"registered_model_name": "own"}, "registered_model_name": "top"}, "mlp", "own"),
        ({"registered_model_name": "top"}, "mlp", "top"),
        ({}, "mlp", DEFAULT_MLP_REGISTERED_MODEL_NAME),
        ({"registered_model_name": DEFAULT_LINEAR_REGISTERED_MODEL_NAME}, "mlp", DEFAULT_MLP_REGISTERED_MODEL_NAME),
        # with no argument the configured model_type decides
        ({"model_type": "mlp"}, None, DEFAULT_MLP_REGISTERED_MODEL_NAME),
        ({"model_type": "mlp", "registered_model_name": "top"}, None, "top"),
        ({"model_type": "linear_regression", "registered_model_name": "lin"}, None, "lin"),
    ],
)
def test_effective_registered_model_name_resolution(training: dict, model_type, expected: str) -> None:
    assert TrainingSettings(**training).effective_registered_model_name(model_type) == expected


def test_effective_registered_model_name_rejects_unknown_model_types() -> None:
    with pytest.raises(ValueError, match="Unsupported training model_type"):
        TrainingSettings().effective_registered_model_name("xgboost")
