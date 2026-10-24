"""Environment and storage configuration: local by default, AWS by overlay, never credentials."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from f1_race_intelligence.config import settings as settings_module
from f1_race_intelligence.config.settings import DEFAULT_CONFIG_PATH, AppSettings, ConfigFileNotFoundError, load_settings
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
from f1_race_intelligence.storage.store import DataStore

STORAGE_VARIABLES = (
    "F1_ENVIRONMENT",
    "F1_STORAGE_BACKEND",
    "F1_STORAGE_LAYOUT",
    "F1_STORAGE_LOCAL_ROOT",
    "F1_STORAGE_BUCKET",
    "F1_STORAGE_PREFIX",
    "F1_STORAGE_REGION",
    "F1_STORAGE_ENDPOINT_URL",
    "F1_LOG_FILE",
    "F1_CONFIG_PATH",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch) -> None:
    for name in STORAGE_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def empty_config(tmp_path: Path) -> Path:
    """An existing, empty config file: every value from the model defaults."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    return path


def test_the_repository_config_runs_locally_on_the_historical_tree() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.environment == "local"
    assert settings.storage.backend == "local"
    assert settings.storage.layout == "auto"
    assert settings.storage.bucket is None
    # Nothing about M1-M6 moved.
    assert settings.historical_extraction.output_path == "data/raw"
    assert settings.consolidation.output_path == "data/processed/lap_dataset"
    assert settings.modeling.split_strategy == "train_years_then_split"
    assert settings.modeling.train_years == [2023, 2024]


def test_the_aws_overlay_switches_to_s3_and_keeps_everything_else(monkeypatch) -> None:
    monkeypatch.setenv("F1_ENVIRONMENT", "aws")
    monkeypatch.setenv("F1_STORAGE_BUCKET", "f1-race-intelligence-dev")
    local = load_settings(DEFAULT_CONFIG_PATH, overrides={"environment": "local"})

    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.environment == "aws"
    assert settings.storage.backend == "s3"
    assert settings.storage.layout == "lake"
    assert settings.storage.bucket == "f1-race-intelligence-dev"
    assert settings.modeling == local.modeling
    assert settings.features == local.features
    assert settings.openf1 == local.openf1


def test_aws_without_a_bucket_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("F1_ENVIRONMENT", "aws")

    with pytest.raises(ValidationError, match="bucket"):
        load_settings(DEFAULT_CONFIG_PATH)


def test_aws_refuses_local_storage() -> None:
    with pytest.raises(ValidationError, match="requires storage.backend 's3'"):
        AppSettings(environment="aws", storage={"backend": "local"})


def test_storage_environment_variables_override_yaml(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("storage:\n  backend: local\n  prefix: from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("F1_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("F1_STORAGE_BUCKET", "bucket-from-env")
    monkeypatch.setenv("F1_STORAGE_PREFIX", "dev")
    monkeypatch.setenv("F1_STORAGE_REGION", "us-east-1")

    storage = load_settings(config).storage

    assert (storage.backend, storage.bucket, storage.prefix, storage.region) == ("s3", "bucket-from-env", "dev", "us-east-1")


def test_explicit_overrides_win_over_environment_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("F1_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("F1_STORAGE_BUCKET", "bucket-from-env")

    settings = load_settings(empty_config(tmp_path), overrides={"storage": {"backend": "local", "layout": "lake"}})

    assert settings.storage.backend == "local"
    assert settings.storage.layout == "lake"
    assert settings.storage.bucket == "bucket-from-env"


def test_an_overlay_for_an_environment_is_merged_section_by_section(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "environment: aws\nlogging:\n  level: DEBUG\n  json_format: false\n", encoding="utf-8"
    )
    (tmp_path / "config.aws.yaml").write_text(
        "storage:\n  backend: s3\n  bucket: overlay-bucket\nlogging:\n  json_format: true\n", encoding="utf-8"
    )

    settings = load_settings(tmp_path / "config.yaml")

    assert settings.storage.bucket == "overlay-bucket"
    assert settings.logging.json_format is True
    assert settings.logging.level == "DEBUG"


@pytest.mark.parametrize("bucket", ["s3://bucket", "bucket/prefix"])
def test_bucket_must_be_a_bare_name(bucket: str) -> None:
    with pytest.raises(ValidationError):
        AppSettings(storage={"backend": "s3", "bucket": bucket})


def test_no_credential_fields_exist_in_configuration() -> None:
    fields = set(AppSettings.model_fields) | set(AppSettings.model_fields["storage"].annotation.model_fields)

    assert not {name for name in fields if "key" in name or "secret" in name or "token" in name or "password" in name}


def test_the_config_files_contain_no_credentials() -> None:
    for path in DEFAULT_CONFIG_PATH.parent.glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        assert "aws_access_key_id" not in text and "aws_secret_access_key" not in text, path


# -- a missing configuration file is an error, never "all defaults" ----------


@pytest.fixture
def installed_package(tmp_path: Path, monkeypatch) -> Path:
    """What a non-editable install sees: no configs/ next to the package."""
    missing = tmp_path / "site-packages" / "configs" / "config.yaml"
    monkeypatch.setattr(settings_module, "DEFAULT_CONFIG_PATH", missing)
    return missing


def test_a_missing_f1_config_path_is_an_error_naming_the_path(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "nowhere" / "config.yaml"
    monkeypatch.setenv("F1_CONFIG_PATH", str(missing))

    with pytest.raises(ConfigFileNotFoundError) as error:
        load_settings()

    assert error.value.path == missing
    assert str(missing) in str(error.value) and "F1_CONFIG_PATH" in str(error.value)


def test_a_missing_explicit_path_is_an_error_even_when_f1_config_path_is_valid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("F1_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

    with pytest.raises(ConfigFileNotFoundError, match="--config"):
        load_settings(tmp_path / "typo.yaml")


def test_an_installed_package_without_f1_config_path_fails_instead_of_running_empty(installed_package: Path) -> None:
    # Before: years=[] and endpoints=[], and M2 "succeeded" extracting nothing.
    with pytest.raises(ConfigFileNotFoundError) as error:
        load_settings()

    message = str(error.value)
    assert str(installed_package) in message
    assert "F1_CONFIG_PATH is not set" in message and "/app/configs/config.yaml" in message


def test_an_installed_package_runs_with_f1_config_path(installed_package: Path, monkeypatch) -> None:
    monkeypatch.setenv("F1_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

    extraction = load_settings().historical_extraction

    assert extraction.years == [2023, 2024, 2025]
    assert len(extraction.endpoints) == 9


def test_an_empty_f1_config_path_counts_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("F1_CONFIG_PATH", "  ")

    assert load_settings().historical_extraction.years == [2023, 2024, 2025]


def test_a_missing_file_is_an_os_error_for_the_cli_contract(tmp_path: Path) -> None:
    # The CLI maps OSError/ValueError to exit code 2 (invalid configuration).
    with pytest.raises(OSError):
        load_settings(tmp_path / "missing.yaml")


def test_the_container_configuration_selects_s3_without_touching_aws(installed_package: Path, monkeypatch) -> None:
    # Exactly the environment the Dockerfile and a future task definition set.
    monkeypatch.setenv("F1_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    monkeypatch.setenv("F1_ENVIRONMENT", "aws")
    monkeypatch.setenv("F1_STORAGE_BUCKET", "f1-race-intelligence-dev-bucket")

    settings = load_settings()
    store = DataStore.from_settings(settings)

    assert (settings.storage.backend, settings.storage.layout, settings.storage.prefix) == ("s3", "lake", "")
    assert isinstance(store.backend, S3StorageBackend)
    assert store.backend.describe()["location"] == "s3://f1-race-intelligence-dev-bucket"
    assert store.backend._client is None  # boto3 is only touched on first use


def test_a_missing_overlay_is_optional_but_aws_still_needs_s3(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("environment: aws\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="requires storage.backend 's3'"):
        load_settings(config)

    monkeypatch.setenv("F1_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("F1_STORAGE_BUCKET", "bucket-from-env")
    assert load_settings(config).storage.bucket == "bucket-from-env"


def test_log_file_is_optional_and_configurable(tmp_path: Path, monkeypatch) -> None:
    assert load_settings(empty_config(tmp_path)).logging.file_path is None

    monkeypatch.setenv("F1_LOG_FILE", "logs/dev.log")

    assert load_settings(empty_config(tmp_path)).logging.file_path == "logs/dev.log"
