from pathlib import Path

from f1_race_intelligence.config.settings import load_settings


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


def test_load_settings_falls_back_to_defaults_when_file_missing(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.yaml"

    settings = load_settings(missing_file)

    assert settings.openf1.base_url == "https://api.openf1.org/v1"
    assert settings.openf1.rate_limit.requests_per_second == 3
    assert settings.openf1.rate_limit.requests_per_minute == 30


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
