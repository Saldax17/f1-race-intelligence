"""Application configuration.

Configuration is layered, in increasing priority:

1. Defaults defined on the pydantic models below.
2. ``configs/config.yaml`` (or the file pointed to by ``F1_CONFIG_PATH``).
3. Environment variables (optionally loaded from a local ``.env`` file),
   which override specific YAML values. This is what lets the same code
   run locally and later inside a container/CI job without editing YAML.

Nothing in the rest of the codebase should read ``config.yaml`` or
``os.environ`` directly: everything goes through :func:`load_settings`,
which returns a validated, typed :class:`AppSettings` object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    load_dotenv = None


class TimeoutSettings(BaseModel):
    """HTTP timeout configuration, in seconds."""

    connect: float = 5.0
    read: float = 30.0


class RateLimitSettings(BaseModel):
    """Client-side rate limit configuration, matching OpenF1's published limits."""

    requests_per_second: float = 3.0
    requests_per_minute: float = 30.0


class RetrySettings(BaseModel):
    """Retry configuration for transient errors.

    ``max_attempts`` is the total number of tries (including the first one),
    not the number of retries. ``max_attempts=3`` means 1 initial attempt
    plus up to 2 retries.
    """

    max_attempts: int = 3
    backoff_factor: float = 1.0


class OpenF1Settings(BaseModel):
    """Everything :class:`~f1_race_intelligence.ingestion.client.BaseAPIClient` needs."""

    base_url: str = "https://api.openf1.org/v1"
    timeout: TimeoutSettings = Field(default_factory=TimeoutSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_format: bool = True


class AppSettings(BaseModel):
    """Top-level application settings."""

    openf1: OpenF1Settings = Field(default_factory=OpenF1Settings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


# src/f1_race_intelligence/config/settings.py -> parents[3] is the repo root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


def _apply_env_overrides(raw: dict) -> dict:
    """Apply environment variable overrides on top of the raw YAML dict.

    This is intentionally a small, explicit allowlist rather than a generic
    "flatten env vars into config" mechanism, so it stays predictable.
    """
    openf1 = raw.setdefault("openf1", {})

    if base_url := os.getenv("OPENF1_BASE_URL"):
        openf1["base_url"] = base_url

    if rps := os.getenv("OPENF1_RATE_LIMIT_RPS"):
        openf1.setdefault("rate_limit", {})["requests_per_second"] = float(rps)

    if rpm := os.getenv("OPENF1_RATE_LIMIT_RPM"):
        openf1.setdefault("rate_limit", {})["requests_per_minute"] = float(rpm)

    if max_attempts := os.getenv("OPENF1_RETRY_MAX_ATTEMPTS"):
        openf1.setdefault("retry", {})["max_attempts"] = int(max_attempts)

    logging_cfg = raw.setdefault("logging", {})
    if log_level := os.getenv("LOG_LEVEL"):
        logging_cfg["level"] = log_level

    return raw


def load_settings(config_path: Optional[Union[str, Path]] = None) -> AppSettings:
    """Load and validate application settings.

    Args:
        config_path: Optional explicit path to a YAML config file. If not
            given, uses ``F1_CONFIG_PATH`` if set, otherwise
            ``configs/config.yaml`` at the repo root.

    Returns:
        A validated :class:`AppSettings` instance.
    """
    if load_dotenv is not None:
        load_dotenv()

    path = Path(config_path) if config_path else Path(os.getenv("F1_CONFIG_PATH", DEFAULT_CONFIG_PATH))

    raw: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    raw = _apply_env_overrides(raw)
    return AppSettings.model_validate(raw)
