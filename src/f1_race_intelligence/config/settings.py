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
from typing import List, Optional, Union

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


class HistoricalExtractionSettings(BaseModel):
    """What the historical extraction pipeline should download.

    This is *data selection policy*, deliberately kept out of the ingestion
    layer: ``F1Client`` knows how to call OpenF1, this says what to ask it
    for. Changing years, session types, endpoints or the output location
    must never require a code change.

    ``session_types`` is matched (case-insensitively) against OpenF1's
    ``session_name`` *or* ``session_type`` field, so both ``"Race"`` and
    coarser values like ``"Practice"`` work. An empty list means "keep
    every session".
    """

    years: List[int] = Field(default_factory=list)
    session_types: List[str] = Field(default_factory=lambda: ["Race"])
    endpoints: List[str] = Field(default_factory=list)
    output_path: str = "data/raw"
    manifest_path: str = "data/manifests"
    overwrite: bool = False


class ValidationSettings(BaseModel):
    """How the extracted raw data should be judged.

    Only the knobs that change a verdict live here. Everything about *what*
    a field may contain is described per endpoint in
    :mod:`f1_race_intelligence.validation.specs`, where it can be justified
    next to the measurement that produced it.
    """

    raw_path: str = "data/raw"
    manifest_path: str = "data/manifests"
    report_path: str = "data/validation"

    fail_on_error: bool = True
    """An ERROR finding makes the run FAIL. Turn off to downgrade it to WARNING."""

    missing_value_threshold: float = 0.5
    """Null ratio above which a nullable field is reported as a WARNING."""

    max_records_per_file: Optional[int] = None
    """Inspect only the first N records of each file.

    ``None`` — the default — validates everything, and is the only mode
    whose result describes the dataset as a whole. Setting a value turns
    the run into a deterministic quick check: the report then says it was
    sampled, how much was inspected, and that the outcome is not
    exhaustive.
    """

    use_manifest: bool = True
    """Read the extraction manifest to explain why data is missing."""

    write_csv: bool = False
    """Also write the findings as a flat CSV, for triage in a spreadsheet."""

    enabled_rules: List[str] = Field(
        default_factory=lambda: [
            "structure",
            "schema",
            "types",
            "values",
            "missing_values",
            "duplicates",
            "temporal",
            "identifiers",
            "coverage",
        ]
    )


class ConsolidationSettings(BaseModel):
    """How the validated raw data is assembled into the lap dataset."""

    raw_path: str = "data/raw"
    output_path: str = "data/processed/lap_dataset"
    report_path: str = "data/consolidation"
    validation_path: str = "data/validation"

    years: List[int] = Field(default_factory=list)
    """Seasons to consolidate. Empty means everything found in the raw tree."""

    session_keys: List[int] = Field(default_factory=list)
    """Specific sessions to consolidate. Empty means all of them."""

    require_validation_pass: bool = False
    """Refuse to consolidate unless the latest validation run came back PASS.

    Off by default: a handful of impossible telemetry samples should not
    block an otherwise sound dataset. The validation verdict is recorded in
    the consolidation report either way, so a dataset built on data that
    failed validation always says so.
    """

    weather_tolerance_seconds: float = 120.0
    """How far back a lap may reach for the last weather reading.

    Readings arrive every 60 seconds; two cadences leaves room for one
    missed reading without pulling in conditions from far away.
    """

    overwrite: bool = False
    """Rebuild sessions whose output file already exists."""


class FeatureSettings(BaseModel):
    """How the consolidated laps become a modelling dataset.

    Only knobs that change the dataset live here. Which columns are features
    and why is described in
    :mod:`f1_race_intelligence.features.selection`, next to the evidence for
    each decision.
    """

    input_path: str = "data/processed/lap_dataset"
    output_path: str = "data/features/lap_features"
    report_path: str = "data/features/reports"

    years: List[int] = Field(default_factory=list)
    session_keys: List[int] = Field(default_factory=list)

    rolling_windows: List[int] = Field(default_factory=lambda: [3, 5])
    """Trailing windows, in laps, ending at the current lap."""

    correlation_alert_threshold: float = 0.99
    """Report any feature correlating with the target at or above this.

    A diagnostic only: a high correlation is a reason to look, not proof of
    leakage, so nothing is dropped automatically.
    """

    fail_on_leakage: bool = True
    """Refuse to write a session whose anti-leakage checks failed."""

    overwrite: bool = False


class AppSettings(BaseModel):
    """Top-level application settings."""

    openf1: OpenF1Settings = Field(default_factory=OpenF1Settings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    historical_extraction: HistoricalExtractionSettings = Field(default_factory=HistoricalExtractionSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)


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

    extraction = raw.setdefault("historical_extraction", {})
    if years := os.getenv("F1_EXTRACTION_YEARS"):
        extraction["years"] = [int(year) for year in years.split(",") if year.strip()]

    if overwrite := os.getenv("F1_EXTRACTION_OVERWRITE"):
        extraction["overwrite"] = overwrite.strip().lower() in {"1", "true", "yes"}

    if output_path := os.getenv("F1_EXTRACTION_OUTPUT_PATH"):
        extraction["output_path"] = output_path

    validation = raw.setdefault("validation", {})
    if raw_path := os.getenv("F1_VALIDATION_RAW_PATH"):
        validation["raw_path"] = raw_path

    if max_records := os.getenv("F1_VALIDATION_MAX_RECORDS"):
        validation["max_records_per_file"] = int(max_records)

    if fail_on_error := os.getenv("F1_VALIDATION_FAIL_ON_ERROR"):
        validation["fail_on_error"] = fail_on_error.strip().lower() in {"1", "true", "yes"}

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
