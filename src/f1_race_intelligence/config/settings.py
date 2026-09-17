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
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

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
    """Logging configuration.

    Logs always go to stdout, which is what a container runtime collects.
    ``file_path`` adds a local log file for development; nothing in the
    pipeline depends on it existing.
    """

    level: str = "INFO"
    json_format: bool = True
    file_path: Optional[str] = None


class StorageSettings(BaseModel):
    """Where the pipeline's data lives.

    Credentials never belong here. On AWS the S3 client resolves them from
    the IAM role of the task or job running the container; locally, from
    the standard AWS credential chain.
    """

    backend: Literal["local", "s3"] = "local"
    """``local`` writes files; ``s3`` writes objects to ``bucket``."""

    layout: Literal["auto", "legacy", "lake"] = "auto"
    """Which path contract to use (see :mod:`f1_race_intelligence.storage.layout`).

    ``auto`` keeps the historical tree (``legacy``) on local storage and uses
    the data lake contract (``lake``) on S3. The legacy layout honours the
    per-stage paths below; the lake layout has fixed layer names.
    """

    local_root: Optional[str] = None
    """Directory local keys are resolved against. Defaults to the working
    directory for the legacy layout and to ``data/lake`` for the lake layout."""

    bucket: Optional[str] = None
    prefix: str = ""
    """Key prefix inside the bucket, so one bucket can hold several environments."""

    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    """Only for S3-compatible endpoints such as LocalStack or MinIO."""

    @model_validator(mode="after")
    def _s3_needs_a_bucket(self) -> "StorageSettings":
        if self.backend == "s3" and not self.bucket:
            raise ValueError("storage.backend is 's3' but storage.bucket is empty (set F1_STORAGE_BUCKET)")
        if self.bucket and ("/" in self.bucket or self.bucket.startswith("s3:")):
            raise ValueError(f"storage.bucket must be a bare bucket name, got {self.bucket!r}")
        return self


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


class ModelingSettings(BaseModel):
    """How the feature dataset becomes train/validation/test artifacts.

    Everything that decides what a model will see lives here rather than in
    code, so a different split or a different imputation is a configuration
    change and the report records exactly which one produced a dataset.
    """

    input_path: str = "data/features/lap_features"
    output_path: str = "data/modeling"
    report_path: str = "data/modeling/reports"

    years: List[int] = Field(default_factory=list)
    session_keys: List[int] = Field(default_factory=list)

    split_strategy: str = "fraction"
    """How races are handed to the splits.

    * ``fraction`` cuts the chronological list of races by proportion.
    * ``years`` assigns whole seasons through the lists below.
    * ``train_years_then_split`` gives the seasons in ``train_years`` to
      training and cuts whatever remains into validation and test.
    """

    validation_share_of_remainder: float = 0.5
    """For ``train_years_then_split``: how much of what is left goes to validation.

    The rest becomes test, so test always sits after validation in time.
    With an odd number of races the extra one goes to validation.
    """

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    """Test takes whatever remains, so the three always sum to the history."""

    train_years: List[int] = Field(default_factory=list)
    validation_years: List[int] = Field(default_factory=list)
    test_years: List[int] = Field(default_factory=list)

    min_sessions_per_split: int = 3
    """Below this a split is reported as too thin to evaluate anything.

    A warning, not an error: the pipeline still runs on a small history so
    it can be validated technically, but the report says the numbers are
    not an evaluation.
    """

    numeric_imputation: str = "median"
    """Learned from the training races only. The median resists the outliers
    that pit laps and safety cars put into lap times."""

    add_missing_indicators: bool = True
    """Keep a column recording that a value was absent.

    Missing here carries meaning — a lapped car has no gap, a first lap has
    no previous lap — and imputing without a marker would erase it.
    """

    scale_numeric: bool = True
    """Standardize numeric features using training statistics only.

    Needed by distance- and gradient-based models and harmless to trees,
    which is why it is on by default; turn it off if the chosen model makes
    it pointless.
    """

    random_seed: int = 42
    """Recorded for reproducibility. Nothing in this stage is random today."""

    fail_on_leakage: bool = True
    """Refuse to write anything when a blocking guard fails."""


class AppSettings(BaseModel):
    """Top-level application settings."""

    environment: Literal["local", "aws"] = "local"
    """Where the pipeline runs. ``aws`` loads ``config.aws.yaml`` on top of the
    base file and requires S3 storage, because a container's disk does not
    outlive the job."""

    storage: StorageSettings = Field(default_factory=StorageSettings)
    openf1: OpenF1Settings = Field(default_factory=OpenF1Settings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    historical_extraction: HistoricalExtractionSettings = Field(default_factory=HistoricalExtractionSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    modeling: ModelingSettings = Field(default_factory=ModelingSettings)

    @model_validator(mode="after")
    def _aws_runs_on_object_storage(self) -> "AppSettings":
        if self.environment == "aws" and self.storage.backend != "s3":
            raise ValueError("environment 'aws' requires storage.backend 's3'")
        return self


# src/f1_race_intelligence/config/settings.py -> parents[3] is the repo root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.yaml"


_TRUE_VALUES = {"1", "true", "yes"}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``overlay`` into ``base``: nested sections merge, everything else is replaced."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def overlay_path(config_path: Path, environment: str) -> Path:
    """``configs/config.yaml`` + ``aws`` -> ``configs/config.aws.yaml``."""
    return config_path.with_name(f"{config_path.stem}.{environment}{config_path.suffix}")


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

    if log_file := os.getenv("F1_LOG_FILE"):
        logging_cfg["file_path"] = log_file

    if environment := os.getenv("F1_ENVIRONMENT"):
        raw["environment"] = environment.strip().lower()

    storage = raw.setdefault("storage", {})
    for variable, key in (
        ("F1_STORAGE_BACKEND", "backend"),
        ("F1_STORAGE_LAYOUT", "layout"),
        ("F1_STORAGE_LOCAL_ROOT", "local_root"),
        ("F1_STORAGE_BUCKET", "bucket"),
        ("F1_STORAGE_PREFIX", "prefix"),
        ("F1_STORAGE_REGION", "region"),
        ("F1_STORAGE_ENDPOINT_URL", "endpoint_url"),
    ):
        if (value := os.getenv(variable)) and value.strip():
            storage[key] = value.strip()

    extraction = raw.setdefault("historical_extraction", {})
    if years := os.getenv("F1_EXTRACTION_YEARS"):
        extraction["years"] = [int(year) for year in years.split(",") if year.strip()]

    if overwrite := os.getenv("F1_EXTRACTION_OVERWRITE"):
        extraction["overwrite"] = overwrite.strip().lower() in _TRUE_VALUES

    if output_path := os.getenv("F1_EXTRACTION_OUTPUT_PATH"):
        extraction["output_path"] = output_path

    validation = raw.setdefault("validation", {})
    if raw_path := os.getenv("F1_VALIDATION_RAW_PATH"):
        validation["raw_path"] = raw_path

    if max_records := os.getenv("F1_VALIDATION_MAX_RECORDS"):
        validation["max_records_per_file"] = int(max_records)

    if fail_on_error := os.getenv("F1_VALIDATION_FAIL_ON_ERROR"):
        validation["fail_on_error"] = fail_on_error.strip().lower() in _TRUE_VALUES

    return raw


def load_settings(
    config_path: Optional[Union[str, Path]] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppSettings:
    """Load and validate application settings.

    Layers, lowest priority first: model defaults, the base YAML file, the
    environment overlay (``config.<environment>.yaml`` next to it, when the
    environment is not ``local`` and that file exists), environment
    variables, and finally ``overrides`` — which is how a command-line flag
    wins over everything else.

    Args:
        config_path: Optional explicit path to a YAML config file. If not
            given, uses ``F1_CONFIG_PATH`` if set, otherwise
            ``configs/config.yaml`` at the repo root.
        overrides: Nested values applied last, e.g.
            ``{"storage": {"backend": "s3", "bucket": "my-bucket"}}``.

    Returns:
        A validated :class:`AppSettings` instance.
    """
    if load_dotenv is not None:
        load_dotenv()

    path = Path(config_path) if config_path else Path(os.getenv("F1_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    raw = _read_yaml(path)

    # The environment decides which overlay to read, so resolve it first.
    environment = (overrides or {}).get("environment") or os.getenv("F1_ENVIRONMENT") or raw.get("environment") or "local"
    environment = str(environment).strip().lower()
    if environment != "local":
        raw = _deep_merge(raw, _read_yaml(overlay_path(path, environment)))

    raw = _apply_env_overrides(raw)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return AppSettings.model_validate(raw)
