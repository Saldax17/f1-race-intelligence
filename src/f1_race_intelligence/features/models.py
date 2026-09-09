"""Result types for the feature engineering stage.

The report exists so that a dataset handed to a modelling stage can be
interrogated: which columns were kept, which were refused and on what
grounds, how the target was defined, what the rolling windows were, and
what the guards concluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SessionFeatureResult:
    """What one session contributed."""

    session_key: int
    year: Optional[int] = None
    rows_in: int = 0
    rows_out: int = 0
    rows_with_target: int = 0
    drivers: int = 0
    target_breakdown: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    blocking_failures: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    output_bytes: Optional[int] = None
    skipped: bool = False
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_key": self.session_key,
            "year": self.year,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_with_target": self.rows_with_target,
            "rows_without_target": self.rows_out - self.rows_with_target,
            "drivers": self.drivers,
            "target": self.target_breakdown,
            "checks": self.checks,
            "duration_seconds": self.duration_seconds,
        }
        for key, value in (
            ("output_path", self.output_path),
            ("output_bytes", self.output_bytes),
            ("error", self.error),
        ):
            if value is not None:
                payload[key] = value
        if self.blocking_failures:
            payload["blocking_failures"] = self.blocking_failures
        if self.skipped:
            payload["skipped"] = True
        return payload


@dataclass
class FeatureReport:
    """The record of one feature engineering run."""

    pipeline_name: str
    pipeline_version: str
    config: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    rolling: Dict[str, Any] = field(default_factory=dict)
    missing_values: Dict[str, Any] = field(default_factory=dict)
    missing_value_policy: str = ""
    sessions: List[SessionFeatureResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    def add(self, result: SessionFeatureResult) -> SessionFeatureResult:
        self.sessions.append(result)
        return result

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 3)

    def summary(self) -> Dict[str, Any]:
        done = [item for item in self.sessions if not item.skipped and item.error is None]
        return {
            "sessions_processed": len(done),
            "sessions_skipped": sum(1 for item in self.sessions if item.skipped),
            "sessions_failed": sum(1 for item in self.sessions if item.error),
            "rows_in": sum(item.rows_in for item in done),
            "rows_out": sum(item.rows_out for item in done),
            "rows_with_target": sum(item.rows_with_target for item in done),
            "rows_without_target": sum(item.rows_out - item.rows_with_target for item in done),
            "feature_count": self.features.get("count", 0),
            "output_bytes": sum(item.output_bytes or 0 for item in done),
            "sessions_with_blocking_failures": [
                item.session_key for item in self.sessions if item.blocking_failures
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "target": self.target,
            "features": self.features,
            "excluded_columns": self.excluded,
            "rolling_windows": self.rolling,
            "missing_value_policy": self.missing_value_policy,
            "missing_values": self.missing_values,
            "summary": self.summary(),
            "sessions": [item.to_dict() for item in self.sessions],
        }
