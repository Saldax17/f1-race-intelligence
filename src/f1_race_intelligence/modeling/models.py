"""Result types for the modelling dataset stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SplitArtifacts:
    """What one split produced on disk."""

    name: str
    rows: int = 0
    sessions: int = 0
    years: List[int] = field(default_factory=list)
    dataset_path: Optional[str] = None
    matrix_path: Optional[str] = None
    bytes_written: int = 0
    missing_before: Dict[str, float] = field(default_factory=dict)
    missing_after: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.name,
            "rows": self.rows,
            "sessions": self.sessions,
            "years": self.years,
            "dataset_path": self.dataset_path,
            "matrix_path": self.matrix_path,
            "bytes_written": self.bytes_written,
            "missing_before_pct": self.missing_before,
            "missing_after_pct": self.missing_after,
        }


@dataclass
class ModelingReport:
    """The record of one modelling-dataset run."""

    pipeline_name: str
    pipeline_version: str
    config: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    split: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[SplitArtifacts] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    blocking_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    aborted: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 3)

    def summary(self) -> Dict[str, Any]:
        by_split = {item.name: item.rows for item in self.artifacts}
        return {
            "rows_in": self.source.get("rows_in", 0),
            "rows_modelled": self.source.get("rows_out", 0),
            "rows_dropped_without_target": self.source.get("rows_dropped_without_target", 0),
            "sessions": self.source.get("sessions", 0),
            "rows_by_split": by_split,
            "numeric_features": len(self.features.get("numeric", [])),
            "categorical_features": len(self.features.get("categorical", [])),
            "encoded_feature_count": self.preprocessing.get("output_feature_count", 0),
            "checks_passed": not self.blocking_failures,
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "aborted": self.aborted,
            "config": self.config,
            "source": self.source,
            "target": self.target,
            "features": self.features,
            "split": self.split,
            "preprocessing": self.preprocessing,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "checks": self.checks,
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "summary": self.summary(),
        }
