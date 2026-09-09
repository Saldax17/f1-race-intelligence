"""Result types for consolidation.

The point of these is accountability. A consolidated row is several joins
away from the raw files it came from, so every stage records how many rows
went in, how many came out, and how many found no match — which is what
makes "did we lose anything?" a question with an answer rather than a
hope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class StageCounter:
    """What one join or aggregation did to the row count."""

    stage: str
    rows_in: int
    rows_out: int
    matched: Optional[int] = None
    unmatched: Optional[int] = None
    note: str = ""

    @property
    def rows_lost(self) -> int:
        return self.rows_in - self.rows_out

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "stage": self.stage,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_lost": self.rows_lost,
        }
        if self.matched is not None:
            payload["matched"] = self.matched
        if self.unmatched is not None:
            payload["unmatched"] = self.unmatched
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class SessionResult:
    """Everything one session's consolidation produced."""

    session_key: int
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    rows_in: int = 0
    rows_out: int = 0
    drivers: int = 0
    laps: int = 0
    stages: List[StageCounter] = field(default_factory=list)
    car_data: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    output_bytes: Optional[int] = None
    skipped: bool = False
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def add_stage(self, counter: StageCounter) -> StageCounter:
        self.stages.append(counter)
        return counter

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_key": self.session_key,
            "year": self.year,
            "meeting_key": self.meeting_key,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_lost": self.rows_in - self.rows_out,
            "drivers": self.drivers,
            "laps": self.laps,
            "stages": [item.to_dict() for item in self.stages],
            "car_data": self.car_data,
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
        if self.skipped:
            payload["skipped"] = True
        return payload


@dataclass
class ConsolidationReport:
    """The record of one consolidation run."""

    pipeline_name: str
    pipeline_version: str
    config: Dict[str, Any] = field(default_factory=dict)
    source_validation: Dict[str, Any] = field(default_factory=dict)
    sessions: List[SessionResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    aborted: Optional[str] = None

    def add(self, result: SessionResult) -> SessionResult:
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
            "rows_lost": sum(item.rows_in - item.rows_out for item in done),
            "drivers": sum(item.drivers for item in done),
            "car_data_samples_read": sum(item.car_data.get("samples_read", 0) for item in done),
            "car_data_samples_in_window": sum(item.car_data.get("samples_in_window", 0) for item in done),
            "car_data_invalid_samples": sum(item.car_data.get("invalid_samples", 0) for item in done),
            "output_bytes": sum(item.output_bytes or 0 for item in done),
        }

    def checks(self) -> Dict[str, Any]:
        """Whether every session came out structurally sound."""
        done = [item for item in self.sessions if not item.skipped and item.error is None]
        return {
            "grain_unique": all(item.checks.get("grain_unique", False) for item in done) if done else None,
            "no_rows_lost": all(item.rows_in == item.rows_out for item in done) if done else None,
            "sessions_with_row_loss": [item.session_key for item in done if item.rows_in != item.rows_out],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "source_validation": self.source_validation,
            "aborted": self.aborted,
            "summary": self.summary(),
            "checks": self.checks(),
            "sessions": [item.to_dict() for item in self.sessions],
        }
