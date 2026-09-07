"""Result types for data validation.

The vocabulary here is deliberately small: a *rule* looks at something and
produces *results*; a result carries how serious the finding is
(:class:`Severity`) and whether the check actually found a problem
(:class:`RuleStatus`). Those two are separate on purpose — a rule that
passes still gets recorded, so a report says what was checked, not only
what went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class Severity(str, Enum):
    """How much a finding should worry the reader.

    The split follows what the data itself can tell us:

    * ``ERROR`` — impossible: the data contradicts what it claims to be
      (negative speed, a duplicate on a verified natural key, invalid JSON).
    * ``WARNING`` — implausible, or incomplete in a way we may need to act on.
    * ``INFO`` — unexpected but legitimate; recorded so it is visible, not
      because anything is wrong (a gap the source itself has, a tyre
      compound we had not seen before).
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RuleStatus(str, Enum):
    """Whether a rule found what it was looking for."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class GlobalStatus(str, Enum):
    """The verdict for a whole validation run."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ValidationResult:
    """One rule's finding about one file, endpoint, or the dataset as a whole."""

    rule: str
    description: str
    severity: Severity = Severity.INFO
    status: RuleStatus = RuleStatus.PASS
    endpoint: Optional[str] = None
    file: Optional[str] = None
    affected_records: Optional[int] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_problem(self) -> bool:
        return self.status is RuleStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rule": self.rule,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
        }
        optional = {
            "endpoint": self.endpoint,
            "file": self.file,
            "affected_records": self.affected_records,
            "details": dict(self.details) or None,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass
class ValidationReport:
    """Everything one validation run looked at and found."""

    pipeline_name: str
    pipeline_version: str
    source: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    scope: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    results: List[ValidationResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    fail_on_error: bool = True

    def add(self, result: ValidationResult) -> ValidationResult:
        self.results.append(result)
        return result

    def extend(self, results: List[ValidationResult]) -> None:
        self.results.extend(results)

    def problems(self, severity: Severity) -> List[ValidationResult]:
        return [item for item in self.results if item.is_problem and item.severity is severity]

    @property
    def errors(self) -> List[ValidationResult]:
        return self.problems(Severity.ERROR)

    @property
    def warnings(self) -> List[ValidationResult]:
        return self.problems(Severity.WARNING)

    @property
    def infos(self) -> List[ValidationResult]:
        return self.problems(Severity.INFO)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 3)

    @property
    def global_status(self) -> GlobalStatus:
        """PASS, WARNING or FAIL for the run as a whole.

        ``fail_on_error`` decides whether errors block: with it turned off
        an error still shows up, but the run degrades to WARNING instead of
        failing outright.
        """
        if self.errors:
            return GlobalStatus.FAIL if self.fail_on_error else GlobalStatus.WARNING
        if self.warnings:
            return GlobalStatus.WARNING
        return GlobalStatus.PASS

    def summary(self) -> Dict[str, int]:
        return {
            "rules_executed": len(self.results),
            "passed": sum(1 for item in self.results if item.status is RuleStatus.PASS),
            "skipped": sum(1 for item in self.results if item.status is RuleStatus.SKIPPED),
            "info": len(self.infos),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "source": self.source,
            "config": self.config,
            "scope": self.scope,
            "sampling": self.sampling,
            "global_status": self.global_status.value,
            "summary": self.summary(),
            "results": [item.to_dict() for item in self.results],
        }
