"""The validation rules themselves.

Rules come in two families:

* **File rules** look at one raw file: structure, schema, types, values,
  missing values, duplicates and temporal consistency.
* **Dataset rules** look at everything at once: referential integrity
  between endpoints, and coverage.

Every file rule reads a :class:`RecordAnalysis` that is computed in a
single pass over the records, rather than iterating the file once per
rule. With telemetry files holding tens of thousands of records each, the
difference between one pass and seven is the difference between a
validation run that finishes and one nobody waits for.

File rules report only problems. The runner adds one passing result per
endpoint and rule that came out clean, so the report says what was
checked without carrying a line per file.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from f1_race_intelligence.validation.models import RuleStatus, Severity, ValidationResult
from f1_race_intelligence.validation.specs import EndpointSpec, FieldSpec

RULE_STRUCTURE = "structure"
RULE_SCHEMA = "schema"
RULE_TYPES = "types"
RULE_VALUES = "values"
RULE_MISSING_VALUES = "missing_values"
RULE_DUPLICATES = "duplicates"
RULE_TEMPORAL = "temporal"
RULE_IDENTIFIERS = "identifiers"
RULE_COVERAGE = "coverage"

ALL_RULES: Tuple[str, ...] = (
    RULE_STRUCTURE,
    RULE_SCHEMA,
    RULE_TYPES,
    RULE_VALUES,
    RULE_MISSING_VALUES,
    RULE_DUPLICATES,
    RULE_TEMPORAL,
    RULE_IDENTIFIERS,
    RULE_COVERAGE,
)

FILE_RULES: Tuple[str, ...] = (
    RULE_STRUCTURE,
    RULE_SCHEMA,
    RULE_TYPES,
    RULE_VALUES,
    RULE_MISSING_VALUES,
    RULE_DUPLICATES,
    RULE_TEMPORAL,
)

_MAX_EXAMPLES = 5


# --------------------------------------------------------------------------
# Single-pass analysis
# --------------------------------------------------------------------------


@dataclass
class FieldStats:
    """What one pass over the records found out about one field."""

    present: int = 0
    missing: int = 0
    nulls: int = 0
    type_violations: int = 0
    type_examples: Set[str] = field(default_factory=set)
    below_minimum: int = 0
    above_maximum: int = 0
    outside_plausible: int = 0
    outside_documented: int = 0
    not_allowed: int = 0
    pattern_violations: int = 0
    examples: List[Any] = field(default_factory=list)
    unexpected_values: List[Any] = field(default_factory=list)

    def note_example(self, value: Any) -> None:
        if len(self.examples) < _MAX_EXAMPLES:
            self.examples.append(value)

    def note_unexpected(self, value: Any) -> None:
        if value not in self.unexpected_values and len(self.unexpected_values) < _MAX_EXAMPLES:
            self.unexpected_values.append(value)


@dataclass
class RecordAnalysis:
    """Everything the file rules need, gathered in one pass."""

    total: int = 0
    non_mapping: int = 0
    fields: Dict[str, FieldStats] = field(default_factory=dict)
    unknown_fields: Set[str] = field(default_factory=set)
    key_variants: int = 0
    duplicate_groups: int = 0
    duplicate_records: int = 0
    duplicate_examples: List[Any] = field(default_factory=list)
    unparseable_timestamps: Dict[str, int] = field(default_factory=dict)
    timestamp_examples: Dict[str, Any] = field(default_factory=dict)
    order_inversions: int = 0
    order_example: Optional[Tuple[Any, Any]] = None

    def stats(self, name: str) -> FieldStats:
        return self.fields.setdefault(name, FieldStats())


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _matches_types(value: Any, types: Sequence[type]) -> bool:
    """Type check that keeps ``bool`` and ``int`` apart.

    ``isinstance(True, int)`` is true in Python, which would let a boolean
    pass as a lap number and an integer pass as a flag.
    """
    if _is_bool(value):
        return bool in types
    return isinstance(value, tuple(types))


def _as_number(value: Any) -> Optional[float]:
    """The value as a number, or ``None`` if it is not one.

    Strings are deliberately excluded: ``intervals.gap_to_leader`` is
    allowed to be ``"+1 LAP"``, and a text gap has no numeric bounds to
    check.
    """
    if _is_bool(value) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def analyze_records(records: Sequence[Any], spec: Optional[EndpointSpec]) -> RecordAnalysis:
    """Walk the records once, collecting what every file rule needs."""
    analysis = RecordAnalysis(total=len(records))
    if spec is None:
        analysis.non_mapping = sum(1 for record in records if not isinstance(record, Mapping))
        return analysis

    known_fields = set(spec.fields)
    key_signatures: Set[frozenset] = set()
    duplicate_counter: Counter = Counter()
    timestamp_failures: Counter = Counter()
    last_ordered: Dict[Any, Any] = {}

    for record in records:
        if not isinstance(record, Mapping):
            analysis.non_mapping += 1
            continue

        key_signatures.add(frozenset(record.keys()))
        analysis.unknown_fields |= set(record.keys()) - known_fields

        for name, field_spec in spec.fields.items():
            _inspect_field(analysis.stats(name), record, name, field_spec)

        if spec.natural_key:
            duplicate_counter[tuple(_hashable(record.get(part)) for part in spec.natural_key)] += 1

        for name in spec.timestamp_fields:
            value = record.get(name)
            if value is None:
                continue
            if _parse_timestamp(value) is None:
                timestamp_failures[name] += 1
                analysis.timestamp_examples.setdefault(name, value)

        if spec.order_field:
            _inspect_order(analysis, record, spec, last_ordered)

    analysis.key_variants = len(key_signatures)
    analysis.unparseable_timestamps = dict(timestamp_failures)

    for key, count in duplicate_counter.items():
        if count > 1:
            analysis.duplicate_groups += 1
            analysis.duplicate_records += count - 1
            if len(analysis.duplicate_examples) < _MAX_EXAMPLES:
                analysis.duplicate_examples.append(list(key))

    return analysis


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _inspect_field(stats: FieldStats, record: Mapping[str, Any], name: str, spec: FieldSpec) -> None:
    if name not in record:
        stats.missing += 1
        return

    value = record[name]
    stats.present += 1

    if value is None:
        stats.nulls += 1
        return

    if not _matches_types(value, spec.types):
        stats.type_violations += 1
        stats.type_examples.add(type(value).__name__)
        stats.note_example(value)
        return

    if isinstance(value, str) and spec.string_pattern and not re.match(spec.string_pattern, value):
        stats.pattern_violations += 1
        stats.note_unexpected(value)

    if spec.allowed is not None and value not in spec.allowed:
        stats.not_allowed += 1
        stats.note_unexpected(value)

    number = _as_number(value)
    if number is None:
        return

    if spec.minimum is not None:
        if number < spec.minimum or (spec.exclusive_minimum and number == spec.minimum):
            stats.below_minimum += 1
            stats.note_example(value)
            return
    if spec.maximum is not None and number > spec.maximum:
        stats.above_maximum += 1
        stats.note_example(value)
        return

    if spec.plausible is not None and not (spec.plausible[0] <= number <= spec.plausible[1]):
        stats.outside_plausible += 1
        stats.note_example(value)
    if spec.documented is not None and not (spec.documented[0] <= number <= spec.documented[1]):
        stats.outside_documented += 1
        stats.note_unexpected(value)


def _inspect_order(
    analysis: RecordAnalysis,
    record: Mapping[str, Any],
    spec: EndpointSpec,
    last_ordered: Dict[Any, Any],
) -> None:
    raw_value = record.get(spec.order_field)
    if raw_value is None:
        return

    value = _parse_timestamp(raw_value) if spec.order_field in spec.timestamp_fields else raw_value
    if value is None:
        return

    group = record.get(spec.order_group) if spec.order_group else None
    previous = last_ordered.get(group)
    if previous is not None:
        try:
            if value < previous:
                analysis.order_inversions += 1
                if analysis.order_example is None:
                    analysis.order_example = (str(previous), str(raw_value))
        except TypeError:
            return
    last_ordered[group] = value


# --------------------------------------------------------------------------
# File-level context and rules
# --------------------------------------------------------------------------


@dataclass
class FileContext:
    """One raw file, ready to be validated."""

    endpoint: str
    path: Path
    partition: Mapping[str, Any]
    spec: Optional[EndpointSpec]
    records: Sequence[Any]
    total_records: int
    sampled: bool = False
    missing_value_threshold: float = 0.5

    @property
    def file_label(self) -> str:
        return str(self.path)

    @property
    def session_key(self) -> Optional[int]:
        return self.partition.get("session_key")

    def result(
        self,
        rule: str,
        description: str,
        *,
        severity: Severity,
        affected_records: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        payload = dict(details or {})
        payload.update({key: value for key, value in self.partition.items()})
        return ValidationResult(
            rule=rule,
            description=description,
            severity=severity,
            status=RuleStatus.FAIL,
            endpoint=self.endpoint,
            file=self.file_label,
            affected_records=affected_records,
            details=payload,
        )


def validate_structure(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """The payload is a list of records, and there are some."""
    results: List[ValidationResult] = []

    if analysis.non_mapping:
        results.append(
            context.result(
                RULE_STRUCTURE,
                "Payload contains entries that are not JSON objects",
                severity=Severity.ERROR,
                affected_records=analysis.non_mapping,
            )
        )

    if analysis.total == 0:
        results.append(
            context.result(
                RULE_STRUCTURE,
                "File holds no records",
                severity=Severity.WARNING,
                affected_records=0,
            )
        )

    if analysis.key_variants > 1:
        results.append(
            context.result(
                RULE_STRUCTURE,
                "Records do not all carry the same set of fields",
                severity=Severity.INFO,
                details={"distinct_field_sets": analysis.key_variants},
            )
        )

    return results


def validate_schema(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Required fields are present; new fields are surfaced."""
    if context.spec is None:
        return []

    results: List[ValidationResult] = []
    for name in context.spec.required_fields:
        stats = analysis.fields.get(name)
        if stats and stats.missing:
            results.append(
                context.result(
                    RULE_SCHEMA,
                    f"Required field '{name}' is absent from some records",
                    severity=Severity.ERROR,
                    affected_records=stats.missing,
                    details={"field": name},
                )
            )

    if analysis.unknown_fields:
        results.append(
            context.result(
                RULE_SCHEMA,
                "Records carry fields this project does not describe yet",
                severity=Severity.INFO,
                details={"fields": sorted(analysis.unknown_fields)},
            )
        )

    return results


def validate_types(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Values have the types the endpoint spec expects."""
    if context.spec is None:
        return []

    results: List[ValidationResult] = []
    for name, spec in context.spec.fields.items():
        stats = analysis.fields.get(name)
        if not stats or not stats.type_violations:
            continue
        expected = sorted(item.__name__ for item in spec.types)
        results.append(
            context.result(
                RULE_TYPES,
                f"Field '{name}' has values of an unexpected type",
                severity=Severity.ERROR,
                affected_records=stats.type_violations,
                details={
                    "field": name,
                    "expected_types": expected,
                    "observed_types": sorted(stats.type_examples),
                    "examples": stats.examples,
                },
            )
        )
    return results


def validate_values(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Values sit inside bounds that are impossible, implausible or merely surprising."""
    if context.spec is None:
        return []

    results: List[ValidationResult] = []
    for name, spec in context.spec.fields.items():
        stats = analysis.fields.get(name)
        if stats is None:
            continue

        impossible = stats.below_minimum + stats.above_maximum
        if impossible:
            bound = "below the minimum" if stats.below_minimum else "outside the valid range"
            results.append(
                context.result(
                    RULE_VALUES,
                    f"Field '{name}' has values {bound}",
                    severity=Severity.ERROR,
                    affected_records=impossible,
                    details={
                        "field": name,
                        "minimum": spec.minimum,
                        "maximum": spec.maximum,
                        "below_minimum": stats.below_minimum,
                        "above_maximum": stats.above_maximum,
                        "examples": stats.examples,
                        "note": spec.note,
                    },
                )
            )

        if stats.outside_plausible:
            results.append(
                context.result(
                    RULE_VALUES,
                    f"Field '{name}' has values outside the plausible range",
                    severity=Severity.WARNING,
                    affected_records=stats.outside_plausible,
                    details={"field": name, "plausible": list(spec.plausible or ()), "examples": stats.examples},
                )
            )

        if stats.outside_documented:
            results.append(
                context.result(
                    RULE_VALUES,
                    f"Field '{name}' exceeds its documented range but is accepted as real data",
                    severity=Severity.INFO,
                    affected_records=stats.outside_documented,
                    details={
                        "field": name,
                        "documented": list(spec.documented or ()),
                        "observed": stats.unexpected_values,
                        "note": spec.note,
                    },
                )
            )

        if stats.not_allowed:
            results.append(
                context.result(
                    RULE_VALUES,
                    f"Field '{name}' has values outside the known set",
                    severity=Severity.INFO,
                    affected_records=stats.not_allowed,
                    details={"field": name, "observed": stats.unexpected_values, "note": spec.note},
                )
            )

        if stats.pattern_violations:
            results.append(
                context.result(
                    RULE_VALUES,
                    f"Field '{name}' has text values in an unrecognized format",
                    severity=Severity.INFO,
                    affected_records=stats.pattern_violations,
                    details={"field": name, "expected_pattern": spec.string_pattern, "observed": stats.unexpected_values},
                )
            )

    results.extend(_validate_cross_fields(context))
    return results


def _validate_cross_fields(context: FileContext) -> List[ValidationResult]:
    """Relations between fields of the same record.

    Only checks the source actually guarantees: a stint cannot end before
    it starts.
    """
    if context.endpoint != "stints":
        return []

    broken = 0
    examples: List[Any] = []
    for record in context.records:
        if not isinstance(record, Mapping):
            continue
        start, end = record.get("lap_start"), record.get("lap_end")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            broken += 1
            if len(examples) < _MAX_EXAMPLES:
                examples.append({"lap_start": start, "lap_end": end})

    if not broken:
        return []
    return [
        context.result(
            RULE_VALUES,
            "Stints end before they start",
            severity=Severity.ERROR,
            affected_records=broken,
            details={"examples": examples},
        )
    ]


def validate_missing_values(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Nulls are counted and judged against what the field is allowed to be.

    Nothing is dropped or filled: raw data stays as extracted, and this
    only says how much of it is absent.
    """
    if context.spec is None or analysis.total == 0:
        return []

    results: List[ValidationResult] = []
    for name, spec in context.spec.fields.items():
        stats = analysis.fields.get(name)
        if stats is None or stats.nulls == 0:
            continue

        ratio = stats.nulls / analysis.total
        details = {
            "field": name,
            "null_records": stats.nulls,
            "null_ratio": round(ratio, 4),
            "note": spec.note,
        }

        if not spec.nullable:
            results.append(
                context.result(
                    RULE_MISSING_VALUES,
                    f"Field '{name}' is null although it must always have a value",
                    severity=Severity.ERROR,
                    affected_records=stats.nulls,
                    details=details,
                )
            )
        elif spec.sparse:
            results.append(
                context.result(
                    RULE_MISSING_VALUES,
                    f"Field '{name}' is sparsely populated, as expected for this endpoint",
                    severity=Severity.INFO,
                    affected_records=stats.nulls,
                    details=details,
                )
            )
        elif ratio > context.missing_value_threshold:
            results.append(
                context.result(
                    RULE_MISSING_VALUES,
                    f"Field '{name}' is missing in more than the configured share of records",
                    severity=Severity.WARNING,
                    affected_records=stats.nulls,
                    details={**details, "threshold": context.missing_value_threshold},
                )
            )

    return results


def validate_duplicates(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Records are unique on the endpoint's natural key.

    Endpoints without a defensible key are skipped rather than checked
    against an invented one.
    """
    if context.spec is None:
        return []

    if context.spec.natural_key is None:
        return [
            ValidationResult(
                rule=RULE_DUPLICATES,
                description="No natural key is defined for this endpoint; duplicate detection skipped",
                severity=Severity.INFO,
                status=RuleStatus.SKIPPED,
                endpoint=context.endpoint,
                file=context.file_label,
            )
        ]

    if not analysis.duplicate_records:
        return []

    return [
        context.result(
            RULE_DUPLICATES,
            "Records repeat on the endpoint's natural key",
            severity=Severity.ERROR,
            affected_records=analysis.duplicate_records,
            details={
                "natural_key": list(context.spec.natural_key),
                "duplicate_groups": analysis.duplicate_groups,
                "examples": analysis.duplicate_examples,
                "note": context.spec.natural_key_note,
            },
        )
    ]


def validate_temporal(context: FileContext, analysis: RecordAnalysis) -> List[ValidationResult]:
    """Timestamps parse, and records arrive in the order they claim."""
    if context.spec is None:
        return []

    results: List[ValidationResult] = []
    for name, count in sorted(analysis.unparseable_timestamps.items()):
        results.append(
            context.result(
                RULE_TEMPORAL,
                f"Field '{name}' has values that are not valid ISO 8601 timestamps",
                severity=Severity.ERROR,
                affected_records=count,
                details={"field": name, "example": analysis.timestamp_examples.get(name)},
            )
        )

    if analysis.order_inversions:
        results.append(
            context.result(
                RULE_TEMPORAL,
                f"Records are not ordered by '{context.spec.order_field}' as expected",
                severity=Severity.WARNING,
                affected_records=analysis.order_inversions,
                details={
                    "order_field": context.spec.order_field,
                    "grouped_by": context.spec.order_group,
                    "example": analysis.order_example,
                },
            )
        )

    return results


FILE_RULE_FUNCTIONS = {
    RULE_STRUCTURE: validate_structure,
    RULE_SCHEMA: validate_schema,
    RULE_TYPES: validate_types,
    RULE_VALUES: validate_values,
    RULE_MISSING_VALUES: validate_missing_values,
    RULE_DUPLICATES: validate_duplicates,
    RULE_TEMPORAL: validate_temporal,
}


# --------------------------------------------------------------------------
# Dataset-level index and rules
# --------------------------------------------------------------------------


@dataclass
class DatasetIndex:
    """Facts accumulated while streaming files, for the cross-file rules.

    Only aggregates are kept — keys, counts and sets — never the records
    themselves, so validating a full season does not depend on how much
    telemetry it holds.
    """

    sessions_declared: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    meetings_declared: Set[int] = field(default_factory=set)
    meeting_refs: Set[int] = field(default_factory=set)
    drivers_by_session: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    session_keys_by_endpoint: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    lap_drivers_by_session: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    laps_per_driver: Dict[Tuple[int, int], int] = field(default_factory=Counter)
    records_by_endpoint: Dict[str, int] = field(default_factory=Counter)
    records_by_endpoint_session: Dict[Tuple[str, int], int] = field(default_factory=Counter)
    files_by_endpoint: Dict[str, int] = field(default_factory=Counter)
    session_years: Dict[int, int] = field(default_factory=dict)
    endpoints_present: Set[str] = field(default_factory=set)

    def observe(self, context: FileContext) -> None:
        """Fold one file into the index."""
        endpoint = context.endpoint
        self.endpoints_present.add(endpoint)
        self.files_by_endpoint[endpoint] += 1
        self.records_by_endpoint[endpoint] += context.total_records

        session_key = context.session_key
        if session_key is not None:
            self.records_by_endpoint_session[(endpoint, session_key)] += context.total_records
            self.session_keys_by_endpoint[endpoint].add(session_key)
            year = context.partition.get("year")
            if year is not None:
                self.session_years.setdefault(session_key, year)

        for record in context.records:
            if not isinstance(record, Mapping):
                continue
            self._observe_record(endpoint, record, session_key)

    def _observe_record(self, endpoint: str, record: Mapping[str, Any], file_session: Optional[int]) -> None:
        if endpoint == "meetings":
            key = record.get("meeting_key")
            if isinstance(key, int):
                self.meetings_declared.add(key)
            return

        if endpoint == "sessions":
            key = record.get("session_key")
            if isinstance(key, int):
                self.sessions_declared[key] = {
                    "meeting_key": record.get("meeting_key"),
                    "year": record.get("year"),
                    "session_name": record.get("session_name"),
                    "session_type": record.get("session_type"),
                }
                if isinstance(record.get("year"), int):
                    self.session_years.setdefault(key, record["year"])
            if isinstance(record.get("meeting_key"), int):
                self.meeting_refs.add(record["meeting_key"])
            return

        session_key = record.get("session_key")
        if isinstance(session_key, int):
            self.session_keys_by_endpoint[endpoint].add(session_key)
        else:
            session_key = file_session

        driver_number = record.get("driver_number")
        if not isinstance(session_key, int) or not isinstance(driver_number, int):
            return

        if endpoint == "drivers":
            self.drivers_by_session[session_key].add(driver_number)
        elif endpoint == "laps":
            self.lap_drivers_by_session[session_key].add(driver_number)
            self.laps_per_driver[(session_key, driver_number)] += 1


def validate_identifiers(index: DatasetIndex) -> List[ValidationResult]:
    """References between endpoints resolve.

    Only relations OpenF1 actually guarantees are checked: every record
    belongs to a session, and every session to a meeting. A driver seen in
    laps but not in ``drivers`` is a warning, not an error — the drivers
    endpoint may simply not have been extracted.
    """
    results: List[ValidationResult] = []

    if not index.sessions_declared:
        results.append(
            ValidationResult(
                rule=RULE_IDENTIFIERS,
                description="No sessions catalog found; cross-endpoint session checks skipped",
                severity=Severity.INFO,
                status=RuleStatus.SKIPPED,
            )
        )
    else:
        known = set(index.sessions_declared)
        for endpoint in sorted(index.session_keys_by_endpoint):
            if endpoint in {"meetings", "sessions"}:
                continue
            unknown = sorted(index.session_keys_by_endpoint[endpoint] - known)
            if unknown:
                results.append(
                    ValidationResult(
                        rule=RULE_IDENTIFIERS,
                        description=f"'{endpoint}' references sessions that are not in the sessions catalog",
                        severity=Severity.ERROR,
                        status=RuleStatus.FAIL,
                        endpoint=endpoint,
                        affected_records=len(unknown),
                        details={"unknown_session_keys": unknown[:_MAX_EXAMPLES]},
                    )
                )

    if index.meetings_declared and index.meeting_refs:
        unknown_meetings = sorted(index.meeting_refs - index.meetings_declared)
        if unknown_meetings:
            results.append(
                ValidationResult(
                    rule=RULE_IDENTIFIERS,
                    description="Sessions reference meetings that are not in the meetings catalog",
                    severity=Severity.ERROR,
                    status=RuleStatus.FAIL,
                    endpoint="sessions",
                    affected_records=len(unknown_meetings),
                    details={"unknown_meeting_keys": unknown_meetings[:_MAX_EXAMPLES]},
                )
            )

    for session_key in sorted(index.lap_drivers_by_session):
        entered = index.drivers_by_session.get(session_key)
        if not entered:
            continue
        unknown_drivers = sorted(index.lap_drivers_by_session[session_key] - entered)
        if unknown_drivers:
            results.append(
                ValidationResult(
                    rule=RULE_IDENTIFIERS,
                    description="Laps reference drivers that are not listed for the session",
                    severity=Severity.WARNING,
                    status=RuleStatus.FAIL,
                    endpoint="laps",
                    affected_records=len(unknown_drivers),
                    details={"session_key": session_key, "driver_numbers": unknown_drivers},
                )
            )

    return results


def validate_coverage(
    index: DatasetIndex,
    expected_endpoints: Sequence[str],
    manifest: Optional["ManifestLookup"] = None,
    expected_sessions: Optional[Sequence[int]] = None,
) -> List[ValidationResult]:
    """How complete the extraction is, and why anything is missing.

    A missing file is not automatically a problem. When the extraction
    manifest says the request came back ``404 No results found``, the data
    does not exist at the source — pit stops for the whole 2023 season, for
    instance — and that is reported as INFO. A missing file that failed for
    any other reason, or that was never attempted, is a WARNING because
    re-running would change the outcome.

    Args:
        expected_sessions: The sessions the extraction actually targeted.
            This matters: the sessions catalogue lists every session of a
            weekend, but an extraction configured for races only was never
            meant to hold practice data, and demanding it would report a
            gap where there is none.
    """
    results: List[ValidationResult] = []
    with_data = {session for keys in index.session_keys_by_endpoint.values() for session in keys}

    if expected_sessions is not None:
        sessions = sorted(expected_sessions)
    else:
        sessions = sorted(with_data)

    not_selected = sorted(set(index.sessions_declared) - set(sessions))
    if not_selected:
        results.append(
            ValidationResult(
                rule=RULE_COVERAGE,
                description="Sessions in the catalogue were not part of this extraction",
                severity=Severity.INFO,
                status=RuleStatus.PASS,
                endpoint="sessions",
                details={
                    "sessions_not_extracted": len(not_selected),
                    "session_keys": not_selected[:_MAX_EXAMPLES],
                    "note": "Expected when the extraction is configured for specific session types.",
                },
            )
        )

    if not sessions:
        return [
            ValidationResult(
                rule=RULE_COVERAGE,
                description="No sessions found in the raw data",
                severity=Severity.WARNING,
                status=RuleStatus.FAIL,
                details={"raw_endpoints_present": sorted(index.endpoints_present)},
            )
        ]

    unavailable: Dict[Tuple[str, Optional[int]], List[int]] = defaultdict(list)
    failed: Dict[Tuple[str, Optional[str]], List[int]] = defaultdict(list)
    absent: Dict[str, List[int]] = defaultdict(list)

    for endpoint in expected_endpoints:
        present = index.session_keys_by_endpoint.get(endpoint, set())
        for session_key in sessions:
            if session_key in present:
                continue
            year = index.session_years.get(session_key)
            failure = manifest.failure_for(endpoint, session_key) if manifest else None
            if failure is None:
                absent[endpoint].append(session_key)
            elif failure.get("status_code") == 404:
                unavailable[(endpoint, year)].append(session_key)
            else:
                reason = failure.get("error_type") or str(failure.get("status_code"))
                failed[(endpoint, reason)].append(session_key)

    for (endpoint, year), keys in sorted(unavailable.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        season = f" for {year}" if year else ""
        results.append(
            ValidationResult(
                rule=RULE_COVERAGE,
                description=f"{endpoint} data unavailable{season} in source",
                severity=Severity.INFO,
                status=RuleStatus.FAIL,
                endpoint=endpoint,
                affected_records=len(keys),
                details={
                    "year": year,
                    "sessions_without_data": len(keys),
                    "session_keys": sorted(keys)[:_MAX_EXAMPLES],
                    "reason": "The extraction received HTTP 404 'No results found' from OpenF1",
                },
            )
        )

    for (endpoint, reason), keys in sorted(failed.items(), key=lambda item: item[0]):
        results.append(
            ValidationResult(
                rule=RULE_COVERAGE,
                description=f"{endpoint} is missing because the extraction failed; re-running may recover it",
                severity=Severity.WARNING,
                status=RuleStatus.FAIL,
                endpoint=endpoint,
                affected_records=len(keys),
                details={"reason": reason, "session_keys": sorted(keys)[:_MAX_EXAMPLES]},
            )
        )

    for endpoint, keys in sorted(absent.items()):
        results.append(
            ValidationResult(
                rule=RULE_COVERAGE,
                description=f"{endpoint} is missing for sessions the extraction never recorded",
                severity=Severity.WARNING,
                status=RuleStatus.FAIL,
                endpoint=endpoint,
                affected_records=len(keys),
                details={"session_keys": sorted(keys)[:_MAX_EXAMPLES]},
            )
        )

    results.extend(_coverage_summary(index, sessions))
    return results


def _coverage_summary(index: DatasetIndex, sessions: Sequence[int]) -> List[ValidationResult]:
    """Report how much data each session holds, without asserting it should match.

    Drivers retire, are disqualified or never start; lap counts differing
    between drivers is normal, so this describes the spread instead of
    demanding uniformity.
    """
    results: List[ValidationResult] = []

    for session_key in sessions:
        lap_counts = [
            count for (session, _driver), count in index.laps_per_driver.items() if session == session_key
        ]
        if not lap_counts:
            continue
        results.append(
            ValidationResult(
                rule=RULE_COVERAGE,
                description="Lap coverage for the session",
                severity=Severity.INFO,
                status=RuleStatus.PASS,
                endpoint="laps",
                details={
                    "session_key": session_key,
                    "drivers_with_laps": len(lap_counts),
                    "laps_min": min(lap_counts),
                    "laps_max": max(lap_counts),
                    "laps_total": sum(lap_counts),
                    "note": "Differences between drivers are expected (retirements, penalties).",
                },
            )
        )

    results.append(
        ValidationResult(
            rule=RULE_COVERAGE,
            description="Records found per endpoint",
            severity=Severity.INFO,
            status=RuleStatus.PASS,
            details={
                "records_by_endpoint": dict(sorted(index.records_by_endpoint.items())),
                "files_by_endpoint": dict(sorted(index.files_by_endpoint.items())),
                "sessions": len(sessions),
            },
        )
    )
    return results


class ManifestLookup:
    """Minimal protocol the coverage rule needs from an extraction manifest."""

    def failure_for(self, endpoint: str, session_key: int) -> Optional[Mapping[str, Any]]:  # pragma: no cover
        raise NotImplementedError
