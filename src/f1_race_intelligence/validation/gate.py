"""The M3 gate: stored validation verdicts that decide whether M4-M6 may run.

This module is the single place the policy lives; the CLI and the M4, M5 and
M6 runners all call it, before they read or write any artifact.

* ``PASS`` and ``WARNING`` are the acceptable verdicts. ``FAIL`` blocks.
* For one session a missing verdict also blocks: M3 never vouched for that raw data.
  Broader runs (a season, every session) keep working without a stored verdict,
  but a non-passing verdict on anything they would touch blocks them.
* A report that exists but cannot be read, or does not have the expected shape,
  is not "absent": it blocks, and because reports are judged newest first it
  cannot be bypassed by falling back to an older PASS.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends.base import key_name
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore

PASSING = frozenset({"PASS", "WARNING"})
KNOWN_VERDICTS = PASSING | {"FAIL"}

_PARTITION = re.compile(r"(?:^|/)(year|session)=(\d+)(?=/|$)")


class ValidationGateBlocked(RuntimeError):
    """M3 does not allow the stage to run on this data."""


def modeling_gate_years(settings: AppSettings) -> Optional[List[int]]:
    """The seasons M6 models: exactly ``modeling.years`` (``None`` = every season)."""
    return list(settings.modeling.years) or None


FILTER_FIELDS = ("year", "meeting_key", "session_key")


def _report_problem(payload: Any) -> Optional[str]:
    """Why a stored report cannot be trusted to say what it covers (``None``: it can).

    The shape the validation runner writes is required in full: ``source.filter`` is
    ``null`` (a full run) or an object with ``year``, ``meeting_key`` and ``session_key``,
    each ``null`` or an integer. Anything else could be silently skipped by a season or
    session comparison, so it is rejected instead.
    """
    if not isinstance(payload, dict):
        return "not a JSON object"
    if payload.get("global_status") not in KNOWN_VERDICTS:
        return f"global_status {payload.get('global_status')!r} is not one of {sorted(KNOWN_VERDICTS)}"
    source = payload.get("source")
    if not isinstance(source, dict):
        return "'source' is missing or not an object"
    if "filter" not in source:
        return "'source.filter' is missing"
    report_filter = source["filter"]
    if report_filter is None:
        return None
    if not isinstance(report_filter, dict):
        return "'source.filter' is not an object"
    for name in FILTER_FIELDS:
        if name not in report_filter:
            return f"'source.filter.{name}' is missing"
        value = report_filter[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            return f"'source.filter.{name}' is {value!r}, not an integer or null"
    return None


def _listings(layout: Any, root: str, scope: SessionScope) -> List[Tuple[str, bool]]:
    """``(prefix, recursive)`` listings that can hold a report relevant to the scope.

    Full-run, season and meeting reports are filed directly under the report root, and a
    session's own reports under its partition, so a session run lists the root without
    descending into other sessions' partitions. Wider scopes may involve any session, so
    they list the whole tree.
    """
    if not scope.is_session or scope.year is None:
        return [(root, True)]
    listings: List[Tuple[str, bool]] = [(root, False)]
    for year in (scope.year, None):
        prefix = layout.records_prefix(root, year=year, session_key=scope.session_key)
        entry = (prefix, prefix != root)
        if entry not in listings and (prefix, not entry[1]) not in listings:
            listings.append(entry)
    return listings


def _stored_reports(
    store: DataStore, settings: AppSettings, scope: SessionScope, report_path: Optional[str]
) -> List[Dict[str, Any]]:
    """Every validation report that could concern the scope, newest first.

    Unreadable or malformed reports are returned with a ``problem`` instead of being dropped.
    """
    backend, layout = store.backend, store.layout
    root = store.root(Layer.VALIDATION_REPORTS, report_path or settings.validation.report_path)
    keys: List[str] = []
    for prefix, recursive in _listings(layout, root, scope):
        for key in backend.list(prefix, recursive=recursive):
            if key not in keys and key_name(key).startswith("validation_report_") and key.endswith(".json"):
                keys.append(key)

    reports = []
    for key in sorted(keys, key=lambda item: (key_name(item), item), reverse=True):
        parts = {name: int(value) for name, value in _PARTITION.findall(key)}
        entry: Dict[str, Any] = {
            "uri": backend.uri(key),
            "year": parts.get("year"),
            "session": parts.get("session"),
            "payload": None,
            "problem": None,
        }
        try:
            payload = backend.read_json(key)
        except Exception as exc:  # corrupt JSON, bad encoding, storage error: never "absent"
            entry["problem"] = f"unreadable: {type(exc).__name__}: {exc}"
        else:
            entry["problem"] = _report_problem(payload)
            if entry["problem"] is None:
                entry["payload"] = payload
        reports.append(entry)
    return reports


def _filter_of(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload.get("source")
    report_filter = source.get("filter") if isinstance(source, dict) else None
    return report_filter if isinstance(report_filter, dict) else {}


def _covers(report_filter: Dict[str, Any], scope: SessionScope) -> bool:
    """Whether a report filtered like this says anything about the scope."""
    for name, wanted in scope.to_dict().items():
        found = report_filter.get(name)
        if wanted is not None and found is not None and found != wanted:
            return False
    return True


def _vouches_for_session(report_filter: Dict[str, Any], scope: SessionScope) -> bool:
    """Whether a valid report says something about this session's raw data.

    A full run does. A season, meeting or session report does when every field it
    names equals the scope's: a field the scope leaves unknown cannot be confirmed,
    so a report that names it does not count.
    """
    return all(report_filter.get(name) is None or report_filter[name] == value for name, value in scope.to_dict().items())


def _invalid_message(entry: Dict[str, Any]) -> str:
    return (
        f"M3 validation report {entry['uri']} is unreadable or invalid ({entry['problem']}); "
        "it is newer than any usable verdict, so no earlier verdict is trusted"
    )


def _year_allowed(year: Optional[int], scope: SessionScope, years: Optional[Sequence[int]]) -> bool:
    if year is None:
        return True
    if scope.year is not None and year != scope.year:
        return False
    return years is None or year in years


def check_validation_gate(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    *,
    years: Optional[Sequence[int]] = None,
    report_path: Optional[str] = None,
) -> Optional[str]:
    """``None`` when M3 allows the stage to run, else the reason it must not.

    Reports are judged newest first. For a session scope the newest report that
    covers the session decides: PASS/WARNING continue, FAIL blocks, an unusable
    report blocks, and no report at all blocks. For wider scopes each report
    group (a full run, a season, one session) is judged by its newest report,
    restricted to ``years`` when given.
    """
    reports = _stored_reports(store, settings, scope, report_path)

    if scope.is_session:
        for entry in reports:
            if entry["problem"] is not None:
                if entry["session"] in (None, scope.session_key) and _year_allowed(entry["year"], scope, None):
                    return _invalid_message(entry)
                continue
            if _vouches_for_session(_filter_of(entry["payload"]), scope):
                status = entry["payload"]["global_status"]
                if status in PASSING:
                    return None
                return f"M3 validation verdict for session {scope.session_key} is {status} ({entry['uri']})"
        return f"no M3 validation report covers session {scope.session_key}; run m3 for it first"

    # One group per exact (year, meeting_key, session_key) triple taken from ``source.filter``:
    # only a newer report of that same scope supersedes an older one, so a PASS for another
    # meeting, session or the whole season never hides a FAIL.
    newest: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    sessions_with_a_valid_report = set()
    for entry in reports:
        if entry["problem"] is not None:
            if not _year_allowed(entry["year"], scope, years):
                continue
            # An unusable report only has its partition, so its meeting is unknown. Outside a
            # session partition nothing can prove which scope it concerned: it always blocks,
            # even when a newer valid report exists. In a session partition a newer valid
            # report of the very same session (keys are unique) settles it.
            if entry["session"] is None:
                newest[("unusable", entry["uri"])] = entry
                continue
            if entry["session"] in sessions_with_a_valid_report:
                continue
            group = (entry["year"], None, entry["session"])
        else:
            report_filter = _filter_of(entry["payload"])
            if not _covers(report_filter, scope) or not _year_allowed(report_filter.get("year"), scope, years):
                continue
            group = tuple(report_filter.get(name) for name in FILTER_FIELDS)
            if group[2] is not None:
                sessions_with_a_valid_report.add(group[2])
        newest.setdefault(group, entry)

    reasons = []
    for entry in newest.values():
        if entry["problem"] is not None:
            reasons.append(_invalid_message(entry))
        elif entry["payload"]["global_status"] not in PASSING:
            reasons.append(f"{entry['payload']['global_status']} verdict in {entry['uri']}")
    if reasons:
        return "M3 validation did not pass for the selected data: " + "; ".join(sorted(reasons))
    return None


def enforce_validation_gate(
    store: DataStore,
    settings: AppSettings,
    scope: SessionScope,
    *,
    years: Optional[Sequence[int]] = None,
    report_path: Optional[str] = None,
) -> None:
    """Raise :class:`ValidationGateBlocked` unless M3 allows the stage to run."""
    reason = check_validation_gate(store, settings, scope, years=years, report_path=report_path)
    if reason is not None:
        raise ValidationGateBlocked(reason)
