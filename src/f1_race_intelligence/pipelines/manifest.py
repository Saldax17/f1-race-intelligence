"""Execution manifest for pipeline runs.

A manifest answers one question after the fact: *what exactly did this run
download?* It records one entry per raw file the pipeline decided about —
downloaded, skipped because it already existed, or failed — plus the
configuration that produced those decisions.

The manifest is an audit record of an execution, not a dataset, so unlike
raw data files it is deliberately timestamped: two runs of the same
configuration produce two manifests, and neither overwrites the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from f1_race_intelligence.utils.files import write_json_unique


class ExtractionStatus(str, Enum):
    """What happened to a single raw file during a run."""

    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class ManifestEntry:
    """One raw file the pipeline handled (or tried to)."""

    endpoint: str
    status: ExtractionStatus
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None
    driver_number: Optional[int] = None
    file_path: Optional[str] = None
    record_count: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    status_code: Optional[int] = None
    """HTTP status of a failed request, when the request got a response at all.

    OpenF1 answers ``404 {"detail": "No results found."}`` when an endpoint
    simply holds no data for a session — pit stops for the whole 2023
    season, for example — which is a gap in the source, not a broken
    extraction. Recording the code separately lets a later stage tell that
    apart from a ``5xx`` without parsing ``error`` as free text. It stays
    ``None`` for failures that never received a response (timeouts,
    connection errors).
    """

    def to_dict(self) -> Dict[str, Any]:
        """Serialize, dropping keys that don't apply to this entry."""
        payload: Dict[str, Any] = {"endpoint": self.endpoint, "status": self.status.value}
        optional = {
            "year": self.year,
            "meeting_key": self.meeting_key,
            "session_key": self.session_key,
            "driver_number": self.driver_number,
            "file_path": self.file_path,
            "record_count": self.record_count,
            "error": self.error,
            "error_type": self.error_type,
            "status_code": self.status_code,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass
class ExecutionManifest:
    """Everything one pipeline execution did, ready to be written to disk."""

    pipeline_name: str
    pipeline_version: str
    config: Dict[str, Any]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    entries: List[ManifestEntry] = field(default_factory=list)
    meetings_processed: List[int] = field(default_factory=list)
    sessions_processed: List[int] = field(default_factory=list)

    def add(self, entry: ManifestEntry) -> ManifestEntry:
        """Record one file outcome and return it."""
        self.entries.append(entry)
        return entry

    def record_meeting(self, meeting_key: Optional[int]) -> None:
        if meeting_key is not None and meeting_key not in self.meetings_processed:
            self.meetings_processed.append(meeting_key)

    def record_session(self, session_key: Optional[int]) -> None:
        if session_key is not None and session_key not in self.sessions_processed:
            self.sessions_processed.append(session_key)

    @property
    def errors(self) -> List[ManifestEntry]:
        return [entry for entry in self.entries if entry.status is ExtractionStatus.FAILED]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 3)

    def summary(self) -> Dict[str, int]:
        """Counts per status, plus how much of the season tree was touched."""
        counts = {status.value: 0 for status in ExtractionStatus}
        for entry in self.entries:
            counts[entry.status.value] += 1
        counts["meetings"] = len(self.meetings_processed)
        counts["sessions"] = len(self.sessions_processed)
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "config": self.config,
            "summary": self.summary(),
            "meetings_processed": self.meetings_processed,
            "sessions_processed": self.sessions_processed,
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": [entry.to_dict() for entry in self.errors],
        }

    def save(self, directory: Union[str, Path]) -> Path:
        """Write the manifest as JSON and return its path.

        Two runs that start within the same clock tick get distinct files:
        an execution record that quietly replaced an earlier one would lose
        the very history this exists to keep.
        """
        stem = f"manifest_{self.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
        return write_json_unique(directory, stem, self.to_dict())
