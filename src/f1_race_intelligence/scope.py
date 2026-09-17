"""The unit of work a batch job runs on.

Every stage can run over everything it finds, as it always has, or over one
session. A session is the natural unit for cloud execution: its raw data,
consolidated laps and features are independent of every other session, so
sessions can be processed in parallel and retried one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SessionScope:
    """Which session (or season) a run is restricted to. ``None`` means unrestricted."""

    year: Optional[int] = None
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None

    @property
    def is_session(self) -> bool:
        return self.session_key is not None

    @property
    def is_empty(self) -> bool:
        return self.year is None and self.meeting_key is None and self.session_key is None

    def to_dict(self) -> Dict[str, Any]:
        return {"year": self.year, "meeting_key": self.meeting_key, "session_key": self.session_key}

    def log_fields(self) -> Dict[str, Any]:
        """The keys worth putting on every log line of a scoped run."""
        return {key: value for key, value in self.to_dict().items() if value is not None}

    def matches(self, *, year: Optional[int] = None, meeting_key: Optional[int] = None, session_key: Optional[int] = None) -> bool:
        """Whether data identified by these keys falls inside the scope."""
        for wanted, actual in ((self.year, year), (self.meeting_key, meeting_key), (self.session_key, session_key)):
            if wanted is not None and actual != wanted:
                return False
        return True
