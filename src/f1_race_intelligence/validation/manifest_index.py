"""Reading the extraction manifest, from the validation side.

The raw tree alone cannot explain an absence: a file that is not there
might mean the source has no such data, or that a request failed, or that
nobody ever asked for it. The manifest M2 writes knows which, and this
turns it into the two questions validation actually asks — what was
expected, and why is something missing.

Manifests written before ``status_code`` existed still load; a failure
without a status code is simply reported as one whose cause is unknown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


class ManifestIndex:
    """Lookup over one extraction manifest."""

    def __init__(self, payload: Mapping[str, Any], path: Optional[Path] = None) -> None:
        self._payload = payload
        self._path = path
        self._failures: Dict[Tuple[str, Any], Mapping[str, Any]] = {}

        for entry in payload.get("entries", []):
            if not isinstance(entry, Mapping) or entry.get("status") != "failed":
                continue
            key = (entry.get("endpoint"), entry.get("session_key"))
            # Keep the first failure per endpoint/session: car_data fans out
            # into one entry per driver and they share the same cause.
            self._failures.setdefault(key, entry)

    @classmethod
    def load_latest(cls, directory: Union[str, Path]) -> Optional["ManifestIndex"]:
        """Load the most recent manifest in a directory, if there is one.

        Manifest file names embed the execution timestamp, so the newest is
        the last in sorted order — no file timestamps involved, which keeps
        this reproducible when files are copied around.
        """
        base = Path(directory)
        if not base.is_dir():
            return None

        candidates = sorted(base.glob("manifest_*.json"))
        for path in reversed(candidates):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping):
                return cls(payload, path=path)
        return None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def config(self) -> Mapping[str, Any]:
        config = self._payload.get("config")
        return config if isinstance(config, Mapping) else {}

    @property
    def expected_endpoints(self) -> List[str]:
        endpoints = self.config.get("endpoints")
        return [str(item) for item in endpoints] if isinstance(endpoints, list) else []

    @property
    def sessions_processed(self) -> List[int]:
        sessions = self._payload.get("sessions_processed")
        return [item for item in sessions if isinstance(item, int)] if isinstance(sessions, list) else []

    def failure_for(self, endpoint: str, session_key: int) -> Optional[Mapping[str, Any]]:
        """The recorded failure for one endpoint of one session, if any."""
        return self._failures.get((endpoint, session_key))

    def describe(self) -> Dict[str, Any]:
        """What went into the report's ``source`` block."""
        return {
            "path": str(self._path) if self._path else None,
            "pipeline_version": self._payload.get("pipeline_version"),
            "started_at": self._payload.get("started_at"),
            "summary": self._payload.get("summary"),
            "configured_endpoints": self.expected_endpoints,
        }
