"""Reading the extraction manifest, from the validation side.

The raw tree alone cannot explain an absence: a file that is not there
might mean the source has no such data, or that a request failed, or that
nobody ever asked for it. The manifest M2 writes knows which, and this
turns it into the two questions validation actually asks — what was
expected, and why is something missing.

Manifests written before ``status_code`` existed still load; a failure
without a status code is simply reported as one whose cause is unknown.

M2 writes two kinds of manifest: one per full run, and one per session job
(recognisable by ``config.scope``). :meth:`ManifestIndex.load_latest` picks
the one that describes what is being validated — see its docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from f1_race_intelligence.storage.backends.base import StorageBackend, key_name
from f1_race_intelligence.storage.backends.local import LocalStorageBackend


class ManifestIndex:
    """Lookup over one extraction manifest."""

    def __init__(self, payload: Mapping[str, Any], path: Optional[Union[Path, str]] = None) -> None:
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
    def load_latest(
        cls,
        directory: Union[str, Path],
        *,
        backend: Optional[StorageBackend] = None,
        session_key: Optional[int] = None,
        search_prefixes: Optional[Sequence[str]] = None,
    ) -> Optional["ManifestIndex"]:
        """Load the manifest that best describes the data being validated.

        Manifest file names embed the execution timestamp, so the newest is
        the last in name order — no file timestamps involved, which keeps
        this reproducible when files are copied around.

        * With ``session_key``: the newest manifest that processed that
          session, whether a full run or a session job.
        * Without it: the newest full-run manifest, exactly as before
          session jobs existed. When only session jobs ran (the normal case
          in the cloud), their newest manifest per session are combined into
          one index, so coverage is judged against everything that was
          extracted rather than against whichever job finished last.

        Args:
            directory: Prefix holding the manifests.
            backend: Storage to read; local files when omitted.
            session_key: Validate one session only.
            search_prefixes: Prefixes to search first, in order (a session's
                own prefix), before ``directory``.
        """
        backend = backend or LocalStorageBackend()
        prefixes = [*(search_prefixes or []), str(directory)]

        if session_key is not None:
            for key, payload in _newest_first(backend, prefixes):
                if session_key in _sessions_processed(payload):
                    return cls(payload, path=backend.location(key))
            return None

        session_jobs: Dict[int, Tuple[str, Mapping[str, Any]]] = {}
        for key, payload in _newest_first(backend, prefixes):
            scope = _scope(payload)
            if scope is None:
                return cls(payload, path=backend.location(key))
            if scope.get("mode") == "session" and isinstance(scope.get("session_key"), int):
                session_jobs.setdefault(scope["session_key"], (key, payload))

        if not session_jobs:
            return None
        return cls.combine([payload for _, payload in session_jobs.values()], source=backend.uri(str(directory)))

    @classmethod
    def combine(cls, payloads: Sequence[Mapping[str, Any]], *, source: Optional[str] = None) -> "ManifestIndex":
        """One index over several session-job manifests."""
        endpoints: List[str] = []
        sessions: List[int] = []
        entries: List[Any] = []
        for payload in payloads:
            config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
            for endpoint in config.get("endpoints") or []:
                if endpoint not in endpoints:
                    endpoints.append(endpoint)
            for session in _sessions_processed(payload):
                if session not in sessions:
                    sessions.append(session)
            entries.extend(payload.get("entries") or [])

        combined = {
            "pipeline_version": sorted({str(p.get("pipeline_version")) for p in payloads}),
            "started_at": min((str(p.get("started_at")) for p in payloads), default=None),
            "summary": {"combined_manifests": len(payloads)},
            "config": {"endpoints": endpoints, "scope": {"mode": "combined"}},
            "sessions_processed": sorted(sessions),
            "entries": entries,
        }
        return cls(combined, path=source)

    @property
    def path(self) -> Optional[Union[Path, str]]:
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
        return _sessions_processed(self._payload)

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


def _newest_first(backend: StorageBackend, prefixes: Sequence[str]) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """Readable manifests under the prefixes, newest first within each prefix."""
    seen = set()
    for prefix in prefixes:
        keys = [key for key in backend.list(prefix) if _is_manifest(key) and key not in seen]
        seen.update(keys)
        for key in sorted(keys, key=lambda item: (key_name(item), item), reverse=True):
            try:
                payload = backend.read_json(key)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, Mapping):
                yield key, payload


def _is_manifest(key: str) -> bool:
    name = key_name(key)
    return name.startswith("manifest_") and name.endswith(".json")


def _scope(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    config = payload.get("config")
    scope = config.get("scope") if isinstance(config, Mapping) else None
    return scope if isinstance(scope, Mapping) else None


def _sessions_processed(payload: Mapping[str, Any]) -> List[int]:
    sessions = payload.get("sessions_processed")
    return [item for item in sessions if isinstance(item, int)] if isinstance(sessions, list) else []
