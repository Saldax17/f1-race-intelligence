"""Minimal abstraction for persisting raw OpenF1 API responses.

This is intentionally simple: it just writes one JSON file per call under
a predictable directory layout (``<base>/<endpoint>/year=.../meeting_key=...
/session_key=...``) so a future ingestion pipeline stage can discover raw
data without guessing where it landed. Callers that need re-runs to be
idempotent pass an explicit ``file_name`` so the same request always maps
to the same path (see :meth:`RawDataStorage.resolve_path`). It is *not* yet the definitive
storage strategy (partitioning, compression, a data lake format, etc. are
future work) — just enough to unblock saving what :class:`F1Client`
returns today.

Two guarantees hold whichever naming a caller picks, because downstream
stages decide what to re-download by looking at what is on disk:

* A file that exists is complete. Writes go to a temporary file that is
  then moved into place, so an interrupted run leaves no truncated file
  that a later run would mistake for finished data.
* An auto-generated (timestamped) name never replaces an existing file,
  even when the clock repeats between two calls.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from f1_race_intelligence.utils.files import write_json_atomically, write_json_unique

logger = logging.getLogger(__name__)


class RawDataStorage:
    """Saves raw OpenF1 responses to disk as JSON, organized by request context."""

    def __init__(self, base_path: Union[str, Path] = "data/raw") -> None:
        self._base_path = Path(base_path)

    def save(
        self,
        endpoint: str,
        parameters: Optional[Mapping[str, Any]] = None,
        data: Any = None,
        *,
        file_name: Optional[str] = None,
    ) -> Path:
        """Persist one API response.

        Args:
            endpoint: Logical endpoint name, e.g. ``"sessions"`` or ``"/laps"``.
            parameters: The query parameters used for the request. When they
                include ``year``, ``meeting_key`` and/or ``session_key``, the
                file is nested under matching subdirectories.
            data: The (already parsed) response payload to persist.
            file_name: An explicit, deterministic file name (e.g.
                ``"session_9158.json"``). Callers that need to recognize
                already-downloaded data on a later run — such as the
                historical extraction pipeline — must pass this, and accept
                that saving again replaces the file. When omitted, the file
                is named after the write timestamp and an existing file is
                never replaced.

        Returns:
            The path of the file that was written.
        """
        parameters = parameters or {}
        directory = self._resolve_directory(endpoint, parameters)
        directory.mkdir(parents=True, exist_ok=True)

        retrieved_at = datetime.now(timezone.utc)
        payload = {
            "endpoint": endpoint,
            "parameters": dict(parameters),
            "retrieved_at": retrieved_at.isoformat(),
            "data": data,
        }

        if file_name is None:
            # An auto-named file must never replace one already there: the
            # clock can repeat between two calls.
            stamp = retrieved_at.strftime("%Y%m%dT%H%M%S%f")
            file_path = write_json_unique(directory, f"{stamp}Z", payload)
        else:
            # An explicit name is meant to be re-writable — that is what
            # makes an overwriting re-run possible — but never half-written,
            # because a later run reads "the file exists" as "already
            # downloaded" and would skip a truncated file forever.
            file_path = write_json_atomically(directory / self._validate_file_name(file_name), payload)

        logger.info("raw_data_saved", extra={"endpoint": endpoint, "path": str(file_path)})
        return file_path

    def resolve_path(
        self,
        endpoint: str,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        file_name: str,
    ) -> Path:
        """Return where :meth:`save` would write this file, without writing it.

        This is what makes idempotent re-runs possible: a caller can check
        whether the data it is about to request already exists on disk and
        skip the HTTP call entirely, instead of spending rate-limit budget
        on something it would only overwrite.
        """
        directory = self._resolve_directory(endpoint, parameters or {})
        return directory / self._validate_file_name(file_name)

    def load(
        self,
        endpoint: str,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        file_name: str,
    ) -> Any:
        """Read back the ``data`` payload of a previously saved raw file.

        Raises:
            OSError: if the file cannot be read.
            ValueError: if the file is not valid JSON.
        """
        path = self.resolve_path(endpoint, parameters, file_name=file_name)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("data")

    @staticmethod
    def _validate_file_name(file_name: str) -> str:
        """Reject names that would escape the resolved directory.

        File names are built from API-provided values (driver numbers,
        session keys), so this guards against a malformed value turning
        into a path traversal.
        """
        if not file_name or file_name in {".", ".."} or any(sep in file_name for sep in ("/", "\\", os.sep)):
            raise ValueError(f"Invalid raw data file name: {file_name!r}")
        return file_name

    def _resolve_directory(self, endpoint: str, parameters: Mapping[str, Any]) -> Path:
        parts = [endpoint.strip("/")]
        for key in ("year", "meeting_key", "session_key"):
            if key in parameters and parameters[key] is not None:
                parts.append(f"{key}={parameters[key]}")
        return self._base_path.joinpath(*parts)
