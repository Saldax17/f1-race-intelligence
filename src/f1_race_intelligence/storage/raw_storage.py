"""Minimal abstraction for persisting raw OpenF1 API responses.

This is intentionally simple: it just writes one JSON file per call under
a predictable directory layout (``<base>/<endpoint>/year=.../meeting_key=...
/session_key=...``) so a future ingestion pipeline stage can discover raw
data without guessing where it landed. It is *not* yet the definitive
storage strategy (partitioning, compression, a data lake format, etc. are
future work) — just enough to unblock saving what :class:`F1Client`
returns today.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

logger = logging.getLogger(__name__)


class RawDataStorage:
    """Saves raw OpenF1 responses to disk as JSON, organized by request context."""

    def __init__(self, base_path: Union[str, Path] = "data/raw") -> None:
        self._base_path = Path(base_path)

    def save(self, endpoint: str, parameters: Optional[Mapping[str, Any]] = None, data: Any = None) -> Path:
        """Persist one API response.

        Args:
            endpoint: Logical endpoint name, e.g. ``"sessions"`` or ``"/laps"``.
            parameters: The query parameters used for the request. When they
                include ``year``, ``meeting_key`` and/or ``session_key``, the
                file is nested under matching subdirectories.
            data: The (already parsed) response payload to persist.

        Returns:
            The path of the file that was written.
        """
        parameters = parameters or {}
        directory = self._resolve_directory(endpoint, parameters)
        directory.mkdir(parents=True, exist_ok=True)

        retrieved_at = datetime.now(timezone.utc)
        file_path = directory / f"{retrieved_at.strftime('%Y%m%dT%H%M%S%f')}Z.json"

        payload = {
            "endpoint": endpoint,
            "parameters": dict(parameters),
            "retrieved_at": retrieved_at.isoformat(),
            "data": data,
        }

        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

        logger.info("raw_data_saved", extra={"endpoint": endpoint, "path": str(file_path)})
        return file_path

    def _resolve_directory(self, endpoint: str, parameters: Mapping[str, Any]) -> Path:
        parts = [endpoint.strip("/")]
        for key in ("year", "meeting_key", "session_key"):
            if key in parameters and parameters[key] is not None:
                parts.append(f"{key}={parameters[key]}")
        return self._base_path.joinpath(*parts)
