"""Minimal abstraction for persisting raw OpenF1 API responses.

It writes one JSON envelope per call under a predictable layout, so later
stages can discover raw data without guessing where it landed. *Where*
that is depends on the :class:`~f1_race_intelligence.storage.store.DataStore`
it is given: by default the historical local tree
(``<base>/<endpoint>/year=.../meeting_key=.../session_key=...``), or the
data lake contract on S3 (see :mod:`f1_race_intelligence.storage.layout`).
Callers that need re-runs to be idempotent pass an explicit ``file_name``
so the same request always maps to the same key (see
:meth:`RawDataStorage.resolve_path`).

Two guarantees hold whichever naming a caller picks, because downstream
stages decide what to re-download by looking at what is stored:

* An object that exists is complete. Local writes go to a temporary file
  that is then moved into place; S3 uploads are atomic. An interrupted run
  leaves no truncated file that a later run would mistake for finished data.
* An auto-generated (timestamped) name never replaces an existing object,
  even when the clock repeats between two calls.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore

logger = logging.getLogger(__name__)

#: A local :class:`~pathlib.Path` on the local backend, an ``s3://`` URI otherwise.
StorageLocation = Union[Path, str]


class RawDataStorage:
    """Saves raw OpenF1 responses as JSON envelopes, organized by request context."""

    def __init__(self, base_path: Union[str, Path] = "data/raw", *, store: Optional[DataStore] = None) -> None:
        """Create a storage.

        Args:
            base_path: Root of the raw tree for the legacy layout. The lake
                layout has a fixed ``raw/`` root and ignores it.
            store: Backend and layout to write through. Defaults to local
                files in the historical layout.
        """
        self._store = store or DataStore.local()
        self._root = self._store.root(Layer.RAW, str(base_path))

    @property
    def backend(self) -> StorageBackend:
        return self._store.backend

    @property
    def store(self) -> DataStore:
        return self._store

    def save(
        self,
        endpoint: str,
        parameters: Optional[Mapping[str, Any]] = None,
        data: Any = None,
        *,
        file_name: Optional[str] = None,
    ) -> StorageLocation:
        """Persist one API response.

        Args:
            endpoint: Logical endpoint name, e.g. ``"sessions"`` or ``"/laps"``.
            parameters: The query parameters used for the request. When they
                include ``year``, ``meeting_key`` and/or ``session_key``, the
                file is nested under matching partitions.
            data: The (already parsed) response payload to persist.
            file_name: An explicit, deterministic file name (e.g.
                ``"session_9158.json"``). Callers that need to recognize
                already-downloaded data on a later run — such as the
                historical extraction pipeline — must pass this, and accept
                that saving again replaces the file. When omitted, the file
                is named after the write timestamp and an existing file is
                never replaced.

        Returns:
            Where the file was written: a local path, or a URI on object storage.
        """
        parameters = parameters or {}
        directory = self._store.layout.raw_directory(self._root, endpoint, parameters)

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
            key = self.backend.write_json_unique(directory, f"{stamp}Z", payload)
        else:
            # An explicit name is meant to be re-writable — that is what
            # makes an overwriting re-run possible — but never half-written,
            # because a later run reads "the file exists" as "already
            # downloaded" and would skip a truncated file forever.
            key = self.key_for(endpoint, parameters, file_name=file_name)
            self.backend.write_json(key, payload)

        logger.info("raw_data_saved", extra={"endpoint": endpoint, "path": self.backend.uri(key)})
        return self.backend.location(key)

    def key_for(self, endpoint: str, parameters: Optional[Mapping[str, Any]] = None, *, file_name: str) -> str:
        """The storage key :meth:`save` would write this file under."""
        return self._store.layout.raw_key(
            self._root, endpoint, parameters or {}, self._validate_file_name(file_name)
        )

    def resolve_path(
        self,
        endpoint: str,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        file_name: str,
    ) -> StorageLocation:
        """Return where :meth:`save` would write this file, without writing it.

        This is what makes idempotent re-runs possible: a caller can check
        whether the data it is about to request already exists and skip the
        HTTP call entirely, instead of spending rate-limit budget on
        something it would only overwrite.
        """
        return self.backend.location(self.key_for(endpoint, parameters, file_name=file_name))

    def uri(self, endpoint: str, parameters: Optional[Mapping[str, Any]] = None, *, file_name: str) -> str:
        """The location of a file as manifests and logs record it."""
        return self.backend.uri(self.key_for(endpoint, parameters, file_name=file_name))

    def exists(self, endpoint: str, parameters: Optional[Mapping[str, Any]] = None, *, file_name: str) -> bool:
        """Whether this file has already been saved."""
        return self.backend.exists(self.key_for(endpoint, parameters, file_name=file_name))

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
        payload = self.backend.read_json(self.key_for(endpoint, parameters, file_name=file_name))
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
