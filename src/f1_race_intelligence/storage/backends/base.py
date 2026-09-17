"""The storage contract every pipeline stage depends on.

Stages address their data by *key*: a ``/``-separated logical path such as
``raw/year=2023/meeting=1141/session=7953/endpoint=laps/session_7953.json``.
A backend decides what a key physically means — a file under a directory
on disk, or an object under a prefix in a bucket — so the same stage code
runs against either without knowing which.

Two guarantees the rest of the project relies on hold for every backend:

* A key that exists is complete. A write either replaces the whole object
  or leaves the previous one in place; "it exists" can safely be read as
  "that work is done".
* :meth:`StorageBackend.write_json_unique` never replaces an existing
  object, even when two writers pick the same name at the same instant.
  Execution records (manifests, reports) depend on it.

Missing keys raise :class:`FileNotFoundError` and any other storage failure
raises :class:`StorageError`. Both are :class:`OSError`, so code written
against the local filesystem ("``except OSError``: treat as unreadable")
keeps its meaning on object storage.
"""

from __future__ import annotations

import io
import json
import posixpath
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Only reached when many writes share a clock tick; bounded so a bug cannot
# spin forever. Same bound as utils.files.
MAX_NAME_COLLISION_RETRIES = 100

JSON_CONTENT_TYPE = "application/json"
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
BINARY_CONTENT_TYPE = "application/octet-stream"


class StorageError(OSError):
    """A storage operation failed for a reason other than a missing key."""


def join_key(*parts: Any) -> str:
    """Join key segments with ``/``, ignoring empty ones.

    Backslashes are normalised so a Windows path taken from configuration
    and a POSIX key describe the same location. A leading ``/`` on the first
    segment is kept: on the local backend it is an absolute path.
    """
    items = [str(part).replace("\\", "/") for part in parts if part is not None and str(part) != ""]
    if not items:
        return ""

    head = items[0]
    rooted = head.startswith("/")
    segments = [head.strip("/")] + [item.strip("/") for item in items[1:]]
    joined = "/".join(segment for segment in segments if segment)
    return f"/{joined}" if rooted else joined


def key_name(key: str) -> str:
    """The last segment of a key — its "file name"."""
    return posixpath.basename(key.replace("\\", "/").rstrip("/"))


def relative_parts(root: str, key: str) -> Optional[List[str]]:
    """Segments of ``key`` below ``root``, or ``None`` when it is not below it."""
    base = join_key(root)
    target = join_key(key)
    if not base:
        return [part for part in target.split("/") if part]
    if not target.startswith(base + "/"):
        return None
    return [part for part in target[len(base) + 1 :].split("/") if part]


class StorageBackend(ABC):
    """Where the pipeline's bytes live."""

    #: Short identifier recorded in reports ("local", "s3").
    name: str = "abstract"

    # -- primitives every backend implements -------------------------------

    @abstractmethod
    def uri(self, key: str) -> str:
        """A human-readable, unambiguous location for ``key``.

        A filesystem path for local storage, ``s3://bucket/key`` for S3. It
        is what manifests and reports record, so a reader can find the data.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether an object is stored under exactly this key."""

    @abstractmethod
    def has_prefix(self, prefix: str) -> bool:
        """Whether anything at all is stored below ``prefix``."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read one object. Raises :class:`FileNotFoundError` when absent."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> str:
        """Store ``data`` under ``key`` in one step, replacing any previous object."""

    @abstractmethod
    def write_bytes_if_absent(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> bool:
        """Store ``data`` only when nothing exists under ``key``.

        Returns:
            ``True`` when this call created the object, ``False`` when the
            key was already taken. The check and the write are one atomic
            operation, so two concurrent writers can never both win.
        """

    @abstractmethod
    def list(self, prefix: str, *, recursive: bool = True) -> List[str]:
        """Keys stored below ``prefix``, sorted.

        Hidden objects (a name starting with ``.``, such as the temporary
        file of an interrupted local write) are never listed.
        """

    @abstractmethod
    def size(self, key: str) -> int:
        """Size of one object in bytes."""

    def local_path(self, key: str) -> Optional[Path]:
        """The filesystem path behind ``key``, when there is one."""
        return None

    def location(self, key: str) -> Any:
        """What a caller that predates storage backends expects back.

        A :class:`~pathlib.Path` on local storage — the type every stage
        returned before backends existed — and the URI string otherwise.
        """
        path = self.local_path(key)
        return path if path is not None else self.uri(key)

    def describe(self) -> dict:
        """What reports record about where a run read or wrote."""
        return {"backend": self.name, "location": self.uri("")}

    # -- formats, built on the primitives ----------------------------------

    def read_json(self, key: str) -> Any:
        """Parse one JSON object. Raises ``ValueError`` on invalid content."""
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def write_json(self, key: str, payload: Any) -> str:
        """Store a JSON document atomically, in the project's single JSON style."""
        self.write_bytes(key, _json_bytes(payload), content_type=JSON_CONTENT_TYPE)
        return key

    def write_json_unique(self, prefix: str, stem: str, payload: Any, *, extension: str = ".json") -> str:
        """Store ``payload`` as ``<stem><extension>``, adding ``-1``, ``-2``… rather than replacing.

        Returns:
            The key actually written.
        """
        data = _json_bytes(payload)
        for attempt in range(MAX_NAME_COLLISION_RETRIES):
            suffix = "" if attempt == 0 else f"-{attempt}"
            key = join_key(prefix, f"{stem}{suffix}{extension}")
            if self.write_bytes_if_absent(key, data, content_type=JSON_CONTENT_TYPE):
                return key
        raise StorageError(
            f"Could not find a free name for {stem}{extension} in {self.uri(prefix)} "
            f"after {MAX_NAME_COLLISION_RETRIES} attempts"
        )

    def write_text(self, key: str, text: str, *, content_type: str = "text/plain") -> str:
        self.write_bytes(key, text.encode("utf-8"), content_type=content_type)
        return key

    def read_parquet(self, key: str) -> "pd.DataFrame":
        """Read one Parquet object.

        Always from a buffer, never from a path: pyarrow infers Hive
        partitions from ``key=value`` directory names when given a path,
        and would then invent columns (``year``) that collide with the ones
        stored in the file.
        """
        import pandas as pd

        return pd.read_parquet(io.BytesIO(self.read_bytes(key)), engine="pyarrow")

    def write_parquet(self, key: str, frame: "pd.DataFrame") -> int:
        """Store a DataFrame as Parquet, without its index. Returns the bytes written."""
        buffer = io.BytesIO()
        frame.to_parquet(buffer, engine="pyarrow", index=False)
        data = buffer.getvalue()
        self.write_bytes(key, data, content_type=PARQUET_CONTENT_TYPE)
        return len(data)

    def write_joblib(self, key: str, value: Any) -> int:
        """Persist a Python object with joblib. Returns the bytes written."""
        import joblib

        buffer = io.BytesIO()
        joblib.dump(value, buffer)
        data = buffer.getvalue()
        self.write_bytes(key, data, content_type=BINARY_CONTENT_TYPE)
        return len(data)

    def read_joblib(self, key: str) -> Any:
        """Load an object written by :meth:`write_joblib`. Only for trusted storage."""
        import joblib

        return joblib.load(io.BytesIO(self.read_bytes(key)))


def _json_bytes(payload: Any) -> bytes:
    # Same style as utils.files.dump_json, which the local backend uses.
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
