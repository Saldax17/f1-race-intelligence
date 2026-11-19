"""Deprecated compatibility helpers for legacy local JSON callers.

New pipeline code must use :class:`StorageBackend`: it serialises JSON once
to canonical UTF-8 bytes and writes through a storage backend.  These names
remain importable for callers outside this repository that used the original
module.  They delegate to that implementation rather than keeping a second
JSON serialiser or a second set of local write primitives.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, TextIO, Union

from f1_race_intelligence.storage.backends.base import MAX_NAME_COLLISION_RETRIES, _json_bytes
from f1_race_intelligence.storage.backends.local import LocalStorageBackend


def _deprecated(name: str) -> None:
    warnings.warn(
        f"f1_race_intelligence.utils.files.{name} is deprecated; use StorageBackend instead",
        DeprecationWarning,
        stacklevel=3,
    )


def dump_json(payload: Any, handle: TextIO) -> None:
    """Write the canonical JSON text to a legacy text handle.

    This preserves the historical signature only.  A text handle controlled
    by a caller may translate newlines on Windows, so it must not be used for
    persisted project artifacts.  StorageBackend writes canonical bytes.
    """
    _deprecated("dump_json")
    handle.write(_json_bytes(payload).decode("utf-8"))


def write_json_unique(
    directory: Union[str, Path],
    stem: str,
    payload: Any,
    *,
    extension: str = ".json",
) -> Path:
    """Create a JSON file exclusively, retaining the former return type."""
    _deprecated("write_json_unique")
    target = Path(directory)
    backend = LocalStorageBackend(target)
    key = backend.write_json_unique("", stem, payload, extension=extension)
    return target / Path(key).name


def write_json_atomically(path: Union[str, Path], payload: Any) -> Path:
    """Replace a JSON file atomically through the canonical local backend."""
    _deprecated("write_json_atomically")
    file_path = Path(path)
    LocalStorageBackend(file_path.parent).write_json(file_path.name, payload)
    return file_path
