"""Local filesystem storage — the behaviour the project has always had.

A key is a path relative to ``root`` (the working directory by default), so
the configured stage paths — ``data/raw``, ``data/processed/lap_dataset`` —
resolve exactly where they always did. An absolute key stays absolute,
which is what lets tests point a stage at a temporary directory.

Every write is binary. JSON is serialised by the base class and reaches the
disk through :meth:`LocalStorageBackend.write_bytes` (temporary file, then an
atomic replace) or :meth:`LocalStorageBackend.write_bytes_if_absent` (atomic
publication without replacement). A text-mode write would let the operating
system translate line endings — ``\\n`` becomes ``\\r\\n`` on Windows — so the
same document would be different bytes, and a different SHA-256, depending on
where it was written.
"""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from f1_race_intelligence.storage.backends.base import MAX_NAME_COLLISION_RETRIES, StorageBackend, join_key


class LocalStorageBackend(StorageBackend):
    """Stores objects as files below a root directory."""

    name = "local"

    def __init__(self, root: Union[str, Path] = ".") -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        return self._root / key if key else self._root

    # -- primitives ---------------------------------------------------------

    def uri(self, key: str) -> str:
        return str(self._path(key))

    def local_path(self, key: str) -> Optional[Path]:
        return self._path(key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def has_prefix(self, prefix: str) -> bool:
        return self._path(prefix).is_dir()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".tmp-", suffix=path.suffix, delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
            os.replace(temp_path, path)
        except BaseException:
            # Never leave a stray .tmp- file behind for the next stage to find.
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        return key

    def write_bytes_if_absent(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> bool:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            for _ in range(MAX_NAME_COLLISION_RETRIES):
                candidate = path.with_name(f".tmp-{secrets.token_hex(16)}{path.suffix}")
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
                try:
                    descriptor = os.open(candidate, flags, 0o666)
                except FileExistsError:
                    continue
                temp_path = candidate
                try:
                    handle = os.fdopen(descriptor, "wb")
                except BaseException:
                    os.close(descriptor)
                    raise
                break
            else:
                raise FileExistsError(f"Could not create a temporary file in {path.parent}")

            with handle:
                written = handle.write(data)
                if written != len(data):
                    raise OSError(f"Short write for temporary file {temp_path}")
            try:
                # A hard link publishes the complete file only if the name is free.
                os.link(temp_path, path)
            except FileExistsError:
                return False
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def list(self, prefix: str, *, recursive: bool = True) -> List[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        candidates = base.rglob("*") if recursive else base.glob("*")
        keys = [
            join_key(prefix, path.relative_to(base).as_posix())
            for path in candidates
            if path.is_file() and not path.name.startswith(".")
        ]
        return sorted(keys)

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    # -- formats ------------------------------------------------------------
    #
    # JSON is deliberately not overridden: the base class serialises once and
    # hands the bytes to write_bytes / write_bytes_if_absent above.

    def read_parquet(self, key: str):
        import pandas as pd

        # A file handle, not a path: see StorageBackend.read_parquet.
        with self._path(key).open("rb") as handle:
            return pd.read_parquet(handle, engine="pyarrow")

    def write_parquet(self, key: str, frame) -> int:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".tmp-", suffix=".parquet", delete=False) as handle:
                temp_path = Path(handle.name)
                frame.to_parquet(handle, engine="pyarrow", index=False)
            os.replace(temp_path, path)
        except BaseException:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        return path.stat().st_size
