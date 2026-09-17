"""Local filesystem storage — the behaviour the project has always had.

A key is a path relative to ``root`` (the working directory by default), so
the configured stage paths — ``data/raw``, ``data/processed/lap_dataset`` —
resolve exactly where they always did. An absolute key stays absolute,
which is what lets tests point a stage at a temporary directory.

JSON goes through :mod:`f1_race_intelligence.utils.files`, the module that
already guarantees atomic replacement and never-overwriting unique names,
so there is still a single implementation of those rules on disk.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Union

from f1_race_intelligence.storage.backends.base import StorageBackend, join_key
from f1_race_intelligence.utils import files


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
        try:
            # Exclusive creation lets the filesystem arbitrate between writers.
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            return False
        return True

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

    # -- formats: keep the existing single implementation on disk ------------

    def write_json(self, key: str, payload: Any) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        files.write_json_atomically(path, payload)
        return key

    def write_json_unique(self, prefix: str, stem: str, payload: Any, *, extension: str = ".json") -> str:
        written = files.write_json_unique(self._path(prefix), stem, payload, extension=extension)
        return join_key(prefix, written.name)

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
