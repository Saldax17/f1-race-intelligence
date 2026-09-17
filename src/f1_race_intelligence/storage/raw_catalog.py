"""Reading side of the raw data layout.

:class:`~f1_race_intelligence.storage.raw_storage.RawDataStorage` decides
where a response is written; this module is the inverse — it lists that
tree and hands back what is there. Both live in ``storage`` so the path
contract (see :mod:`f1_race_intelligence.storage.layout`) and the
``{endpoint, parameters, retrieved_at, data}`` envelope are described in
exactly one package. A consumer (validation, consolidation) should never
have to parse a key or an envelope itself.

Nothing here writes: raw data is immutable once extracted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from f1_race_intelligence.storage.backends.base import key_name
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore


class RawFileError(Exception):
    """Raised when a raw file cannot be read or is not shaped like an envelope.

    Callers are expected to catch this and report it, rather than let one
    damaged file end a whole pass over the tree.
    """


@dataclass(frozen=True)
class RawFile:
    """A raw JSON file in storage, described by where it sits in the tree."""

    key: str
    endpoint: str
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None
    driver_number: Optional[int] = None
    uri: str = ""
    """Where the file is, as reports record it: a filesystem path or ``s3://…``."""

    @property
    def name(self) -> str:
        return key_name(self.key)

    @property
    def path(self) -> PurePath:
        """A local :class:`~pathlib.Path` on disk; a pure path of the key on object storage.

        Kept for code written before storage backends existed. Use
        :attr:`uri` to display a location and the catalog to read it.
        """
        location = self.uri or self.key
        return PurePosixPath(self.key) if "://" in location else Path(location)

    @property
    def partition(self) -> Dict[str, Any]:
        """The identifying keys of this file, omitting the ones that don't apply."""
        keys = {
            "year": self.year,
            "meeting_key": self.meeting_key,
            "session_key": self.session_key,
            "driver_number": self.driver_number,
        }
        return {key: value for key, value in keys.items() if value is not None}

    @property
    def sort_key(self) -> tuple:
        """Total order over files, so a pass over the tree is reproducible."""
        return (
            self.endpoint,
            self.year if self.year is not None else -1,
            self.meeting_key if self.meeting_key is not None else -1,
            self.session_key if self.session_key is not None else -1,
            self.driver_number if self.driver_number is not None else -1,
            self.name,
        )


@dataclass(frozen=True)
class RawEnvelope:
    """The contents of a raw file, as :class:`RawDataStorage` wrote it."""

    endpoint: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    retrieved_at: Optional[str] = None
    data: Any = None

    @property
    def records(self) -> List[Any]:
        """The payload as a list, or an empty list when it is not one."""
        return self.data if isinstance(self.data, list) else []


class RawDataCatalog:
    """Discovers and reads the raw files produced by the extraction pipeline."""

    def __init__(self, base_path: Union[str, Path] = "data/raw", *, store: Optional[DataStore] = None) -> None:
        """Create a catalog.

        Args:
            base_path: Root of the raw tree for the legacy layout; the lake
                layout has a fixed ``raw/`` root and ignores it.
            store: Backend and layout to read through. Defaults to local
                files in the historical layout.
        """
        self._configured = str(base_path)
        self._store = store or DataStore.local()
        self._root = self._store.root(Layer.RAW, self._configured)

    @property
    def base_path(self) -> Path:
        """The configured root, as a path. Prefer :attr:`location` for display."""
        return Path(self._configured)

    @property
    def location(self) -> str:
        """Where the raw tree is, as reports record it."""
        return self._store.backend.uri(self._root)

    @property
    def store(self) -> DataStore:
        return self._store

    def exists(self) -> bool:
        return self._store.backend.has_prefix(self._root)

    def discover(
        self,
        endpoints: Optional[Iterable[str]] = None,
        *,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        session_key: Optional[int] = None,
    ) -> List[RawFile]:
        """List raw files, in a stable order.

        Args:
            endpoints: Optional allow-list of endpoint names. ``None`` lists
                everything that is stored, which is what lets validation
                notice data nobody asked for as well as data that is missing.
            year, meeting_key, session_key: Keep only files carrying exactly
                these partition values. On the lake layout they also narrow
                what is listed, so a single-session job does not enumerate
                the whole history.

        Returns:
            Files sorted by endpoint and partition keys. Hidden files and
            leftovers from an interrupted write (``.tmp-…``) are skipped.
        """
        if not self.exists():
            return []

        wanted = set(endpoints) if endpoints is not None else None
        backend, layout = self._store.backend, self._store.layout
        prefix = layout.raw_listing_prefix(self._root, year=year, meeting_key=meeting_key, session_key=session_key)

        files: List[RawFile] = []
        for key in backend.list(prefix):
            name = key_name(key)
            if not name.endswith(".json") or name.startswith("."):
                continue
            parsed = layout.parse_raw_key(self._root, key)
            if parsed is None:
                continue
            if wanted is not None and parsed.endpoint not in wanted:
                continue
            if (
                (year is not None and parsed.year != year)
                or (meeting_key is not None and parsed.meeting_key != meeting_key)
                or (session_key is not None and parsed.session_key != session_key)
            ):
                continue
            files.append(
                RawFile(
                    key=key,
                    endpoint=parsed.endpoint,
                    year=parsed.year,
                    meeting_key=parsed.meeting_key,
                    session_key=parsed.session_key,
                    driver_number=_driver_number_from_name(name),
                    uri=backend.uri(key),
                )
            )

        return sorted(files, key=lambda item: item.sort_key)

    def load(self, raw_file: Union[RawFile, Path, str]) -> RawEnvelope:
        """Read one raw file.

        Args:
            raw_file: A discovered file, or the key/path of one.

        Raises:
            RawFileError: if the file cannot be read, is not valid JSON, or
                does not carry an envelope object.
        """
        key = raw_file.key if isinstance(raw_file, RawFile) else str(raw_file)
        backend = self._store.backend
        location = backend.uri(key)

        try:
            payload = json.loads(backend.read_bytes(key).decode("utf-8"))
        except OSError as exc:
            raise RawFileError(f"Cannot read {location}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RawFileError(f"Invalid JSON in {location}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RawFileError(f"{location} does not contain a raw data envelope (found {type(payload).__name__})")
        if "data" not in payload:
            raise RawFileError(f"{location} has no 'data' key; not a raw data envelope")

        parameters = payload.get("parameters")
        return RawEnvelope(
            endpoint=str(payload.get("endpoint", "")),
            parameters=parameters if isinstance(parameters, Mapping) else {},
            retrieved_at=payload.get("retrieved_at"),
            data=payload.get("data"),
        )


def _driver_number_from_name(file_name: str) -> Optional[int]:
    """Recover the driver from a per-driver file name (``driver_44.json``)."""
    stem = file_name.rsplit(".", 1)[0]
    prefix, separator, value = stem.partition("_")
    if prefix != "driver" or not separator:
        return None
    try:
        return int(value)
    except ValueError:
        return None
