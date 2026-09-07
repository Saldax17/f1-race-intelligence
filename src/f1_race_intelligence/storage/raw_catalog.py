"""Reading side of the raw data layout.

:class:`~f1_race_intelligence.storage.raw_storage.RawDataStorage` decides
where a response is written; this module is the inverse — it walks that
tree and hands back what is there. Both live in ``storage`` so the
``<endpoint>/year=…/meeting_key=…/session_key=…`` convention and the
``{endpoint, parameters, retrieved_at, data}`` envelope are described in
exactly one package. A consumer (validation today, dataset building
later) should never have to parse a path or an envelope itself.

Nothing here writes: raw data is immutable once extracted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

# Partition keys RawDataStorage encodes as "key=value" directory names,
# in the order it nests them.
_PARTITION_KEYS = ("year", "meeting_key", "session_key")


class RawFileError(Exception):
    """Raised when a raw file cannot be read or is not shaped like an envelope.

    Callers are expected to catch this and report it, rather than let one
    damaged file end a whole pass over the tree.
    """


@dataclass(frozen=True)
class RawFile:
    """A raw JSON file on disk, described by where it sits in the tree."""

    path: Path
    endpoint: str
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None
    driver_number: Optional[int] = None

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
            self.path.name,
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

    def __init__(self, base_path: Union[str, Path] = "data/raw") -> None:
        self._base_path = Path(base_path)

    @property
    def base_path(self) -> Path:
        return self._base_path

    def exists(self) -> bool:
        return self._base_path.is_dir()

    def discover(self, endpoints: Optional[Iterable[str]] = None) -> List[RawFile]:
        """List every raw file under the tree, in a stable order.

        Args:
            endpoints: Optional allow-list of endpoint names. ``None`` walks
                everything that is on disk, which is what lets validation
                notice data nobody asked for as well as data that is missing.

        Returns:
            Files sorted by endpoint and partition keys. Hidden files and
            leftovers from an interrupted write (``.tmp-…``) are skipped.
        """
        if not self.exists():
            return []

        wanted = set(endpoints) if endpoints is not None else None
        files: List[RawFile] = []

        for path in self._base_path.rglob("*.json"):
            if not path.is_file() or path.name.startswith("."):
                continue
            raw_file = self._describe(path)
            if raw_file is None:
                continue
            if wanted is not None and raw_file.endpoint not in wanted:
                continue
            files.append(raw_file)

        return sorted(files, key=lambda item: item.sort_key)

    def load(self, raw_file: Union[RawFile, Path]) -> RawEnvelope:
        """Read one raw file.

        Raises:
            RawFileError: if the file cannot be read, is not valid JSON, or
                does not carry an envelope object.
        """
        path = raw_file.path if isinstance(raw_file, RawFile) else Path(raw_file)

        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except OSError as exc:
            raise RawFileError(f"Cannot read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RawFileError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RawFileError(f"{path} does not contain a raw data envelope (found {type(payload).__name__})")
        if "data" not in payload:
            raise RawFileError(f"{path} has no 'data' key; not a raw data envelope")

        parameters = payload.get("parameters")
        return RawEnvelope(
            endpoint=str(payload.get("endpoint", "")),
            parameters=parameters if isinstance(parameters, Mapping) else {},
            retrieved_at=payload.get("retrieved_at"),
            data=payload.get("data"),
        )

    def _describe(self, path: Path) -> Optional[RawFile]:
        """Turn a path back into the partition keys that produced it."""
        try:
            relative = path.relative_to(self._base_path)
        except ValueError:  # pragma: no cover - rglob only yields children
            return None

        parts = relative.parts
        if len(parts) < 2:  # an endpoint directory plus a file name at minimum
            return None

        endpoint = parts[0]
        partition: Dict[str, int] = {}
        for part in parts[1:-1]:
            key, separator, value = part.partition("=")
            if separator and key in _PARTITION_KEYS:
                try:
                    partition[key] = int(value)
                except ValueError:
                    continue

        return RawFile(
            path=path,
            endpoint=endpoint,
            year=partition.get("year"),
            meeting_key=partition.get("meeting_key"),
            session_key=partition.get("session_key"),
            driver_number=_driver_number_from_name(path.name),
        )


def _driver_number_from_name(file_name: str) -> Optional[int]:
    """Recover the driver from a per-driver file name (``driver_44.json``)."""
    stem = Path(file_name).stem
    prefix, separator, value = stem.partition("_")
    if prefix != "driver" or not separator:
        return None
    try:
        return int(value)
    except ValueError:
        return None
