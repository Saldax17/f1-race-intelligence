"""Where each dataset lives: the path contract.

A backend answers *how* bytes are stored; a layout answers *which key* a
given piece of data gets. Two layouts exist:

``legacy``
    Exactly the tree the project has always written, driven by the
    per-stage paths in ``config.yaml``::

        data/raw/<endpoint>/year=<y>/meeting_key=<m>/session_key=<s>/session_<s>.json
        data/processed/lap_dataset/<year>/session_<s>.parquet
        data/features/lap_features/<year>/session_<s>.parquet
        data/modeling/<split>/{dataset,features}.parquet

``lake``
    The data lake contract for object storage. Everything about one session
    shares a single prefix, which is what makes a session an independent
    unit of work (and an IAM/lifecycle boundary)::

        raw/year=<y>/meeting=<m>/session=<s>/endpoint=<endpoint>/<file>.json
        raw/year=<y>/endpoint=meetings/meetings.json
        raw/year=<y>/meeting=<m>/endpoint=sessions/sessions.json
        validated/year=<y>/session=<s>/validation_report_<ts>.json
        consolidated/year=<y>/session=<s>/session_<s>.parquet
        features/year=<y>/session=<s>/session_<s>.parquet
        modeling/split=<train|validation|test>/{dataset,features}.parquet
        modeling/preprocessing/preprocessor.joblib
        manifests/<stage>/[year=<y>/session=<s>/]<record>_<ts>.json
        models/   (reserved for training)
        logs/     (reserved)

    The lake layout ignores the per-stage paths: its layer names are the
    contract, and the bucket and prefix the only things that vary.

``year=`` in the lake is a Hive-style partition that shares its name with
the ``year`` column stored inside the Parquet files. The pipeline never
lets a reader infer partitions from paths (see
:meth:`StorageBackend.read_parquet`); a query engine over these prefixes
should map the partition to a different column name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from f1_race_intelligence.storage.backends.base import join_key, key_name, relative_parts

# The partition keys a raw file is identified by, in nesting order.
RAW_PARTITION_KEYS = ("year", "meeting_key", "session_key")


class Layer(str, Enum):
    """Every logical dataset the pipeline reads or writes."""

    RAW = "raw"
    EXTRACTION_MANIFESTS = "extraction_manifests"
    VALIDATION_REPORTS = "validation_reports"
    CONSOLIDATED = "consolidated"
    CONSOLIDATION_REPORTS = "consolidation_reports"
    FEATURES = "features"
    FEATURE_REPORTS = "feature_reports"
    MODELING = "modeling"
    MODELING_REPORTS = "modeling_reports"
    MODELS = "models"
    LOGS = "logs"


# The lake contract. Changing a value here moves data: treat it as an API.
LAKE_ROOTS: Dict[Layer, str] = {
    Layer.RAW: "raw",
    Layer.EXTRACTION_MANIFESTS: "manifests/extraction",
    Layer.VALIDATION_REPORTS: "validated",
    Layer.CONSOLIDATED: "consolidated",
    Layer.CONSOLIDATION_REPORTS: "manifests/consolidation",
    Layer.FEATURES: "features",
    Layer.FEATURE_REPORTS: "manifests/features",
    Layer.MODELING: "modeling",
    Layer.MODELING_REPORTS: "manifests/modeling",
    Layer.MODELS: "models",
    Layer.LOGS: "logs",
}


@dataclass(frozen=True)
class RawKey:
    """What a raw key says about the file behind it."""

    endpoint: str
    year: Optional[int] = None
    meeting_key: Optional[int] = None
    session_key: Optional[int] = None


class DatasetLayout(ABC):
    """Maps logical datasets to storage keys, and keys back to what they hold."""

    name: str = "abstract"

    @abstractmethod
    def root(self, layer: Layer, configured: str) -> str:
        """Top-level key of a layer. ``configured`` is the stage's path from config."""

    # -- raw ------------------------------------------------------------------

    @abstractmethod
    def raw_directory(self, root: str, endpoint: str, partition: Mapping[str, Any]) -> str:
        """The prefix a raw file for this endpoint and partition is written under."""

    def raw_key(self, root: str, endpoint: str, partition: Mapping[str, Any], file_name: str) -> str:
        return join_key(self.raw_directory(root, endpoint, partition), file_name)

    @abstractmethod
    def raw_listing_prefix(
        self,
        root: str,
        *,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        session_key: Optional[int] = None,
    ) -> str:
        """The narrowest prefix guaranteed to contain every matching raw file."""

    @abstractmethod
    def parse_raw_key(self, root: str, key: str) -> Optional[RawKey]:
        """Recover endpoint and partition from a raw key; ``None`` if it is not one."""

    # -- per-session datasets (consolidated laps, features) -------------------

    @abstractmethod
    def session_dataset_key(self, root: str, year: Optional[int], session_key: int) -> str:
        """Key of one session's Parquet file."""

    @abstractmethod
    def session_dataset_listing_prefix(self, root: str, year: Optional[int] = None) -> str:
        """Prefix to list when looking for session files, narrowed by season when known."""

    def parse_session_dataset_key(self, root: str, key: str) -> Optional[Tuple[Optional[int], int]]:
        """``(year, session_key)`` of a session file, or ``None`` if the key is not one."""
        parts = relative_parts(root, key)
        if not parts:
            return None
        session_key = _session_key_from_name(parts[-1])
        if session_key is None:
            return None
        return self._year_from_directories(parts[:-1]), session_key

    @abstractmethod
    def _year_from_directories(self, directories: List[str]) -> Optional[int]:
        """The season a session file sits under, read from its directory names."""

    # -- modelling ------------------------------------------------------------

    @abstractmethod
    def split_prefix(self, root: str, split: str) -> str:
        """Prefix holding one split's artifacts."""

    def preprocessing_prefix(self, root: str) -> str:
        return join_key(root, "preprocessing")

    # -- execution records ----------------------------------------------------

    @abstractmethod
    def records_prefix(
        self,
        root: str,
        *,
        year: Optional[int] = None,
        session_key: Optional[int] = None,
    ) -> str:
        """Where a manifest or report of a run is written.

        A run scoped to one session may get its own prefix, so a reader
        looking for "the latest record about this session" lists only that.
        """

    def describe(self) -> Dict[str, Any]:
        return {"layout": self.name}


class LegacyLayout(DatasetLayout):
    """The tree the project has always produced on disk."""

    name = "legacy"

    def root(self, layer: Layer, configured: str) -> str:
        return join_key(configured)

    def raw_directory(self, root: str, endpoint: str, partition: Mapping[str, Any]) -> str:
        parts = [endpoint.strip("/")]
        for key in RAW_PARTITION_KEYS:
            if partition.get(key) is not None:
                parts.append(f"{key}={partition[key]}")
        return join_key(root, *parts)

    def raw_listing_prefix(self, root, *, year=None, meeting_key=None, session_key=None) -> str:
        # The endpoint comes first in this tree, so no partition narrows a listing.
        return join_key(root)

    def parse_raw_key(self, root: str, key: str) -> Optional[RawKey]:
        parts = relative_parts(root, key)
        if parts is None or len(parts) < 2:  # an endpoint directory plus a file name at minimum
            return None
        partition = _parse_partitions(parts[1:-1], {key: key for key in RAW_PARTITION_KEYS})
        return RawKey(endpoint=parts[0], **partition)

    def session_dataset_key(self, root: str, year: Optional[int], session_key: int) -> str:
        # Seasons are plain directories, not "year=2023": see ConsolidationRunner.
        if year is None:
            return join_key(root, f"session_{session_key}.parquet")
        return join_key(root, str(year), f"session_{session_key}.parquet")

    def session_dataset_listing_prefix(self, root: str, year: Optional[int] = None) -> str:
        return join_key(root)

    def _year_from_directories(self, directories: List[str]) -> Optional[int]:
        return _as_int(directories[-1]) if directories else None

    def split_prefix(self, root: str, split: str) -> str:
        return join_key(root, split)

    def records_prefix(self, root, *, year=None, session_key=None) -> str:
        return join_key(root)


class LakeLayout(DatasetLayout):
    """The data lake contract for object storage."""

    name = "lake"

    # Directory names in the lake, and the partition field each one encodes.
    _RAW_ALIASES = {"year": "year", "meeting": "meeting_key", "session": "session_key"}

    def root(self, layer: Layer, configured: str) -> str:
        return LAKE_ROOTS[layer]

    def raw_directory(self, root: str, endpoint: str, partition: Mapping[str, Any]) -> str:
        return join_key(root, *self._partition_segments(partition), f"endpoint={endpoint.strip('/')}")

    def raw_listing_prefix(self, root, *, year=None, meeting_key=None, session_key=None) -> str:
        # Only a contiguous run of known keys narrows the prefix.
        segments: List[str] = []
        for name, value in (("year", year), ("meeting", meeting_key), ("session", session_key)):
            if value is None:
                break
            segments.append(f"{name}={value}")
        return join_key(root, *segments)

    def parse_raw_key(self, root: str, key: str) -> Optional[RawKey]:
        parts = relative_parts(root, key)
        if not parts or len(parts) < 2:
            return None
        directories = parts[:-1]
        endpoint = next((part.partition("=")[2] for part in directories if part.startswith("endpoint=")), "")
        if not endpoint:
            return None
        partition = _parse_partitions(directories, self._RAW_ALIASES)
        return RawKey(endpoint=endpoint, **partition)

    def session_dataset_key(self, root: str, year: Optional[int], session_key: int) -> str:
        season = [f"year={year}"] if year is not None else []
        return join_key(root, *season, f"session={session_key}", f"session_{session_key}.parquet")

    def session_dataset_listing_prefix(self, root: str, year: Optional[int] = None) -> str:
        return join_key(root, f"year={year}") if year is not None else join_key(root)

    def _year_from_directories(self, directories: List[str]) -> Optional[int]:
        return _parse_partitions(directories, {"year": "year"}).get("year")

    def split_prefix(self, root: str, split: str) -> str:
        return join_key(root, f"split={split}")

    def records_prefix(self, root, *, year=None, session_key=None) -> str:
        if session_key is None:
            return join_key(root)
        season = [f"year={year}"] if year is not None else []
        return join_key(root, *season, f"session={session_key}")

    @staticmethod
    def _partition_segments(partition: Mapping[str, Any]) -> List[str]:
        segments = []
        for directory, field in (("year", "year"), ("meeting", "meeting_key"), ("session", "session_key")):
            if partition.get(field) is not None:
                segments.append(f"{directory}={partition[field]}")
        return segments


LAYOUTS = {LegacyLayout.name: LegacyLayout, LakeLayout.name: LakeLayout}


def get_layout(name: str) -> DatasetLayout:
    try:
        return LAYOUTS[name]()
    except KeyError:
        raise ValueError(f"Unknown storage layout {name!r}. Supported: {sorted(LAYOUTS)}") from None


def _parse_partitions(directories: List[str], aliases: Mapping[str, str]) -> Dict[str, int]:
    partition: Dict[str, int] = {}
    for part in directories:
        name, separator, value = part.partition("=")
        field = aliases.get(name) if separator else None
        number = _as_int(value) if field else None
        if field and number is not None:
            partition[field] = number
    return partition


def _as_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _session_key_from_name(file_name: str) -> Optional[int]:
    """``session_7953.parquet`` -> 7953."""
    name = key_name(file_name)
    if not name.endswith(".parquet"):
        return None
    prefix, separator, value = name[: -len(".parquet")].partition("_")
    if prefix != "session" or not separator:
        return None
    return _as_int(value)
