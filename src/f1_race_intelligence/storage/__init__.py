"""Data persistence: storage backends, the path contract, and the raw tree.

Stages depend on :class:`StorageBackend` (through a :class:`DataStore`),
never on a concrete filesystem or on boto3.
"""

from f1_race_intelligence.storage.backends import LocalStorageBackend, S3StorageBackend, StorageBackend, StorageError
from f1_race_intelligence.storage.layout import DatasetLayout, LakeLayout, Layer, LegacyLayout
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from f1_race_intelligence.storage.store import DataStore, store_from_uri

__all__ = [
    "DataStore",
    "DatasetLayout",
    "LakeLayout",
    "Layer",
    "LegacyLayout",
    "LocalStorageBackend",
    "RawDataStorage",
    "S3StorageBackend",
    "StorageBackend",
    "StorageError",
    "store_from_uri",
]
