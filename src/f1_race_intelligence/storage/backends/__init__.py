"""Storage backends: the interface stages depend on, and its implementations.

``S3StorageBackend`` is importable without boto3 installed; boto3 is only
needed once a client has to be created.
"""

from f1_race_intelligence.storage.backends.base import StorageBackend, StorageError, join_key
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend

__all__ = ["LocalStorageBackend", "S3StorageBackend", "StorageBackend", "StorageError", "join_key"]
