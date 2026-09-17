"""A backend paired with a layout: everything a stage needs to find its data.

Stages receive a :class:`DataStore` rather than building paths themselves.
Which one they get is decided by configuration (``storage`` in
``config.yaml``) or, for a single run, by a storage URI on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.backends.s3 import S3StorageBackend
from f1_race_intelligence.storage.layout import DatasetLayout, Layer, LegacyLayout, get_layout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from f1_race_intelligence.config.settings import AppSettings, StorageSettings

S3_SCHEME = "s3://"

# Where the lake layout lives on local disk when nothing else is configured,
# so a local rehearsal of the S3 tree never lands in the working directory.
DEFAULT_LOCAL_LAKE_ROOT = "data/lake"


@dataclass(frozen=True)
class DataStore:
    """Storage for one side of a stage: its inputs or its outputs."""

    backend: StorageBackend
    layout: DatasetLayout

    def root(self, layer: Layer, configured: str = "") -> str:
        """Top-level key of a layer; ``configured`` is the stage path from config."""
        return self.layout.root(layer, configured)

    def describe(self) -> Dict[str, Any]:
        return {**self.backend.describe(), **self.layout.describe()}

    @classmethod
    def local(cls, root: str = ".") -> "DataStore":
        """The historical default: files under the working directory, legacy tree."""
        return cls(backend=LocalStorageBackend(root), layout=LegacyLayout())

    @classmethod
    def from_settings(cls, settings: "AppSettings") -> "DataStore":
        return store_from_storage_settings(settings.storage)


def default_layout_name(backend_name: str) -> str:
    """``auto`` keeps local runs on the historical tree and puts S3 on the lake contract."""
    return "lake" if backend_name == "s3" else "legacy"


def store_from_storage_settings(storage: "StorageSettings", *, client: Any = None) -> DataStore:
    """Build the store described by the ``storage`` configuration section."""
    layout_name = storage.layout if storage.layout != "auto" else default_layout_name(storage.backend)

    if storage.backend == "s3":
        if not storage.bucket:
            raise ValueError("storage.backend is 's3' but no bucket is configured (set F1_STORAGE_BUCKET)")
        backend: StorageBackend = S3StorageBackend(
            storage.bucket,
            storage.prefix,
            client=client,
            region_name=storage.region,
            endpoint_url=storage.endpoint_url,
        )
    elif storage.backend == "local":
        root = storage.local_root or (DEFAULT_LOCAL_LAKE_ROOT if layout_name == "lake" else ".")
        backend = LocalStorageBackend(root)
    else:  # pragma: no cover - settings validation rejects this first
        raise ValueError(f"Unknown storage backend {storage.backend!r}")

    return DataStore(backend=backend, layout=get_layout(layout_name))


def store_from_uri(
    uri: str,
    *,
    layout: str = "auto",
    storage: Optional["StorageSettings"] = None,
    client: Any = None,
) -> DataStore:
    """Build a store from ``s3://bucket/prefix`` or a local directory.

    Args:
        uri: Where the data lives.
        layout: ``legacy``, ``lake``, or ``auto`` (lake for S3, legacy locally).
        storage: Configured storage settings, for S3 client options (region,
            endpoint) that a URI cannot carry.
        client: An S3 client to use instead of creating one.
    """
    if uri.startswith(S3_SCHEME):
        bucket, _, prefix = uri[len(S3_SCHEME) :].partition("/")
        backend: StorageBackend = S3StorageBackend(
            bucket,
            prefix,
            client=client,
            region_name=storage.region if storage else None,
            endpoint_url=storage.endpoint_url if storage else None,
        )
    elif "://" in uri:
        raise ValueError(f"Unsupported storage URI {uri!r}: use s3://bucket/prefix or a local directory")
    else:
        backend = LocalStorageBackend(uri)

    layout_name = layout if layout != "auto" else default_layout_name(backend.name)
    return DataStore(backend=backend, layout=get_layout(layout_name))
