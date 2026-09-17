"""Copying raw data between stores without asking OpenF1 again.

The history already extracted to local disk is the expensive part of the
dataset: re-downloading it costs hours of rate-limited requests. This copies
raw envelopes byte for byte from one store and layout to another — the
local legacy tree into the lake layout on S3, typically — together with the
extraction manifests that explain gaps in them.

Copies are idempotent (an object already at the destination is left alone
unless ``overwrite``) and nothing is ever deleted at the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends.base import join_key, key_name
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CopySummary:
    copied: int = 0
    skipped: int = 0
    bytes_copied: int = 0
    manifests_copied: int = 0
    manifests_skipped: int = 0
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copied": self.copied,
            "skipped": self.skipped,
            "bytes_copied": self.bytes_copied,
            "manifests_copied": self.manifests_copied,
            "manifests_skipped": self.manifests_skipped,
            "failed": len(self.errors),
            "errors": dict(list(self.errors.items())[:20]),
        }


def copy_raw(
    source: DataStore,
    target: DataStore,
    *,
    raw_path: str = "data/raw",
    manifest_path: str = "data/manifests",
    scope: Optional[SessionScope] = None,
    overwrite: bool = False,
    include_manifests: bool = True,
) -> CopySummary:
    """Copy raw files (and extraction manifests) from ``source`` to ``target``.

    Args:
        source, target: Stores to copy between; their layouts may differ.
        raw_path, manifest_path: Configured legacy paths, used by whichever
            side has the legacy layout.
        scope: Copy only one season or session. Manifests are copied only
            for an unscoped copy, since a manifest describes a whole run.
        overwrite: Replace objects that already exist at the destination.
    """
    scope = scope or SessionScope()
    summary = CopySummary()
    catalog = RawDataCatalog(base_path=raw_path, store=source)
    target_root = target.root(Layer.RAW, raw_path)

    for raw_file in catalog.discover(year=scope.year, meeting_key=scope.meeting_key, session_key=scope.session_key):
        partition = {"year": raw_file.year, "meeting_key": raw_file.meeting_key, "session_key": raw_file.session_key}
        destination = target.layout.raw_key(target_root, raw_file.endpoint, partition, raw_file.name)
        try:
            if not overwrite and target.backend.exists(destination):
                summary.skipped += 1
                continue
            data = source.backend.read_bytes(raw_file.key)
            target.backend.write_bytes(destination, data, content_type="application/json")
        except OSError as exc:
            summary.errors[raw_file.uri] = f"{type(exc).__name__}: {exc}"
            logger.error("raw_copy_failed", extra={"path": raw_file.uri, "error": str(exc), "error_type": type(exc).__name__})
            continue
        summary.copied += 1
        summary.bytes_copied += len(data)
        logger.debug("raw_copied", extra={"source": raw_file.uri, "destination": target.backend.uri(destination)})

    if include_manifests and scope.is_empty:
        source_root = source.root(Layer.EXTRACTION_MANIFESTS, manifest_path)
        target_manifests = target.root(Layer.EXTRACTION_MANIFESTS, manifest_path)
        for key in source.backend.list(source_root):
            name = key_name(key)
            if not (name.startswith("manifest_") and name.endswith(".json")):
                continue
            # Keep any per-session sub-prefix the source used.
            relative = key[len(join_key(source_root)) :].lstrip("/")
            destination = join_key(target_manifests, relative)
            if not overwrite and target.backend.exists(destination):
                summary.manifests_skipped += 1
                continue
            target.backend.write_bytes(destination, source.backend.read_bytes(key), content_type="application/json")
            summary.manifests_copied += 1

    logger.info("raw_copy_summary", extra=summary.to_dict())
    return summary
