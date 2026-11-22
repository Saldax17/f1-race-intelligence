"""The versioned model file and where it lives.

A model version is one JSON document (``model.json``) holding every fitted
quantity: level-1 noise and tyre terms, the season filter's hyperparameters
*and* its state after the last race it has seen, the pre-race prior of every
race replayed, the event model and the transition baseline. JSON rather than
pickle: it is diffable, safe to load, and readable from any language.

Keys (lake layout; the legacy layout puts the same tree under
``race_model.output_path``)::

    models/m7/version=<v>/model.json
    models/m7/latest.json                 {"version": <v>, ...}  — what forecasts use
    manifests/models/m7/<record>_<ts>.json   training / evaluation / update reports
    forecasts/m7/year=<y>/session=<s>/lap=<k>.json
    forecasts/m7/year=<y>/session=<s>/latest.json

Versions are never overwritten; promotion only moves the pointer, so
rolling back is writing the previous version into ``latest.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from f1_race_intelligence.race_model.contracts import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    PUBLISHED_MARKER,
    ModelPackageIntegrityError,
    canonical_json_bytes,
    publish_version_package,
    verify_version_package,
)
from f1_race_intelligence.storage.backends.base import join_key
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore

FORMAT = "f1-race-model/1"
MODEL_NAME = "m7"


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _root(store: DataStore, layer: Layer, configured: str, legacy_sub: str) -> str:
    root = store.root(layer, configured)
    if store.layout.name == "legacy":
        return join_key(root, legacy_sub) if legacy_sub else root
    return join_key(root, MODEL_NAME)


def models_prefix(store: DataStore, output_path: str) -> str:
    return _root(store, Layer.MODELS, output_path, "")


def reports_prefix(store: DataStore, output_path: str) -> str:
    return _root(store, Layer.MODEL_REPORTS, output_path, "reports")


def forecasts_prefix(store: DataStore, output_path: str) -> str:
    return _root(store, Layer.FORECASTS, output_path, "forecasts")


@dataclass
class ModelArtifact:
    version: str
    created_at: str
    settings: Dict[str, Any]
    level1: Dict[str, Any]
    season: Dict[str, Any]
    events: Dict[str, Any]
    forecast: Dict[str, Any]
    transition: Dict[str, Any]
    races: List[Dict[str, Any]]
    metrics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    format: str = FORMAT

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelArtifact":
        if data.get("format") != FORMAT:
            raise ValueError(f"unsupported model format {data.get('format')!r} (expected {FORMAT})")
        kwargs = {k: data.get(k) for k in cls.__dataclass_fields__}
        if kwargs.get("metrics") is None:
            kwargs["metrics"] = {}
        if kwargs.get("provenance") is None:
            kwargs["provenance"] = {}
        return cls(**kwargs)


class ModelRegistry:
    """Reads and writes immutable model versions, SHA-256 manifests and ``latest``."""

    def __init__(self, store: DataStore, output_path: str) -> None:
        self.store = store
        self.prefix = models_prefix(store, output_path)

    def version_prefix(self, version: str) -> str:
        return join_key(self.prefix, f"version={version}")

    def version_key(self, version: str) -> str:
        return join_key(self.version_prefix(version), MODEL_FILENAME)

    @property
    def latest_key(self) -> str:
        return join_key(self.prefix, "latest.json")

    def _version_has_any_object(self, version: str) -> bool:
        prefix = self.version_prefix(version)
        backend = self.store.backend
        return (
            backend.exists(self.version_key(version))
            or backend.exists(join_key(prefix, MANIFEST_FILENAME))
            or backend.exists(join_key(prefix, PUBLISHED_MARKER))
            or bool(backend.list(prefix))
        )

    def _require_verified(self, version: str) -> Dict[str, Any]:
        if not self._version_has_any_object(version):
            raise LookupError(f"model version {version} does not exist")
        verification = self.verify(version)
        if not verification["ok"]:
            raise ModelPackageIntegrityError(
                f"model version {version} failed package integrity verification: "
                f"{'; '.join(verification['errors'])}"
            )
        return verification

    def save(self, artifact: ModelArtifact) -> str:
        data = canonical_json_bytes(artifact.to_dict())
        published = publish_version_package(
            self.store.backend,
            self.version_prefix(artifact.version),
            version=artifact.version,
            created_at=artifact.created_at,
            model_format=artifact.format,
            model_bytes=data,
            provenance=artifact.provenance,
            parent_version=artifact.parent_version,
        )
        return published["model_uri"]

    def read_model_bytes(self, version: Optional[str] = None) -> bytes:
        version = version or self.latest_version()
        if version is None:
            raise LookupError(f"no promoted model at {self.store.backend.uri(self.latest_key)}; run m7-train and m7-evaluate --promote")
        self._require_verified(version)
        return self.store.backend.read_bytes(self.version_key(version))

    def verify(self, version: Optional[str] = None) -> Dict[str, Any]:
        version = version or self.latest_version()
        if version is None:
            return {
                "ok": False,
                "version": None,
                "files_checked": 0,
                "errors": [f"no promoted model at {self.store.backend.uri(self.latest_key)}"],
            }
        return verify_version_package(
            self.store.backend,
            self.version_prefix(version),
            version,
            expected_format=FORMAT,
        )

    def load(self, version: Optional[str] = None) -> ModelArtifact:
        version = version or self.latest_version()
        if version is None:
            raise LookupError(f"no promoted model at {self.store.backend.uri(self.latest_key)}; run m7-train and m7-evaluate --promote")
        self._require_verified(version)
        return ModelArtifact.from_dict(json.loads(self.store.backend.read_bytes(self.version_key(version))))

    def latest_version(self) -> Optional[str]:
        if not self.store.backend.exists(self.latest_key):
            return None
        return json.loads(self.store.backend.read_bytes(self.latest_key)).get("version")

    def promote(self, version: str, reason: Mapping[str, Any]) -> str:
        verification = self._require_verified(version)
        pointer = {
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": verification["manifest_sha256"],
            **reason,
        }
        return self.store.backend.write_json(self.latest_key, pointer)

    def versions(self) -> List[str]:
        """Published model versions (requires ``PUBLISHED`` marker so partial saves stay invisible)."""
        out = []
        for key in self.store.backend.list(self.prefix):
            parts = key.split("/")
            if len(parts) >= 2 and parts[-1] == PUBLISHED_MARKER and parts[-2].startswith("version="):
                out.append(parts[-2][len("version="):])
        return sorted(set(out))
