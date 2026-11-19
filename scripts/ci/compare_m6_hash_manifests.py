"""Fail unless two platform manifests contain identical M6 fingerprints and hashes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

EXPECTED_ARTIFACT_COUNT = 8
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ArtifactKey = Tuple[str | None, str]


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {path} must be a JSON object")
    fingerprint = manifest.get("input_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Manifest {path} has no input_fingerprint")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError(f"Manifest {path} has no artifact list")

    artifacts: Dict[ArtifactKey, Dict[str, str | None]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"artifact_id", "split", "artifact", "sha256"}:
            raise ValueError(f"Manifest {path} contains an invalid artifact record")
        split, artifact, artifact_id, digest = (
            record["split"],
            record["artifact"],
            record["artifact_id"],
            record["sha256"],
        )
        if split is not None and not isinstance(split, str):
            raise ValueError(f"Manifest {path} contains an invalid split")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError(f"Manifest {path} contains an invalid artifact name")
        expected_id = f"{split or 'shared'}/{artifact}"
        if artifact_id != expected_id:
            raise ValueError(f"Manifest {path} has a noncanonical artifact_id: {artifact_id!r}")
        if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
            raise ValueError(f"Manifest {path} contains an invalid SHA-256 for {artifact_id}")
        key = (split, artifact)
        if key in artifacts:
            raise ValueError(f"Manifest {path} contains duplicate artifact {artifact_id}")
        artifacts[key] = {"artifact_id": artifact_id, "sha256": digest}

    if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError(f"Manifest {path} must contain exactly eight artifacts; found {len(artifacts)}")
    return {"input_fingerprint": fingerprint, "artifacts": artifacts}


def compare_manifests(first_path: Path, second_path: Path) -> None:
    first, second = load_manifest(first_path), load_manifest(second_path)
    if first["input_fingerprint"] != second["input_fingerprint"]:
        raise ValueError("input_fingerprint differs between platforms")

    first_artifacts, second_artifacts = first["artifacts"], second["artifacts"]
    missing = sorted(first_artifacts.keys() - second_artifacts.keys(), key=str)
    unexpected = sorted(second_artifacts.keys() - first_artifacts.keys(), key=str)
    if missing or unexpected:
        raise ValueError(f"Artifact sets differ: missing={missing}, unexpected={unexpected}")
    different = sorted(
        first_artifacts[key]["artifact_id"]
        for key in first_artifacts
        if first_artifacts[key]["sha256"] != second_artifacts[key]["sha256"]
    )
    if different:
        raise ValueError(f"Artifact SHA-256 differs: {different}")
    print("input_fingerprint and all eight artifact SHA-256 values match.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    try:
        compare_manifests(args.first, args.second)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
