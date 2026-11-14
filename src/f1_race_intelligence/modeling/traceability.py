"""Execution metadata for reproducible M6 reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def new_run_id() -> str:
    return uuid.uuid4().hex


def source_revision() -> Tuple[Optional[str], Optional[bool]]:
    """Return a real source revision and dirty state, when available."""
    commit = os.environ.get("F1_GIT_COMMIT", "").strip()
    dirty_value = os.environ.get("F1_GIT_DIRTY")
    root = Path(__file__).resolve().parents[3]

    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None, None

    if not _COMMIT_RE.fullmatch(commit):
        return None, None

    if dirty_value is not None:
        normalized_dirty = dirty_value.strip().lower()
        if normalized_dirty in {"1", "true", "yes"}:
            return commit.lower(), True
        if normalized_dirty in {"0", "false", "no"}:
            return commit.lower(), False

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return commit.lower(), None
    return commit.lower(), bool(status.strip())


def dependency_lock_sha256() -> Optional[str]:
    """Hash the project lockfile or the path injected by the container build."""
    configured = os.environ.get("F1_DEPENDENCY_LOCK_PATH")
    root = Path(__file__).resolve().parents[3]
    path = Path(configured) if configured else root / "requirements" / "runtime-lock.txt"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None
    return digest if _SHA256_RE.fullmatch(digest) else None


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Hash only M6 settings; storage locations and credentials are excluded."""
    safe_config = {key: value for key, value in config.items() if key not in {"input_storage", "output_storage"}}
    encoded = json.dumps(safe_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
