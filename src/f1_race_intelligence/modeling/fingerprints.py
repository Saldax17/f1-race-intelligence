"""Content fingerprints: which bytes a run read, and which bytes it wrote.

A path says where data was; a hash says what it was. The report already
records the paths, and a path can be overwritten afterwards without the
report noticing. Recording the SHA-256 of the bytes themselves makes the
claim checkable: re-hash the object and compare.

``input_fingerprint`` condenses every input into one value, so two runs can
be compared at a glance — same fingerprint, same data in. It is computed
from a canonical document that holds, for each input file and nothing else:

* ``input``: the file's logical name (``session_<key>.parquet``), which is
  the same in every layout;
* ``year`` and ``session_key``: the session the layout assigns the file to;
* ``sha256``: the hash of the file's bytes.

Entries are sorted by ``year`` (``null`` first), ``session_key``,
``input`` and finally ``sha256``, so the order a backend happens to list
files in cannot change the result. The document is
``{"inputs": [...], "scheme": "<INPUT_FINGERPRINT_SCHEME>"}`` serialised as
JSON with sorted keys, no whitespace and ASCII escapes, and the fingerprint
is ``sha256:`` plus the hex digest of those bytes.

Nothing about the location — URI, absolute path, bucket, prefix, backend —
and nothing about the run — clock, randomness, run number — enters it, so
the same files give the same fingerprint on local disk and on S3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

HASH_ALGORITHM = "sha256"

# Bump when the canonical document changes, so fingerprints computed under
# different rules are never mistaken for comparable.
INPUT_FINGERPRINT_SCHEME = "m6-input-fingerprint/v1"


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class InputFile:
    """One M5 file M6 read, identified by content rather than location."""

    input: str
    year: Optional[int]
    session_key: int
    sha256: str
    size_bytes: int

    def canonical(self) -> Dict[str, Any]:
        """The fields that enter the fingerprint. Size is implied by the hash."""
        return {"input": self.input, "year": self.year, "session_key": self.session_key, "sha256": self.sha256}

    def to_dict(self) -> Dict[str, Any]:
        return {**self.canonical(), "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ArtifactFile:
    """One object M6 wrote, hashed from the bytes read back from storage."""

    artifact: str
    split: Optional[str]
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact": self.artifact,
            "split": self.split,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _canonical_order(item: InputFile) -> tuple:
    return (item.year is not None, item.year or 0, item.session_key, item.input, item.sha256)


def input_fingerprint(inputs: Iterable[InputFile]) -> str:
    """One value for every input, independent of order and location."""
    document = {
        "inputs": [item.canonical() for item in sorted(inputs, key=_canonical_order)],
        "scheme": INPUT_FINGERPRINT_SCHEME,
    }
    data = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return f"{HASH_ALGORITHM}:{sha256_hex(data)}"
