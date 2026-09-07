"""Writing files without losing any.

Two rules the rest of the project relies on, in one place because three
modules need them and a subtly different copy in each is how data goes
missing:

* A record of an execution — an extraction manifest, a validation report —
  must never overwrite an earlier one, even when two runs land on the same
  clock tick. The system clock does not always advance between calls (on
  Windows its granularity is around 15 ms), so a timestamped name is not
  unique on its own.
* A file written under a name a later run will look for must appear whole
  or not at all, because "the file exists" is read as "that work is done".
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

# Only reached when many writes share a clock tick; bounded so a bug cannot
# spin forever.
MAX_NAME_COLLISION_RETRIES = 100


def dump_json(payload: Any, handle: Any) -> None:
    """Serialize with the project's single JSON style."""
    json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_json_unique(
    directory: Union[str, Path],
    stem: str,
    payload: Any,
    *,
    extension: str = ".json",
) -> Path:
    """Write ``payload`` under ``stem``, adding a suffix rather than replacing.

    Creating the file exclusively lets the filesystem decide who owns a
    name. That also holds when two processes write at the same instant,
    which a "check whether it exists, then write" cannot promise.

    Returns:
        The path actually written, which may carry a ``-1``, ``-2``… suffix.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_NAME_COLLISION_RETRIES):
        suffix = "" if attempt == 0 else f"-{attempt}"
        candidate = target / f"{stem}{suffix}{extension}"
        try:
            with candidate.open("x", encoding="utf-8") as handle:
                dump_json(payload, handle)
        except FileExistsError:
            continue
        return candidate

    raise OSError(
        f"Could not find a free file name for {stem}{extension} in {target} "
        f"after {MAX_NAME_COLLISION_RETRIES} attempts"
    )


def write_json_atomically(path: Union[str, Path], payload: Any) -> Path:
    """Put a JSON file in place in a single step, never half-written.

    The payload goes to a temporary file in the same directory and is then
    moved onto the target name, so an interrupted run leaves either the
    previous file or nothing — never a truncated one that a later run would
    accept as complete.
    """
    file_path = Path(path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=".tmp-",
            suffix=".json",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            dump_json(payload, handle)
        os.replace(temp_path, file_path)
    except BaseException:
        # Never leave a stray .tmp- file behind for the next stage to find.
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return file_path
