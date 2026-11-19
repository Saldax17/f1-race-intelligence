"""Generate a stable manifest from the existing synthetic M6 test scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from test_modeling_fingerprints import (  # noqa: E402
    EXPECTED_ARTIFACTS,
    Env,
    saved_artifact_hashes,
    stored_artifact_bytes,
)


def build_manifest() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m6-hash-parity-") as directory:
        env = Env("local", Path(directory))
        env.publish()
        report = env.run()
        if report.aborted is not None or report.verdict.value not in {"VALID", "VALID_WITH_WARNINGS"}:
            raise RuntimeError(
                f"Synthetic M6 run did not complete successfully: "
                f"verdict={report.verdict.value}, aborted={report.aborted!r}"
            )

        hashes = saved_artifact_hashes(env)
        stored = stored_artifact_bytes(env)
        if set(hashes) != EXPECTED_ARTIFACTS or len(hashes) != 8:
            raise RuntimeError(f"Expected eight M6 artifacts, found {len(hashes)}")
        actual_hashes = {key: hashlib.sha256(data).hexdigest() for key, data in stored.items()}
        if actual_hashes != hashes:
            raise RuntimeError("Stored artifact bytes do not match the SHA-256 values in the M6 report")
        input_fingerprint = report.source.get("input_fingerprint")
        if not isinstance(input_fingerprint, str) or not input_fingerprint:
            raise RuntimeError("The M6 run did not produce an input_fingerprint")

        artifacts = [
            {
                "artifact_id": f"{split or 'shared'}/{artifact}",
                "split": split,
                "artifact": artifact,
                "sha256": digest,
            }
            for (split, artifact), digest in sorted(
                actual_hashes.items(), key=lambda item: (item[0][0] or "", item[0][1])
            )
        ]
        return {"input_fingerprint": input_fingerprint, "artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path to the JSON manifest to write")
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"Wrote {len(manifest['artifacts'])} M6 artifact hashes to {args.output}")


if __name__ == "__main__":
    main()
