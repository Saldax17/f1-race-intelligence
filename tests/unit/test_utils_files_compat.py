"""Compatibility coverage for the deprecated ``utils.files`` module."""

import io
import json
from pathlib import Path

import pytest

from f1_race_intelligence.utils import files


def canonical_json(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def test_dump_json_keeps_the_legacy_text_handle_signature() -> None:
    payload = {"driver": "Pérez", "values": [1, None]}
    handle = io.StringIO()

    with pytest.warns(DeprecationWarning):
        files.dump_json(payload, handle)

    assert handle.getvalue() == canonical_json(payload).decode("utf-8")


def test_legacy_file_helpers_delegate_to_canonical_binary_storage(tmp_path: Path) -> None:
    payload = {"driver": "Pérez", "values": [1, None]}
    target = tmp_path / "reports" / "report.json"

    with pytest.warns(DeprecationWarning):
        written = files.write_json_atomically(target, payload)
    with pytest.warns(DeprecationWarning):
        first = files.write_json_unique(target.parent, "manifest", payload)
    with pytest.warns(DeprecationWarning):
        second = files.write_json_unique(target.parent, "manifest", {"run": 2})

    assert written == target
    assert target.read_bytes() == canonical_json(payload)
    assert first.name == "manifest.json"
    assert second.name == "manifest-1.json"
    assert first.read_bytes() == canonical_json(payload)
    assert second.read_bytes() == canonical_json({"run": 2})
