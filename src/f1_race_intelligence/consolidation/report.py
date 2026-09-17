"""Persisting a consolidation report.

Like the extraction manifest and the validation report, this is the record
of one execution: timestamped, and never overwriting an earlier verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from f1_race_intelligence.consolidation.models import ConsolidationReport
from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.backends.local import LocalStorageBackend


def save_report(
    report: ConsolidationReport,
    directory: Union[str, Path],
    *,
    backend: Optional[StorageBackend] = None,
) -> Union[Path, str]:
    """Write the report as JSON and return where it went."""
    backend = backend or LocalStorageBackend()
    stem = f"consolidation_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    return backend.location(backend.write_json_unique(str(directory), stem, report.to_dict()))
