"""Persisting a validation report.

Like the extraction manifest, a report is the record of one execution, so
its file name carries the timestamp and a new run never overwrites an
older verdict.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Union

from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.validation.models import ValidationReport

_CSV_COLUMNS = (
    "rule",
    "severity",
    "status",
    "endpoint",
    "affected_records",
    "description",
    "file",
)


def save_report(
    report: ValidationReport,
    directory: Union[str, Path],
    *,
    write_csv: bool = False,
    backend: Optional[StorageBackend] = None,
) -> List[Union[Path, str]]:
    """Write the report as JSON, optionally alongside a flat CSV.

    Returns:
        Where the files went (local paths, or URIs on object storage), JSON
        first. The CSV is a triage aid — one row per finding, sortable in a
        spreadsheet — and holds nothing the JSON does not.
    """
    backend = backend or LocalStorageBackend()
    stem = f"validation_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    json_key = backend.write_json_unique(str(directory), stem, report.to_dict())
    written = [backend.location(json_key)]

    if write_csv:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in report.results:
            row = result.to_dict()
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})
        # Derive the CSV from the name the JSON actually got, so a pair that
        # collided on the timestamp stays a pair.
        csv_key = json_key[: -len(".json")] + ".csv"
        backend.write_text(csv_key, buffer.getvalue(), content_type="text/csv")
        written.append(backend.location(csv_key))

    return written
