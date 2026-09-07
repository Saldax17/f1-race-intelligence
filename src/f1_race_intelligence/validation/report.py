"""Persisting a validation report.

Like the extraction manifest, a report is the record of one execution, so
its file name carries the timestamp and a new run never overwrites an
older verdict.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Union

from f1_race_intelligence.utils.files import write_json_unique
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
) -> List[Path]:
    """Write the report as JSON, optionally alongside a flat CSV.

    Returns:
        The files written, JSON first. The CSV is a triage aid — one row
        per finding, sortable in a spreadsheet — and holds nothing the
        JSON does not.
    """
    stem = f"validation_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    json_path = write_json_unique(directory, stem, report.to_dict())
    written = [json_path]

    if write_csv:
        # Derive the CSV from the name the JSON actually got, so a pair that
        # collided on the timestamp stays a pair.
        csv_path = json_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for result in report.results:
                row = result.to_dict()
                writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})
        written.append(csv_path)

    return written
