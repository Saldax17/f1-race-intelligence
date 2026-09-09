"""Persisting a consolidation report.

Like the extraction manifest and the validation report, this is the record
of one execution: timestamped, and never overwriting an earlier verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from f1_race_intelligence.consolidation.models import ConsolidationReport
from f1_race_intelligence.utils.files import write_json_unique


def save_report(report: ConsolidationReport, directory: Union[str, Path]) -> Path:
    """Write the report as JSON and return its path."""
    stem = f"consolidation_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    return write_json_unique(directory, stem, report.to_dict())
