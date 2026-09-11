"""Persisting a modelling-dataset report.

Same convention as every other stage: one timestamped file per execution,
never replacing an earlier one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from f1_race_intelligence.modeling.models import ModelingReport
from f1_race_intelligence.utils.files import write_json_unique


def save_report(report: ModelingReport, directory: Union[str, Path]) -> Path:
    """Write the report as JSON and return its path."""
    stem = f"modeling_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    return write_json_unique(directory, stem, report.to_dict())
