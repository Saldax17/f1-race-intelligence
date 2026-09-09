"""Persisting a feature engineering report.

Same convention as the extraction manifest, the validation report and the
consolidation report: one timestamped file per execution, never replacing
an earlier one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from f1_race_intelligence.features.models import FeatureReport
from f1_race_intelligence.utils.files import write_json_unique


def save_report(report: FeatureReport, directory: Union[str, Path]) -> Path:
    """Write the report as JSON and return its path."""
    stem = f"feature_engineering_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    return write_json_unique(directory, stem, report.to_dict())
