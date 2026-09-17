"""Persisting a feature engineering report.

Same convention as the extraction manifest, the validation report and the
consolidation report: one timestamped file per execution, never replacing
an earlier one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from f1_race_intelligence.features.models import FeatureReport
from f1_race_intelligence.storage.backends.base import StorageBackend
from f1_race_intelligence.storage.backends.local import LocalStorageBackend


def save_report(
    report: FeatureReport,
    directory: Union[str, Path],
    *,
    backend: Optional[StorageBackend] = None,
) -> Union[Path, str]:
    """Write the report as JSON and return where it went."""
    backend = backend or LocalStorageBackend()
    stem = f"feature_engineering_report_{report.started_at.strftime('%Y%m%dT%H%M%S%f')}Z"
    return backend.location(backend.write_json_unique(str(directory), stem, report.to_dict()))
