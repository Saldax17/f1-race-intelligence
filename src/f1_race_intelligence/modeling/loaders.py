"""Reading the feature dataset M5 produced.

M6 consumes M5's catalogue as given: the roles assigned there decide what
is a feature, what is traceability and what is refused. Nothing is added
or reclassified here — if a feature turns out to be unusable, that belongs
back in M5's catalogue where the reasoning lives, not in a quiet exclusion
at modelling time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from f1_race_intelligence.features import selection
from f1_race_intelligence.features.selection import TARGET_COLUMN


@dataclass
class LoadedDataset:
    """The M5 rows M6 will work with, and what was left behind."""

    frame: pd.DataFrame
    rows_in: int = 0
    rows_without_target: int = 0
    sessions: int = 0
    files: List[str] = field(default_factory=list)

    @property
    def rows_out(self) -> int:
        return len(self.frame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped_without_target": self.rows_without_target,
            "sessions": self.sessions,
            "files": self.files,
        }


class FeatureDatasetLoader:
    """Finds and reads the Parquet files written by M5."""

    def __init__(self, input_path: Union[str, Path] = "data/features/lap_features") -> None:
        self._input_path = Path(input_path)

    @property
    def input_path(self) -> Path:
        return self._input_path

    def load(
        self,
        years: Optional[List[int]] = None,
        session_keys: Optional[List[int]] = None,
    ) -> LoadedDataset:
        """Read the dataset and keep only rows that can be modelled.

        A row whose target is missing — a driver's last lap — cannot train
        or score a model, so it is dropped here and counted. M5 kept it on
        purpose; dropping it is a modelling decision, made once, in the
        open.
        """
        files = sorted(self._input_path.rglob("session_*.parquet")) if self._input_path.is_dir() else []
        if not files:
            return LoadedDataset(frame=_empty_frame())

        frames = [pd.read_parquet(path, engine="pyarrow") for path in files]
        combined = pd.concat(frames, ignore_index=True)

        if years:
            combined = combined[combined["year"].isin(years)]
        if session_keys:
            combined = combined[combined["session_key"].isin(session_keys)]

        rows_in = len(combined)
        has_target = combined["has_target"].fillna(False).astype(bool) if "has_target" in combined.columns else combined[TARGET_COLUMN].notna()
        modelling = combined[has_target & combined[TARGET_COLUMN].notna()].copy()

        modelling = modelling.sort_values(
            ["session_key", "driver_number", "lap_number"], kind="stable"
        ).reset_index(drop=True)

        return LoadedDataset(
            frame=modelling,
            rows_in=rows_in,
            rows_without_target=rows_in - len(modelling),
            sessions=int(modelling["session_key"].nunique()) if not modelling.empty else 0,
            files=[str(path) for path in files],
        )


def feature_columns(frame: pd.DataFrame) -> Dict[str, List[str]]:
    """Split M5's approved features into numeric and categorical.

    Booleans travel with the numeric block: once cast to 0/1 they need the
    same imputation and scaling, and one-hot encoding a two-valued column
    would only duplicate it.
    """
    approved = [name for name in selection.feature_columns() if name in frame.columns]
    categorical = [name for name in selection.categorical_columns() if name in approved]
    numeric = [name for name in approved if name not in categorical]
    return {"numeric": numeric, "categorical": categorical, "all": approved}


def _empty_frame() -> pd.DataFrame:
    columns = list(selection.output_columns())
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
