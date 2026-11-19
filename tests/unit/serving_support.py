"""Synthetic M6 artifacts and stub models for the serving tests.

Everything here is made up: rows shaped like M5's output, run through the
real M6 runner so the preprocessing artifacts are genuine M6 output, and
models that only record what they receive. No real dataset, no real model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.modeling.runner import ModelingRunner
from f1_race_intelligence.serving.model import LoadedModel
from f1_race_intelligence.validation.states import RunVerdict

RACE_START = pd.Timestamp("2023-03-05T15:00:00Z")
CATEGORICAL = ("compound", "team_name", "driver_acronym")


def m5_rows(session_key: int, year: int, day_offset: int, laps: int = 5) -> List[Dict[str, Any]]:
    """Rows shaped like M5's output for one synthetic race."""
    start = RACE_START + pd.Timedelta(days=day_offset)
    rows = []
    for driver, compound in ((1, "SOFT"), (44, "MEDIUM")):
        for lap in range(1, laps + 1):
            rows.append(
                {
                    "year": year,
                    "meeting_key": 1000 + session_key,
                    "session_key": session_key,
                    "driver_number": driver,
                    "lap_number": lap,
                    "session_date": start,
                    "lap_duration": 90.0 + lap + driver * 0.1,
                    "tyre_age": lap,
                    "position_at_lap_start": 1 if driver == 1 else 2,
                    "speed_mean": 200.0 + lap + session_key * 10,
                    "gap_to_leader_s": None if lap == 1 else float(lap),
                    "lap_time_prev_1": None if lap == 1 else 90.0 + lap - 1,
                    "compound": compound,
                    "team_name": "Red Bull" if driver == 1 else "Mercedes",
                    "driver_acronym": f"D{driver}",
                    "is_pit_out_lap": lap == 1,
                    "pit_in_lap": False,
                    "is_lapped": False,
                    "next_lap_time": 91.0 + lap + driver * 0.1 if lap < laps else None,
                    "has_target": lap < laps,
                }
            )
    return rows


@dataclass(frozen=True)
class M6Run:
    preprocessing_dir: Path
    output_dir: Path
    run_id: str
    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]

    def split_file(self, split: str, name: str) -> pd.DataFrame:
        (path,) = (self.output_dir / split / "runs" / self.run_id).glob(name)
        return pd.read_parquet(path, engine="pyarrow")


def build_m6_run(root: Path) -> M6Run:
    """Three synthetic races through the real M6 runner."""
    for session_key, year, day in ((1, 2023, 0), (2, 2024, 365), (3, 2025, 730)):
        base = root / "lap_features" / str(year)
        base.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(m5_rows(session_key, year, day))
        for column in CATEGORICAL:
            frame[column] = frame[column].astype("category")
        frame.to_parquet(base / f"session_{session_key}.parquet", engine="pyarrow", index=False)

    output = root / "modeling"
    settings = AppSettings(
        modeling={
            "input_path": str(root / "lap_features"),
            "output_path": str(output),
            "report_path": str(output / "reports"),
            "train_fraction": 1 / 3,
            "validation_fraction": 1 / 3,
            "min_sessions_per_split": 1,
        }
    )
    report = ModelingRunner(settings).run()
    assert report.verdict in (RunVerdict.VALID, RunVerdict.VALID_WITH_WARNINGS), report.blocking_failures
    (preprocessing_dir,) = [path.parent for path in output.rglob("preprocessor.joblib")]
    return M6Run(
        preprocessing_dir=preprocessing_dir,
        output_dir=output,
        run_id=report.run_id,
        numeric=tuple(report.features["numeric"]),
        categorical=tuple(report.features["categorical"]),
    )


def to_records(frame: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Rows as a JSON client would send them: plain scalars, null for missing."""
    records = []
    for _, row in frame.iterrows():
        record: Dict[str, Any] = {}
        for column in columns:
            value = row[column]
            if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
                record[column] = None
            elif isinstance(value, (bool, np.bool_)):
                record[column] = bool(value)
            elif isinstance(value, (np.integer,)):
                record[column] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[column] = float(value)
            else:
                record[column] = value if isinstance(value, (int, str)) else str(value)
        records.append(record)
    return records


class RecordingModel:
    """A stand-in model: remembers its inputs and answers 80, 81, 82..."""

    def __init__(self, output: Optional[Any] = None) -> None:
        self.inputs: List[pd.DataFrame] = []
        self._output = output

    def predict(self, data: pd.DataFrame) -> Any:
        self.inputs.append(data.copy())
        if self._output is not None:
            return self._output
        return 80.0 + np.arange(len(data), dtype="float64")


class StubLoader:
    """Hands out a given model, as an MLflow loader would hand out a pyfunc."""

    def __init__(self, model: Any, input_names: Optional[Sequence[str]] = None, run_id: str = "stub-run") -> None:
        self.model, self.input_names, self.run_id = model, input_names, run_id
        self.requested: List[str] = []

    def load(self, uri: str) -> LoadedModel:
        self.requested.append(uri)
        return LoadedModel(
            model=self.model,
            uri=uri,
            run_id=self.run_id,
            flavors=("python_function",),
            input_names=None if self.input_names is None else tuple(self.input_names),
        )
