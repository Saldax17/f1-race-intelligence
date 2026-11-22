"""Optional MLflow Pyfunc wrapper for the M7 Race Model.

This module is imported strictly lazily inside ``log_race_model_to_mlflow()``
(and by ``mlflow.pyfunc.load_model()`` when reloading a logged model), so
importing base ``f1_race_intelligence.race_model`` modules never imports
``mlflow`` even when the optional ``training`` extra is installed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd

from f1_race_intelligence.race_model.artifact import ModelArtifact
from f1_race_intelligence.race_model.contracts import (
    canonical_json_bytes,
    validate_forecast_response,
)
from f1_race_intelligence.race_model.racedata import (
    race_from_forecast_payload,
    truncate_at_lap,
)
from f1_race_intelligence.race_model.runner import LoadedModel


def _optional_int(value: Any, field_name: str, *, minimum: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name!r} must be an integer, got boolean {value!r}")
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise ValueError(f"{field_name!r} must be an integer, got {value!r}")
        ival = int(value)
    elif isinstance(value, (int, np.integer)):
        ival = int(value)
    else:
        raise ValueError(f"{field_name!r} must be an integer, got {value!r}")
    if minimum is not None and ival < minimum:
        raise ValueError(f"{field_name!r} must be >= {minimum}, got {ival}")
    return ival


def _normalize_input_rows(model_input: Any) -> List[Dict[str, Any]]:
    if isinstance(model_input, pd.DataFrame):
        rows = model_input.to_dict(orient="records")
    elif isinstance(model_input, Mapping):
        rows = [dict(model_input)]
    elif isinstance(model_input, list):
        rows = [dict(r) for r in model_input]
    else:
        raise ValueError("model_input must be a DataFrame, dict, or list of dicts")
    if not rows:
        raise ValueError("model_input must contain at least one forecast request row")
    return rows


class RaceModelPyfuncModel(mlflow.pyfunc.PythonModel):
    """Stateless MLflow Pyfunc executing the real Race Model forecast pipeline.

    Each input row provides:
      - ``race_json`` (JSON string or dict from ``race_to_forecast_payload``)
      - ``lap`` (optional positive int; defaults to ``race.leader_laps()``)
      - ``total_laps`` (optional positive int >= lap; defaults to model circuit distance)
      - ``seed`` (optional int; defaults to ``artifact.settings["random_seed"]``)

    Each output row returns the real, contract-validated forecast payload
    (identical to ``RaceModelRunner.forecast`` for the same inputs and seed):
      - scalar summary columns: ``model_version``, ``session_key``, ``race``,
        ``lap``, ``total_laps``, ``regime_at_k``, ``scenarios``
      - ``forecast_json``: canonical JSON encoding of the full validated
        ``ForecastResult.to_dict()`` payload (``drivers``, ``matrix``, ``ahead``,
        ``p_dnf``, ``p_win``, ``p_podium``, ``p_points``, etc.).
    """

    def load_context(self, context) -> None:
        model_path = Path(context.artifacts["model_json"])
        artifact = ModelArtifact.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
        self._artifact = artifact
        self._model = LoadedModel.from_artifact(artifact)

    def predict(self, context, model_input, params=None):
        rows = _normalize_input_rows(model_input)
        default_seed = int((self._artifact.settings or {}).get("random_seed", 42))
        outputs: List[Dict[str, Any]] = []

        for idx, row in enumerate(rows):
            if "race_json" not in row or row["race_json"] is None:
                raise ValueError(f"row {idx}: missing required 'race_json' field")
            raw_race = row["race_json"]
            if isinstance(raw_race, str):
                try:
                    race_payload = json.loads(raw_race)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"row {idx}: invalid 'race_json' ({exc})") from exc
            elif isinstance(raw_race, Mapping):
                race_payload = raw_race
            else:
                raise ValueError(f"row {idx}: 'race_json' must be a JSON string or object")

            lap = _optional_int(row.get("lap"), "lap", minimum=1)
            total_laps_override = _optional_int(row.get("total_laps"), "total_laps", minimum=1)
            seed = _optional_int(row.get("seed"), "seed")
            if seed is None:
                seed = default_seed

            race = race_from_forecast_payload(race_payload)
            k = int(lap) if lap is not None else int(race.leader_laps())
            if k < 1:
                raise ValueError(f"row {idx}: the race has not completed a lap yet (lap={k})")
            if total_laps_override is not None and k > total_laps_override:
                raise ValueError(f"row {idx}: 'lap' ({k}) cannot exceed 'total_laps' ({total_laps_override})")

            race_k = truncate_at_lap(race, k)
            total = self._model.total_laps(race, total_laps_override)
            if k > total:
                raise ValueError(f"row {idx}: 'lap' ({k}) exceeds resolved 'total_laps' ({total})")

            result = self._model.run(race_k, self._model.prior_for(race), total, seed=seed)
            payload = result.to_dict()
            validate_forecast_response(payload)

            outputs.append(
                {
                    "model_version": str(payload["model_version"]),
                    "session_key": int(payload["session_key"]),
                    "race": str(payload["race"]),
                    "lap": int(payload["lap"]),
                    "total_laps": int(payload["total_laps"]),
                    "regime_at_k": str(payload["regime_at_k"]),
                    "scenarios": int(payload["scenarios"]),
                    "forecast_json": canonical_json_bytes(payload).decode("utf-8"),
                }
            )

        return pd.DataFrame(outputs)
