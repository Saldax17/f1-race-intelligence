"""The inference contract: what a prediction request may contain.

Nothing here lists features. The contract is read from the
``feature_schema.json`` that M6 stored next to the fitted preprocessor, so the
service accepts exactly the columns that transformer was fitted on, in the
form M5 produces them (untransformed). Encoding is M6's job and happens later,
with M6's own code and M6's own fitted transformer.

Validation is strict on purpose. A misspelled or missing column would
otherwise be imputed as "missing" and produce a confident prediction from
data nobody sent, and a target or leaky column in the request is a sign the
caller is wiring the wrong dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from f1_race_intelligence.features.selection import FORBIDDEN_AS_FEATURES

# The schema M6's runner writes next to PREPROCESSOR_FILE in
# preprocessing/runs/<run_id>/. The runner spells it inline, so the tests pin
# this name against a real runner output.
FEATURE_SCHEMA_FILE = "feature_schema.json"

_SCALARS = (str, int, float, bool, type(None))


class ArtifactError(RuntimeError):
    """The M6 artifacts or the model cannot be served together."""


class ContractError(ValueError):
    """A request does not satisfy the inference contract.

    ``errors`` holds one entry per problem, each with the record index, the
    field and a message, so a caller can fix every issue in one round trip.
    """

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"{len(errors)} contract violation(s)")


@dataclass(frozen=True)
class FeatureContract:
    """Input features, their kinds and the encoded columns a model receives."""

    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]
    encoded_feature_names: Tuple[str, ...]
    target: str
    # Identifying columns (year, session_key, lap_number...): accepted so a
    # caller can correlate answers, echoed back, never sent to the model
    # unless they are also features.
    traceability: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def features(self) -> Tuple[str, ...]:
        return self.numeric + self.categorical

    @classmethod
    def from_schema(cls, payload: Mapping[str, Any]) -> "FeatureContract":
        """Build the contract from M6's ``feature_schema.json``."""
        try:
            inputs = payload["input_features"]
            numeric = _names(inputs["numeric"], "input_features.numeric")
            categorical = _names(inputs["categorical"], "input_features.categorical")
            encoded = _names(payload["encoded_feature_names"], "encoded_feature_names")
            target = payload["target"]
            traceability = _names(payload.get("traceability", []), "traceability")
        except (KeyError, TypeError) as exc:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} is not an M6 feature schema: {exc!r}") from exc
        if not isinstance(target, str) or not target:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} has no target name")
        if not numeric and not categorical:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} lists no input features")
        overlap = set(numeric) & set(categorical)
        if overlap:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} lists features as both numeric and categorical: {sorted(overlap)}")
        forbidden = (set(numeric) | set(categorical)) & (set(FORBIDDEN_AS_FEATURES) | {target})
        if forbidden:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} includes target or leakage features: {sorted(forbidden)}")
        if not encoded:
            raise ArtifactError(f"{FEATURE_SCHEMA_FILE} lists no encoded features")
        return cls(numeric, categorical, encoded, target, traceability)

    # -- requests ---------------------------------------------------------

    def validate(self, records: Sequence[Any]) -> None:
        """Raise :class:`ContractError` listing every problem in ``records``."""
        errors: List[Dict[str, Any]] = []
        features = set(self.features)
        accepted = features | set(self.traceability)
        forbidden = (set(FORBIDDEN_AS_FEATURES) | {self.target}) - features

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                errors.append(_error(index, None, "each record must be a JSON object"))
                continue
            for name in sorted(set(record) - accepted):
                if name in forbidden:
                    errors.append(_error(index, name, "this column must never reach the model (target or leakage)"))
                else:
                    errors.append(_error(index, name, "unknown field: not an M6 input feature or traceability column"))
            for name in self.features:
                if name not in record:
                    errors.append(_error(index, name, "missing feature: send null if the value is unknown"))
            for name in self.numeric:
                if name in record and not _is_number(record[name]):
                    errors.append(_error(index, name, "numeric feature: expected a finite number, a boolean or null"))
            for name in self.categorical:
                if name in record and not (record[name] is None or isinstance(record[name], str)):
                    errors.append(_error(index, name, "categorical feature: expected a string or null"))
            for name in self.traceability:
                if name in record and name not in features and not isinstance(record[name], _SCALARS):
                    errors.append(_error(index, name, "traceability field: expected a JSON scalar"))

        if errors:
            raise ContractError(errors)

    def to_frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """The validated records as an M5-shaped frame of feature columns.

        Booleans become 1.0/0.0 and null becomes NaN in the numeric block,
        which is how M6's ``prepare_matrix`` itself reads M5's booleans;
        categoricals stay strings or None. Nothing is imputed, scaled or
        encoded here: that is done by M6's fitted preprocessor.
        """
        self.validate(records)
        frame = pd.DataFrame(index=range(len(records)))
        for name in self.numeric:
            frame[name] = pd.Series(
                [float("nan") if record[name] is None else float(record[name]) for record in records],
                dtype="float64",
            )
        for name in self.categorical:
            frame[name] = pd.Series([record[name] for record in records], dtype="object")
        return frame

    def keys_of(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """The traceability values a record carried, to echo in the response."""
        return {name: record[name] for name in self.traceability if name in record}

    def describe(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "numeric_features": list(self.numeric),
            "categorical_features": list(self.categorical),
            "traceability_fields": list(self.traceability),
            "encoded_feature_count": len(self.encoded_feature_names),
        }


def _names(values: Any, where: str) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
        raise TypeError(f"{where} must be a list of column names")
    if len(set(values)) != len(values):
        raise TypeError(f"{where} repeats a column name")
    return tuple(values)


def _is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)


def _error(index: int, name: Any, message: str) -> Dict[str, Any]:
    return {"record": index, "field": name, "error": message}
