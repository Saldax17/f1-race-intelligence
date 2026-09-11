"""Preprocessing fitted on the training split, and only on it.

Every statistic a transformer learns — a median, a mean, a set of
categories — is estimated from the training races alone. Fitting on the
whole dataset would let the model absorb properties of the races it is
later scored on, and the score would stop meaning what it appears to mean.

Two details of this dataset drive the implementation:

* M5 emits pandas' nullable dtypes (``Float64``, ``Int64``, ``boolean``)
  and ``category``. scikit-learn works on numpy, and ``pd.NA`` does not
  convert to ``np.nan`` on its own, so the frame is cast explicitly before
  it reaches a transformer.
* Missing values here mean different things. A lapped car has no numeric
  gap to the leader, a first lap has no previous lap, and a speed trap
  sometimes does not report — none of those are the same as a value of
  zero. They are imputed with a training median so a model can consume
  them, and an indicator column records that the value was absent, so the
  fact of the absence is not thrown away.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_CATEGORY = "__MISSING__"

NUMERIC_BLOCK = "numeric"
CATEGORICAL_BLOCK = "categorical"


def prepare_matrix(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Cast the feature columns into something scikit-learn accepts.

    Nullable integers, nullable floats and nullable booleans all become
    ``float64`` with ``np.nan`` for missing; categoricals become plain
    strings. Nothing is filled in here — that is the imputer's job, and the
    imputer must learn its values from the training split.
    """
    prepared = pd.DataFrame(index=frame.index)

    for column in numeric:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            # A nullable boolean cannot go straight to float: convert the
            # missing marker first, then the True/False.
            values = values.astype("object").map({True: 1.0, False: 0.0})
        prepared[column] = pd.to_numeric(values, errors="coerce").astype("float64")

    for column in categorical:
        values = frame[column].astype("object")
        prepared[column] = values.where(values.notna(), None)

    return prepared


def build_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    numeric_imputation: str = "median",
    scale_numeric: bool = True,
    add_missing_indicators: bool = True,
    missing_category: str = MISSING_CATEGORY,
) -> ColumnTransformer:
    """Assemble the transformer. Nothing is fitted yet.

    Categories unseen during training are encoded as all-zeros rather than
    raising: a driver who debuts in a later season, or a compound that
    appears for the first time, must not make the pipeline fail at the
    moment it is needed most.
    """
    numeric_steps: List[Any] = [
        ("impute", SimpleImputer(strategy=numeric_imputation, add_indicator=add_missing_indicators))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value=missing_category)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if numeric:
        transformers.append((NUMERIC_BLOCK, Pipeline(numeric_steps), list(numeric)))
    if categorical:
        transformers.append((CATEGORICAL_BLOCK, Pipeline(categorical_steps), list(categorical)))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transform_to_frame(preprocessor: ColumnTransformer, matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor and keep the column names."""
    if matrix.empty:
        return pd.DataFrame(columns=list(preprocessor.get_feature_names_out()))
    values = preprocessor.transform(matrix)
    return pd.DataFrame(values, columns=list(preprocessor.get_feature_names_out()), index=matrix.index)


def learned_parameters(preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Everything the transformer estimated from data.

    Used by the leakage guards: refitting on a different set of rows and
    comparing these values is what turns "we fitted on train" from a claim
    into something that can be checked.
    """
    parameters: Dict[str, Any] = {}

    numeric_pipeline = _block(preprocessor, NUMERIC_BLOCK)
    if numeric_pipeline is not None:
        imputer = numeric_pipeline.named_steps.get("impute")
        if imputer is not None:
            parameters["numeric_imputer_statistics"] = _round(imputer.statistics_)
            indicator = getattr(imputer, "indicator_", None)
            if indicator is not None:
                parameters["numeric_indicator_features"] = [int(i) for i in indicator.features_]
        scaler = numeric_pipeline.named_steps.get("scale")
        if scaler is not None:
            parameters["numeric_scaler_mean"] = _round(scaler.mean_)
            parameters["numeric_scaler_scale"] = _round(scaler.scale_)

    categorical_pipeline = _block(preprocessor, CATEGORICAL_BLOCK)
    if categorical_pipeline is not None:
        encoder = categorical_pipeline.named_steps.get("encode")
        if encoder is not None:
            parameters["categories"] = [[str(value) for value in group] for group in encoder.categories_]

    return parameters


def _block(preprocessor: ColumnTransformer, name: str) -> Optional[Pipeline]:
    for block_name, transformer, _columns in preprocessor.transformers_:
        if block_name == name and transformer not in ("drop", "passthrough"):
            return transformer
    return None


def _round(values: Any, digits: int = 9) -> List[Optional[float]]:
    """Round so that a comparison is not defeated by floating point noise."""
    return [None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), digits)
            for value in np.asarray(values, dtype="float64")]


def describe(
    preprocessor: ColumnTransformer,
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    scale_numeric: bool,
    add_missing_indicators: bool,
    numeric_imputation: str,
) -> Dict[str, Any]:
    """The preprocessing summary that goes into the report."""
    names = list(preprocessor.get_feature_names_out())
    encoder_block = _block(preprocessor, CATEGORICAL_BLOCK)
    categories: Dict[str, List[str]] = {}
    if encoder_block is not None:
        encoder = encoder_block.named_steps["encode"]
        for column, group in zip(categorical, encoder.categories_):
            categories[column] = [str(value) for value in group]

    return {
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": f"constant '{MISSING_CATEGORY}'",
        "missing_indicators": add_missing_indicators,
        "scaling": "standard" if scale_numeric else "none",
        "categorical_encoding": "one-hot, handle_unknown='ignore'",
        "categories_learned_from_train": categories,
        "output_feature_count": len(names),
        "output_feature_names": names,
        "fitted_on": "train split only",
    }
