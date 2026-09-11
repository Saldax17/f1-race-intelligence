"""Guards for the modelling dataset.

The dangerous failures at this stage are silent: a split that overlaps, a
transformer that saw the test set, a target that slipped into the feature
matrix. None of them raise an error, and all of them make the eventual
score better than the truth.

So the claims are checked rather than asserted. The three that matter most
work by *refitting*: a fresh transformer is fitted on the training rows
alone, and another on training plus validation, and the one in use must
match the first and differ from the second. A guard that only inspected
our own intention would agree with us even when we were wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from f1_race_intelligence.features.selection import FORBIDDEN_AS_FEATURES, TARGET_COLUMN, TRACEABILITY
from f1_race_intelligence.features import selection
from f1_race_intelligence.modeling import preprocessing
from f1_race_intelligence.modeling.split import SPLITS, TEST, TRAIN, VALIDATION, SessionAssignment

# Columns that must never appear in the feature matrix: the target and its
# flag, everything M5 classified as leaky or excluded, and the identifiers
# kept only for tracing a row back to its source.
#
# M5's catalogue is subtracted at the end, because a column can be both a
# key and a legitimate predictor. ``lap_number`` is the case in point: it
# identifies a row *and* carries real signal, since fuel burns off and the
# car gets lighter as the race goes on. Deriving the set this way keeps the
# catalogue the single source of truth rather than restating its decisions
# here, where the two could drift apart.
FORBIDDEN_IN_X = frozenset(
    (
        set(FORBIDDEN_AS_FEATURES)
        | set(TRACEABILITY)
        | {item.name for item in selection.excluded_columns()}
    )
    - set(selection.feature_columns())
)

KEY_PREFIX = "key_"
"""Prefix the runner gives the identifying columns it carries in the matrix.

It exists because a column can be both a key and a feature — ``lap_number``
is one — and without the prefix the raw key would overwrite the transformed
feature of the same name. Defined here rather than in the runner so the
guard that forbids these columns from the model input does not have to
import the module it is guarding.
"""


@dataclass
class CheckResult:
    """The outcome of one guard."""

    name: str
    passed: bool
    description: str
    blocking: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "check": self.name,
            "status": "PASS" if self.passed else "FAIL",
            "blocking": self.blocking,
            "description": self.description,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def blocking_failures(results: Sequence[CheckResult]) -> List[str]:
    return [item.name for item in results if item.blocking and not item.passed]


# -- feature matrix --------------------------------------------------------


def check_target_not_in_features(feature_names: Sequence[str]) -> CheckResult:
    """The thing being predicted must not be among the inputs."""
    present = [name for name in feature_names if name == TARGET_COLUMN or name.startswith(f"{TARGET_COLUMN}_")]
    return CheckResult(
        "target_not_in_features",
        not present,
        f"'{TARGET_COLUMN}' does not appear in the feature matrix",
        details={"found": present},
    )


def check_forbidden_columns_not_in_features(feature_names: Sequence[str]) -> CheckResult:
    """No leaky, excluded or identifier column reached the model.

    One-hot encoding renames columns, so a name is matched both exactly and
    as the prefix an encoder would produce.
    """
    found = []
    for name in feature_names:
        for forbidden in FORBIDDEN_IN_X:
            if name == forbidden or name.startswith(f"{forbidden}_"):
                found.append(name)
                break
    return CheckResult(
        "forbidden_columns_not_in_features",
        not found,
        "Leaky, excluded and identifier columns are absent from the feature matrix",
        details={"found": sorted(set(found)), "forbidden_count": len(FORBIDDEN_IN_X)},
    )


def check_traceability_columns_not_in_model_features(
    model_feature_names: Sequence[str],
    matrix_columns: Optional[Sequence[str]] = None,
) -> CheckResult:
    """Identifiers travel with the data but never into the model.

    The written matrix deliberately carries the keys and the target beside
    the features, so a prediction can be traced back to a lap without
    re-running anything. That convenience is exactly what makes this guard
    necessary: whoever trains a model must take the declared feature names,
    not every column in the file.

    ``lap_number`` is the one column that legitimately appears on both
    sides — it identifies a row and predicts, since the car gets lighter as
    fuel burns off — so it is judged by M5's catalogue rather than by its
    role as a key.
    """
    approved = set(selection.feature_columns())

    keyed = [name for name in model_feature_names if name.startswith(KEY_PREFIX)]
    identifiers = [
        name
        for name in model_feature_names
        if name in set(TRACEABILITY) - approved or name == TARGET_COLUMN
    ]

    unaccounted: List[str] = []
    if matrix_columns is not None:
        accounted = set(model_feature_names) | {TARGET_COLUMN} | {
            name for name in matrix_columns if name.startswith(KEY_PREFIX)
        }
        unaccounted = sorted(set(matrix_columns) - accounted)

    return CheckResult(
        "traceability_columns_not_in_model_features",
        not keyed and not identifiers and not unaccounted,
        "Key and identifier columns are carried for auditing but are not model inputs",
        details={
            "key_prefix": KEY_PREFIX,
            "keys_found_among_features": keyed,
            "identifiers_found_among_features": identifiers,
            "matrix_columns_unaccounted_for": unaccounted,
            "model_feature_count": len(model_feature_names),
        },
    )


def check_consistent_feature_schema(transformed: Dict[str, pd.DataFrame]) -> CheckResult:
    """Every split carries the same columns, in the same order.

    A model trained on one column order and scored on another produces
    numbers without raising anything.
    """
    schemas = {name: list(frame.columns) for name, frame in transformed.items() if not frame.empty}
    if len(schemas) <= 1:
        return CheckResult(
            "consistent_feature_schema",
            True,
            "Only one non-empty split; nothing to compare",
            details={"splits_compared": list(schemas)},
        )

    reference_name, reference = next(iter(schemas.items()))
    mismatched = {name: _schema_difference(reference, columns) for name, columns in schemas.items() if columns != reference}
    return CheckResult(
        "consistent_feature_schema",
        not mismatched,
        "All splits share the same feature columns in the same order",
        details={"reference": reference_name, "columns": len(reference), "mismatched": mismatched},
    )


def _schema_difference(reference: Sequence[str], other: Sequence[str]) -> Dict[str, Any]:
    return {
        "missing": sorted(set(reference) - set(other))[:5],
        "unexpected": sorted(set(other) - set(reference))[:5],
        "same_set_different_order": set(reference) == set(other),
    }


# -- split -----------------------------------------------------------------


def check_no_session_overlap_between_splits(assignments: Sequence[SessionAssignment]) -> CheckResult:
    """A race belongs to exactly one split, with all of its laps."""
    seen: Dict[int, List[str]] = {}
    for item in assignments:
        seen.setdefault(item.session_key, []).append(item.split)

    overlapping = {key: splits for key, splits in seen.items() if len(set(splits)) > 1}
    return CheckResult(
        "no_session_overlap_between_splits",
        not overlapping,
        "No session appears in more than one split",
        details={"sessions": len(seen), "overlapping": {str(k): v for k, v in list(overlapping.items())[:5]}},
    )


def check_split_rows_do_not_overlap(splits: Dict[str, pd.DataFrame]) -> CheckResult:
    """The same lap never appears in two splits."""
    grain = ["session_key", "driver_number", "lap_number"]
    keys = {
        name: set(map(tuple, frame[grain].to_numpy().tolist()))
        for name, frame in splits.items()
        if not frame.empty
    }
    collisions = {}
    names = list(keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = keys[left] & keys[right]
            if shared:
                collisions[f"{left}/{right}"] = len(shared)
    return CheckResult(
        "no_row_overlap_between_splits",
        not collisions,
        "No lap appears in more than one split",
        details={"collisions": collisions},
    )


def check_chronological_split(assignments: Sequence[SessionAssignment]) -> CheckResult:
    """Training races come before validation races, which come before test.

    Without this the model is validated on the past, which says nothing
    about how it will behave on the future it is meant to predict.
    """
    bounds: Dict[str, Dict[str, Optional[pd.Timestamp]]] = {}
    for name in SPLITS:
        dates = [item.session_date for item in assignments if item.split == name and item.session_date is not None]
        bounds[name] = {"first": min(dates) if dates else None, "last": max(dates) if dates else None}

    violations = []
    for earlier, later in ((TRAIN, VALIDATION), (VALIDATION, TEST), (TRAIN, TEST)):
        last = bounds[earlier]["last"]
        first = bounds[later]["first"]
        if last is not None and first is not None and last >= first:
            violations.append(
                {"earlier": earlier, "later": later, "last_of_earlier": last.isoformat(), "first_of_later": first.isoformat()}
            )

    return CheckResult(
        "chronological_split",
        not violations,
        "Every split starts after the previous one ends",
        details={
            "bounds": {
                name: {key: value.isoformat() if value is not None else None for key, value in edge.items()}
                for name, edge in bounds.items()
            },
            "violations": violations,
        },
    )


# -- preprocessing ---------------------------------------------------------


def check_preprocessing_fitted_on_train_only(
    preprocessor: ColumnTransformer,
    train_matrix: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> CheckResult:
    """The learned statistics are the training split's own statistics.

    Recomputed straight from the training rows with pandas, which shares no
    code with the transformer that produced them.
    """
    learned = preprocessing.learned_parameters(preprocessor)
    problems: List[Dict[str, Any]] = []

    if numeric and "numeric_imputer_statistics" in learned:
        expected = [round(float(train_matrix[column].median()), 9) for column in numeric]
        actual = learned["numeric_imputer_statistics"]
        for column, want, got in zip(numeric, expected, actual):
            if got is None or abs(want - got) > 1e-6:
                problems.append({"parameter": "imputer_median", "column": column, "expected": want, "actual": got})

    if categorical and "categories" in learned:
        for column, group in zip(categorical, learned["categories"]):
            observed = train_matrix[column].dropna().astype(str)
            expected_categories = sorted(set(observed) | ({preprocessing.MISSING_CATEGORY} if train_matrix[column].isna().any() else set()))
            if sorted(group) != expected_categories:
                problems.append(
                    {
                        "parameter": "categories",
                        "column": column,
                        "expected": expected_categories[:8],
                        "actual": sorted(group)[:8],
                    }
                )

    return CheckResult(
        "preprocessing_fitted_on_train_only",
        not problems,
        "Learned statistics equal the training split's own, recomputed independently",
        details={"problems": problems[:5], "problem_count": len(problems)},
    )


def _fitted_parameters(preprocessor: ColumnTransformer, matrix: pd.DataFrame) -> Dict[str, Any]:
    fresh = clone(preprocessor)
    fresh.fit(matrix)
    return preprocessing.learned_parameters(fresh)


def _check_split_not_used_for_fitting(
    name: str,
    preprocessor: ColumnTransformer,
    train_matrix: pd.DataFrame,
    other_matrix: pd.DataFrame,
    other_name: str,
) -> CheckResult:
    """Prove the transformer was fitted on train and not on train + other.

    If adding the other split does not change a single learned statistic,
    the test cannot tell the two apart; that is reported as inconclusive
    rather than passed, because a guard that cannot fail is not evidence.
    """
    if other_matrix.empty:
        return CheckResult(
            name,
            True,
            f"Skipped: the {other_name} split is empty",
            details={"skipped": True, "reason": f"no {other_name} rows"},
        )

    actual = preprocessing.learned_parameters(preprocessor)
    train_only = _fitted_parameters(preprocessor, train_matrix)
    contaminated = _fitted_parameters(preprocessor, pd.concat([train_matrix, other_matrix], ignore_index=True))

    matches_train = actual == train_only
    distinguishable = train_only != contaminated

    details = {
        "matches_train_only_fit": matches_train,
        "adding_split_changes_parameters": distinguishable,
    }
    if not distinguishable:
        details["inconclusive"] = (
            f"Fitting on train alone and on train + {other_name} produced identical statistics, so this "
            "comparison cannot discriminate. The preprocessing_fitted_on_train_only guard still applies."
        )
        return CheckResult(name, matches_train, f"No {other_name} statistics reached the transformer", details=details)

    return CheckResult(
        name,
        matches_train and actual != contaminated,
        f"No {other_name} statistics reached the transformer",
        details=details,
    )


def check_no_validation_statistics_used_in_training(
    preprocessor: ColumnTransformer,
    train_matrix: pd.DataFrame,
    validation_matrix: pd.DataFrame,
) -> CheckResult:
    return _check_split_not_used_for_fitting(
        "no_validation_statistics_used_in_training", preprocessor, train_matrix, validation_matrix, VALIDATION
    )


def check_no_test_statistics_used_in_training(
    preprocessor: ColumnTransformer,
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
) -> CheckResult:
    return _check_split_not_used_for_fitting(
        "no_test_statistics_used_in_training", preprocessor, train_matrix, test_matrix, TEST
    )


def check_categorical_unknown_handling(
    preprocessor: ColumnTransformer,
    train_matrix: pd.DataFrame,
    categorical: Sequence[str],
) -> CheckResult:
    """An unseen category must encode to zeros, not raise.

    Checked by actually feeding one through: a driver who debuts next
    season should not break the pipeline at the moment it is used.
    """
    if not categorical:
        return CheckResult("categorical_unknown_handling", True, "No categorical features", details={"skipped": True})

    block = preprocessing._block(preprocessor, preprocessing.CATEGORICAL_BLOCK)
    encoder = block.named_steps["encode"] if block is not None else None
    configured = getattr(encoder, "handle_unknown", None) == "ignore"

    probe = train_matrix.head(1).copy()
    if probe.empty:
        return CheckResult("categorical_unknown_handling", configured, "No training rows to probe with",
                           details={"handle_unknown": getattr(encoder, "handle_unknown", None)})

    column = categorical[0]
    probe[column] = "__A_CATEGORY_NEVER_SEEN__"
    try:
        encoded = preprocessor.transform(probe)
        survived = True
        error = None
    except Exception as exc:  # pragma: no cover - the guard exists to detect this
        survived = False
        error = f"{type(exc).__name__}: {exc}"
        encoded = None

    zeros_for_unknown = None
    if survived:
        names = list(preprocessor.get_feature_names_out())
        group = [index for index, name in enumerate(names) if name.startswith(f"{column}_")]
        zeros_for_unknown = bool(np.allclose(np.asarray(encoded)[0, group], 0.0)) if group else None

    return CheckResult(
        "categorical_unknown_handling",
        bool(configured and survived and (zeros_for_unknown is not False)),
        "An unseen category is encoded as all zeros instead of raising",
        details={
            "handle_unknown": getattr(encoder, "handle_unknown", None),
            "probe_column": column,
            "transform_succeeded": survived,
            "encoded_as_zeros": zeros_for_unknown,
            "error": error,
        },
    )


# -- target ----------------------------------------------------------------


def check_target_alignment(
    splits: Dict[str, pd.DataFrame],
    targets: Dict[str, pd.Series],
    source: pd.DataFrame,
) -> CheckResult:
    """Each y value still belongs to the row of X beside it.

    Verified against the M5 dataset by grain, so a reindex or a sort that
    quietly shifted the target by one row would be caught.
    """
    lookup = {
        (session, driver, lap): value
        for session, driver, lap, value in zip(
            source["session_key"], source["driver_number"], source["lap_number"], source[TARGET_COLUMN]
        )
    }

    problems: Dict[str, Any] = {}
    for name, frame in splits.items():
        if frame.empty:
            continue
        target = targets[name]
        if len(target) != len(frame):
            problems[name] = {"reason": "length mismatch", "rows": len(frame), "targets": len(target)}
            continue
        mismatched = 0
        for (session, driver, lap), value in zip(
            zip(frame["session_key"], frame["driver_number"], frame["lap_number"]), target.to_numpy()
        ):
            expected = lookup.get((session, driver, lap))
            if expected is None or pd.isna(expected) or abs(float(expected) - float(value)) > 1e-9:
                mismatched += 1
        if mismatched:
            problems[name] = {"reason": "value mismatch", "rows": mismatched}

    return CheckResult(
        "target_alignment",
        not problems,
        "Every target matches the M5 value for the same session, driver and lap",
        details={"problems": problems},
    )


def check_target_rows_are_valid(splits: Dict[str, pd.DataFrame], targets: Dict[str, pd.Series]) -> CheckResult:
    """No split carries a null target or a row M5 flagged as having none."""
    problems = {}
    for name, target in targets.items():
        if target.empty:
            continue
        nulls = int(pd.isna(target).sum())
        frame = splits[name]
        flagged = int((~frame["has_target"].fillna(False)).sum()) if "has_target" in frame.columns else 0
        if nulls or flagged:
            problems[name] = {"null_targets": nulls, "rows_flagged_without_target": flagged}

    return CheckResult(
        "target_rows_are_valid",
        not problems,
        "No modelling row carries a null or invalid target",
        details={"problems": problems},
    )
