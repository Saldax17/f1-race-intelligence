"""Proper scores for a position matrix (math note §9).

All scores take the forecast rows in the same driver order as ``actual``
(1-based final positions under convention (a); ``None`` = not scored, e.g.
DSQ). Cars already retired at lap k are excluded by the caller: their rows
are deterministic and would deflate the score.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

EPS = 1e-4


def rps_rows(P: np.ndarray, actual: Sequence[Optional[int]]) -> np.ndarray:
    """Ranked probability score of each row; NaN where the row is not scored."""
    N = P.shape[1]
    out = np.full(P.shape[0], np.nan)
    F = np.cumsum(P, axis=1)
    for i, a in enumerate(actual):
        if a is None or not (1 <= a <= N):
            continue
        O = (np.arange(1, N + 1) >= a).astype(float)
        out[i] = float(np.sum((F[i, : N - 1] - O[: N - 1]) ** 2) / (N - 1))
    return out


def row_log_score(P: np.ndarray, actual: Sequence[Optional[int]], floor: float = EPS) -> np.ndarray:
    """log P[i, actual_i], floored so an unseen cell is never −∞."""
    out = np.full(P.shape[0], np.nan)
    for i, a in enumerate(actual):
        if a is not None and 1 <= a <= P.shape[1]:
            out[i] = float(np.log(max(P[i, a - 1], floor)))
    return out


def pairwise_log_loss(ahead: np.ndarray, actual: Sequence[Optional[int]], floor: float = EPS) -> float:
    """Mean −log Pr(realised i-vs-j) over scored pairs."""
    losses = []
    n = len(actual)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = actual[i], actual[j]
            if a is None or b is None:
                continue
            p = ahead[i, j] if a < b else ahead[j, i]
            losses.append(-np.log(min(max(p, floor), 1 - floor)))
    return float(np.mean(losses)) if losses else float("nan")


def event_probabilities(P: np.ndarray) -> Dict[str, np.ndarray]:
    return {"win": P[:, 0], "podium": P[:, :3].sum(axis=1), "points": P[:, :10].sum(axis=1)}


def calibration_table(
    probabilities: Iterable[float], outcomes: Iterable[bool], bins: Sequence[float] = (0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0)
) -> List[Dict[str, float]]:
    """Reliability diagram: mean forecast vs observed frequency per bin."""
    p = np.asarray(list(probabilities), float)
    y = np.asarray(list(outcomes), float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & ((p < hi) | (hi == bins[-1]))
        if mask.sum():
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(mask.sum()), "forecast": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return rows


def brier(probabilities: Iterable[float], outcomes: Iterable[bool]) -> float:
    p = np.asarray(list(probabilities), float)
    y = np.asarray(list(outcomes), float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def paired_bootstrap(deltas_by_race: Mapping[int, float], draws: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean race-level difference and its bootstrap 95% interval (the race is the unit)."""
    values = np.array([v for v in deltas_by_race.values() if np.isfinite(v)])
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "races": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, (draws, len(values))).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "races": int(len(values)),
    }
