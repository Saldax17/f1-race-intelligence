"""The two baselines the model must beat (math note §10, scope phase 1).

* ``persist``: the order at lap k stands, after moving each car that still
  owes its mandatory stop back by the pit loss. Degenerate (0/1 cells), so it
  is compared on RPS only.
* ``transition``: Pr(final = b | position a at lap k), counted per k-bucket
  on training races and smoothed; Sinkhorn makes it doubly stochastic for
  the field size of the race at hand.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from f1_race_intelligence.race_model.forecast import CarState


def order_at_k(cars: Sequence[CarState], penalty: Optional[np.ndarray] = None) -> List[int]:
    """Indices of ``cars`` in running order; retired cars last, earliest retirement last."""
    penalty = np.zeros(len(cars)) if penalty is None else penalty
    running = [i for i, c in enumerate(cars) if not c.retired]
    retired = [i for i, c in enumerate(cars) if c.retired]
    running.sort(key=lambda i: (-cars[i].laps_done, cars[i].banked + penalty[i]))
    retired.sort(key=lambda i: (-cars[i].laps_done, cars[i].banked))
    return running + retired


def permutation_matrix(order: Sequence[int]) -> np.ndarray:
    n = len(order)
    P = np.zeros((n, n))
    for place, i in enumerate(order):
        P[i, place] = 1.0
    return P


def persist_baseline(cars: Sequence[CarState], owed: Sequence[bool], pit_loss: float) -> np.ndarray:
    penalty = np.array([pit_loss if o else 0.0 for o in owed])
    # banked is ≤ 0 (seconds before the cutoff): a stop adds time, i.e. moves the car back
    order = order_at_k(cars, penalty)
    return permutation_matrix(order)


def sinkhorn(Q: np.ndarray, iterations: int = 200, tol: float = 1e-9) -> np.ndarray:
    Q = np.clip(Q, 1e-12, None)
    for _ in range(iterations):
        Q = Q / Q.sum(axis=1, keepdims=True)
        Q = Q / Q.sum(axis=0, keepdims=True)
        if np.abs(Q.sum(axis=1) - 1).max() < tol:
            break
    return Q / Q.sum(axis=1, keepdims=True)


class TransitionBaseline:
    """Pooled position-at-k → final-position counts, one matrix per k-bucket."""

    def __init__(self, size: int = 22, alpha: float = 0.5) -> None:
        self.size = size
        self.alpha = alpha
        self.counts: Dict[str, np.ndarray] = {}

    def add(self, bucket: str, position_at_k: Sequence[int], final: Sequence[Optional[int]]) -> None:
        C = self.counts.setdefault(bucket, np.zeros((self.size, self.size)))
        for a, b in zip(position_at_k, final):
            if b is not None and 1 <= a <= self.size and 1 <= b <= self.size:
                C[a - 1, b - 1] += 1

    def matrix(self, bucket: str, n: int) -> np.ndarray:
        C = self.counts.get(bucket, np.zeros((self.size, self.size)))[:n, :n] + self.alpha
        return sinkhorn(C)

    def predict(self, bucket: str, order: Sequence[int]) -> np.ndarray:
        """Rows in car order, given the cars' running order at lap k."""
        n = len(order)
        Q = self.matrix(bucket, n)
        P = np.zeros((n, n))
        for place, i in enumerate(order):
            P[i] = Q[place]
        return P

    def to_dict(self) -> Dict[str, object]:
        return {"size": self.size, "alpha": self.alpha, "counts": {k: v.tolist() for k, v in self.counts.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TransitionBaseline":
        out = cls(int(data["size"]), float(data["alpha"]))
        out.counts = {k: np.asarray(v, float) for k, v in dict(data["counts"]).items()}
        return out


def ahead_from_matrix(P: np.ndarray) -> np.ndarray:
    """Pairwise Pr(i ahead of j) implied by independent rows (used for baselines only)."""
    F = np.cumsum(P, axis=1)
    n = P.shape[0]
    ahead = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Pr(pos_i < pos_j) = Σ_a P[i,a] Pr(pos_j > a)
                ahead[i, j] = float(np.sum(P[i] * (1 - F[j])))
    return ahead
