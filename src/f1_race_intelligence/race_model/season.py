"""Level 2: car and driver form across races, plus qualifying (math note §3.3 x–xi).

Race-r pace of driver i in team t, in % of the field-median lap:

    θ_ir = γ_tr + δ_ir + c_tr + w_ir
    γ_tr = ρ γ_t,r−1 + ξ,  ξ ~ N(0, q_γ²)   (q_γ,season across a winter)
    δ_ir = δ_i,r−1 + ξ',   ξ' ~ N(0, q_δ²)  (travels with the driver)
    c_tr ~ N(0, s_c²)  team × circuit fit, redrawn each race
    w_ir ~ N(0, s_w²)  race-day transient, seen only in the laps

On race weekend r the filter (1) predicts, (2) reads qualifying,
``q_ir = λ(γ + δ + c) + u`` with ``u ~ N(0, σ_q²)``, (3) hands the pre-race
prior ``N(Hx, HPHᵀ + s_w² I)`` to the in-race model, and (4) after the race
reads the full-race θ̂ with noise ``s_w² + SE²``. Hyperparameters are fitted by
the Kalman prediction-error likelihood on training races, skipping a burn-in.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOG2PI = float(np.log(2 * np.pi))


@dataclass
class SeasonParams:
    rho_in: float = 1.0
    rho_season: float = 1.0
    q_gamma: float = 0.071
    q_gamma_season: float = 0.21
    q_delta: float = 0.017
    s_c: float = 0.23
    s_w: float = 0.21
    sigma_q: float = 0.88
    lam: float = 0.97
    p0_gamma: float = 0.25
    """Initial variance of a team never seen before (≈ field spread²)."""
    p0_delta: float = 0.05
    """Initial variance of a rookie's driver effect."""

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeasonParams":
        return cls(**{k: float(data[k]) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Prior:
    """Pre-race prior for θ of the listed drivers (% of a lap)."""

    drivers: List[str]
    mean: np.ndarray
    cov: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {"drivers": list(self.drivers), "mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Prior":
        return cls(list(data["drivers"]), np.asarray(data["mean"], float), np.asarray(data["cov"], float))

    def subset(self, drivers: Sequence[str], fallback_var: float) -> "Prior":
        """The prior of ``drivers``; anyone unknown gets mean 0 and ``fallback_var``."""
        index = {d: i for i, d in enumerate(self.drivers)}
        n = len(drivers)
        mean, cov = np.zeros(n), np.eye(n) * fallback_var
        known = [(a, index[d]) for a, d in enumerate(drivers) if d in index]
        if known:
            rows, src = zip(*known)
            mean[list(rows)] = self.mean[list(src)]
            cov[np.ix_(rows, rows)] = self.cov[np.ix_(src, src)]
        return Prior(list(drivers), mean, cov)


@dataclass
class RaceObservation:
    """One race as level 2 sees it: who drove for whom, qualifying, and the full-race θ̂."""

    year: int
    drivers: List[str]
    teams: List[str]
    quali: np.ndarray
    theta_hat: np.ndarray
    se: np.ndarray
    key: int = 0


class SeasonFilter:
    """The persistent state (γ per team, δ per driver) and the race-weekend steps."""

    def __init__(self, params: SeasonParams, quali_max_abs_pct: float = 4.0) -> None:
        self.params = params
        self.quali_max = quali_max_abs_pct
        self.teams: List[str] = []
        self.drivers: List[str] = []
        self.x = np.zeros(0)
        self.P = np.zeros((0, 0))
        self.last_year: Optional[int] = None
        self.races_seen: List[int] = []

    # -- state bookkeeping ------------------------------------------------

    @property
    def _T(self) -> int:
        return len(self.teams)

    def _grow(self, teams: Sequence[str], drivers: Sequence[str]) -> None:
        new_t = [t for t in dict.fromkeys(teams) if t not in self.teams]
        new_d = [d for d in dict.fromkeys(drivers) if d not in self.drivers]
        if not new_t and not new_d:
            return
        T, D = self._T, len(self.drivers)
        T2, D2 = T + len(new_t), D + len(new_d)
        x = np.zeros(T2 + D2)
        P = np.zeros((T2 + D2, T2 + D2))
        old = list(range(T)) + list(range(T2, T2 + D))
        x[old] = self.x
        P[np.ix_(old, old)] = self.P
        for j in range(T, T2):
            P[j, j] = self.params.p0_gamma
        for j in range(T2 + D, T2 + D2):
            P[j, j] = self.params.p0_delta
        self.teams += new_t
        self.drivers += new_d
        self.x, self.P = x, P

    def _augmented(self, teams: Sequence[str], drivers: Sequence[str]):
        """State plus a fresh c per team in this race; H maps it to θ."""
        self._grow(teams, drivers)
        race_teams = list(dict.fromkeys(teams))
        n_state = len(self.x)
        m = len(race_teams)
        x = np.concatenate([self.x, np.zeros(m)])
        P = np.zeros((n_state + m, n_state + m))
        P[:n_state, :n_state] = self.P
        P[n_state:, n_state:] = np.eye(m) * self.params.s_c**2
        H = np.zeros((len(drivers), n_state + m))
        T = self._T
        for r, (team, driver) in enumerate(zip(teams, drivers)):
            H[r, self.teams.index(team)] = 1.0
            H[r, T + self.drivers.index(driver)] = 1.0
            H[r, n_state + race_teams.index(team)] = 1.0
        return x, P, H

    # -- the race-weekend steps -----------------------------------------------

    def predict(self, year: int) -> None:
        p = self.params
        if self.last_year is None:
            self.last_year = year
            return
        rho, q = (p.rho_in, p.q_gamma) if year == self.last_year else (p.rho_season, p.q_gamma_season)
        self.last_year = year
        T = self._T
        F = np.eye(len(self.x))
        F[:T, :T] *= rho
        self.x = F @ self.x
        self.P = F @ self.P @ F.T
        self.P[:T, :T] += np.eye(T) * q**2
        self.P[T:, T:] += np.eye(len(self.drivers)) * p.q_delta**2

    @staticmethod
    def _update(x, P, H, y, R):
        S = H @ P @ H.T + R
        S_inv = np.linalg.inv(S)
        v = y - H @ x
        K = P @ H.T @ S_inv
        ll = -0.5 * (v @ S_inv @ v + np.linalg.slogdet(S)[1] + len(v) * LOG2PI)
        return x + K @ v, P - K @ H @ P, float(ll)

    def weekend(
        self, obs: RaceObservation, update_race: bool = True
    ) -> Tuple[Prior, float, float]:
        """Run steps 1–4 for one race. Returns the pre-race prior and the two log-likelihoods."""
        p = self.params
        self.predict(obs.year)
        x, P, H = self._augmented(obs.teams, obs.drivers)
        ll_q = ll_r = 0.0

        q = obs.quali
        mq = np.isfinite(q) & (np.abs(q) <= self.quali_max)
        if mq.any():
            x, P, ll_q = self._update(x, P, p.lam * H[mq], q[mq], np.eye(int(mq.sum())) * p.sigma_q**2)

        prior = Prior(
            list(obs.drivers),
            H @ x,
            H @ P @ H.T + np.eye(len(obs.drivers)) * p.s_w**2,
        )

        mr = np.isfinite(obs.theta_hat) & np.isfinite(obs.se)
        if update_race and mr.any():
            R = np.diag(p.s_w**2 + obs.se[mr] ** 2)
            x, P, ll_r = self._update(x, P, H[mr], obs.theta_hat[mr], R)

        n = len(self.x)
        self.x, self.P = x[:n], (P[:n, :n] + P[:n, :n].T) / 2
        self.races_seen.append(obs.key)
        return prior, ll_q, ll_r

    def preview(self, year: int, teams: Sequence[str], drivers: Sequence[str], quali: np.ndarray) -> Prior:
        """The pre-race prior for an upcoming race, without changing the state."""
        clone = copy.deepcopy(self)
        prior, _, _ = clone.weekend(
            RaceObservation(year, list(drivers), list(teams), np.asarray(quali, float),
                            np.full(len(drivers), np.nan), np.full(len(drivers), np.nan)),
            update_race=False,
        )
        return prior

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params.to_dict(),
            "quali_max_abs_pct": self.quali_max,
            "teams": self.teams,
            "drivers": self.drivers,
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "last_year": self.last_year,
            "races_seen": self.races_seen,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeasonFilter":
        f = cls(SeasonParams.from_dict(data["params"]), float(data.get("quali_max_abs_pct", 4.0)))
        f.teams, f.drivers = list(data["teams"]), list(data["drivers"])
        f.x, f.P = np.asarray(data["x"], float), np.asarray(data["P"], float).reshape(len(data["x"]), -1)
        f.last_year = data.get("last_year")
        f.races_seen = list(data.get("races_seen", []))
        return f


# -- building observations, replaying, fitting -----------------------------------


def observations(obs: pd.DataFrame) -> List[RaceObservation]:
    """Level-1 rows grouped by race, in calendar order; drivers keyed by acronym."""
    out = []
    frame = obs[obs["team"].notna()].copy()
    frame["driver_id"] = frame["name"].fillna(frame["driver"].astype(str)).astype(str)
    for (_, key), g in frame.sort_values(["date", "session_key"]).groupby(["date", "session_key"], sort=False):
        g = g.drop_duplicates("driver_id")
        out.append(
            RaceObservation(
                year=int(g["year"].iloc[0]),
                drivers=g["driver_id"].tolist(),
                teams=g["team"].astype(str).tolist(),
                quali=g["quali_pct"].to_numpy(float),
                theta_hat=g["theta_hat"].to_numpy(float),
                se=g["se"].to_numpy(float),
                key=int(key),
            )
        )
    return out


def replay(
    races: Sequence[RaceObservation],
    params: SeasonParams,
    quali_max_abs_pct: float = 4.0,
    likelihood_keys: Optional[set] = None,
) -> Tuple[SeasonFilter, Dict[int, Prior], float]:
    """Run the filter through ``races`` in order; collect every pre-race prior."""
    f = SeasonFilter(params, quali_max_abs_pct)
    priors: Dict[int, Prior] = {}
    ll = 0.0
    for race in races:
        prior, ll_q, ll_r = f.weekend(race)
        priors[race.key] = prior
        if likelihood_keys is None or race.key in likelihood_keys:
            ll += ll_q + ll_r
    return f, priors, ll


FREE = ("q_gamma", "q_gamma_season", "q_delta", "s_c", "s_w", "sigma_q", "lam")


def fit_season(
    races: Sequence[RaceObservation],
    train_keys: set,
    burn_in: int = 6,
    fit_rho: bool = False,
    base: Optional[SeasonParams] = None,
    quali_max_abs_pct: float = 4.0,
) -> Tuple[SeasonParams, float]:
    """Maximum prediction-error likelihood on training races after the burn-in.

    Validation and test races never enter the likelihood; the filter only
    runs up to the last training race while fitting.
    """
    base = base or SeasonParams()
    ordered_train = [r for r in races if r.key in train_keys]
    ll_keys = {r.key for r in ordered_train[burn_in:]}
    upto = races[: max(i for i, r in enumerate(races) if r.key in train_keys) + 1] if ordered_train else []
    names = list(FREE) + (["rho_in", "rho_season"] if fit_rho else [])

    def unpack(v: np.ndarray) -> SeasonParams:
        p = copy.deepcopy(base)
        for name, value in zip(names, v):
            if name.startswith("rho"):
                setattr(p, name, float(1 / (1 + np.exp(-value))))
            elif name == "lam":
                setattr(p, name, float(value))
            else:
                setattr(p, name, float(np.exp(value)))
        return p

    def pack(p: SeasonParams) -> np.ndarray:
        out = []
        for name in names:
            value = getattr(p, name)
            if name.startswith("rho"):
                value = min(max(value, 1e-3), 1 - 1e-3)
                out.append(np.log(value / (1 - value)))
            elif name == "lam":
                out.append(value)
            else:
                out.append(np.log(max(value, 1e-4)))
        return np.array(out)

    def objective(v: np.ndarray) -> float:
        try:
            return -replay(upto, unpack(v), quali_max_abs_pct, ll_keys)[2]
        except np.linalg.LinAlgError:
            return 1e12

    start = copy.deepcopy(base)
    if fit_rho:
        start.rho_in, start.rho_season = 0.95, 0.9
    result = minimize(objective, pack(start), method="L-BFGS-B", options={"maxiter": 300})
    result = minimize(objective, result.x, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3})
    return unpack(result.x), float(-result.fun)
