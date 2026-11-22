"""The in-race forecast: pace posterior at lap k, then the Monte Carlo position matrix.

Math note §3.2 (update), §4.3 (sampler), §5–7 (stops and events). Per scenario:

1. draw the pace of every running car jointly from its posterior (this carries
   the n²·v pace term and the teammate covariance exactly);
2. draw the first neutralisation after lap k (SC or red; an SC already out at
   lap k starts now), its length, a VSC, and each car's retirement lap;
3. place the owed stop at s* + noise, or take it cheaply under SC/VSC (free
   under a red flag);
4. accumulate time to the finish: banked time, pace × laps, tyre cost, stop
   losses and AR(1) lap noise; a neutralisation compresses the queue at the
   restart to ``g`` seconds per place (lapped cars rejoin the lead lap);
5. order finishers by total time, then retired cars by laps completed.

Averaging the permutation matrices gives ``P``, doubly stochastic for any S.
The no-event special case is Thurstonian and has a closed form for pairwise
probabilities (:func:`pairwise_closed_form`), which the tests use as an exact
check of the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from f1_race_intelligence.race_model.baselines import TransitionBaseline

from f1_race_intelligence.race_model.events import EventModel, next_compound, optimal_stop_lap
from f1_race_intelligence.race_model.pace import (
    Level1Config,
    TyreModel,
    clean_laps,
    fit_race_pace,
    tyre_prior_means,
    tyre_state,
)
from f1_race_intelligence.race_model.racedata import DRY_COMPOUNDS, RaceData
from f1_race_intelligence.race_model.regime import RED, SC, current_regime, regime_events
from f1_race_intelligence.race_model.season import Prior

NO_EVENT = 10**6


@dataclass
class ForecastConfig:
    scenarios: int = 4000
    restart_gap_seconds: float = 0.8
    pit_under_sc_probability: float = 0.7
    pit_under_vsc_probability: float = 0.4
    noise_scale: float = 1.5
    noise_scale_by_fraction: Optional[Dict[str, float]] = None
    """Per race phase: ``{"0.1": s, "0.5": s, ...}`` keyed by the k/L fraction it was
    chosen at; a forecast uses the nearest one. ``None`` uses ``noise_scale``."""
    pass_cost_seconds: float = 0.0
    """Overtaking friction (seconds per on-track place gained × circuit difficulty); 0 disables."""
    transition_shrinkage_by_fraction: Optional[Dict[str, float]] = None
    """Per race phase convex shrinkage weights ``w_k`` in ``[0, 1]`` towards the
    empirical transition baseline: ``P_blend = (1 - w_k) P_model + w_k P_transition``.
    Fitted strictly on training races with leave-one-race-out transition counts."""

    def noise_for(self, k: int, total_laps: int) -> float:
        if not self.noise_scale_by_fraction:
            return self.noise_scale
        here = k / max(total_laps, 1)
        nearest = min(self.noise_scale_by_fraction, key=lambda f: abs(float(f) - here))
        return float(self.noise_scale_by_fraction[nearest])

    def shrinkage_for(self, k: int, total_laps: int) -> Tuple[Optional[str], float]:
        if not self.transition_shrinkage_by_fraction:
            return None, 0.0
        here = k / max(total_laps, 1)
        nearest = min(self.transition_shrinkage_by_fraction, key=lambda f: abs(float(f) - here))
        bucket = f"{int(round(float(nearest) * 100))}%"
        weight = float(np.clip(float(self.transition_shrinkage_by_fraction[nearest]), 0.0, 1.0))
        return bucket, weight
    events: bool = True
    """False switches off SC/VSC/red, DNFs, extra stops and stop-loss noise (the Gaussian core)."""
    pooled_sigma_laps: float = 40.0
    """Weight, in clean laps, of the pooled σ_ε against the in-race estimate."""


@dataclass
class CarState:
    driver_number: int
    driver_id: str
    team: Optional[str]
    laps_done: int
    banked: float
    """End time of the last completed lap, seconds from the leader's lap-k crossing."""
    retired: bool
    compound: Optional[str]
    tyre_age: int
    used: List[str]
    stops: int


@dataclass
class ForecastResult:
    session_key: int
    race: str
    lap: int
    total_laps: int
    drivers: List[Dict[str, Any]]
    matrix: np.ndarray
    p_dnf: np.ndarray
    ahead: np.ndarray
    """``ahead[i, j] = Pr(i finishes ahead of j)``."""
    pace_mean: np.ndarray
    pace_sd: np.ndarray
    scenarios: int
    model_version: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    regime_at_k: str = "GREEN"

    def expected_position(self) -> np.ndarray:
        return self.matrix @ np.arange(1, self.matrix.shape[1] + 1)

    def to_dict(self) -> Dict[str, Any]:
        n = self.matrix.shape[1]
        points = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * max(0, n - 10))[:n]
        rows = []
        exp_pos = self.expected_position()
        for i, d in enumerate(self.drivers):
            rows.append(
                {
                    **d,
                    "p_position": [round(float(x), 6) for x in self.matrix[i]],
                    "p_win": round(float(self.matrix[i, 0]), 6),
                    "p_podium": round(float(self.matrix[i, :3].sum()), 6),
                    "p_points": round(float(self.matrix[i, :10].sum()), 6),
                    "p_dnf": round(float(self.p_dnf[i]), 6),
                    "expected_position": round(float(exp_pos[i]), 3),
                    "expected_points": round(float(self.matrix[i] @ points), 3),
                    "pace_mean_pct": round(float(self.pace_mean[i]), 4),
                    "pace_sd_pct": round(float(self.pace_sd[i]), 4),
                }
            )
        return {
            "session_key": self.session_key,
            "race": self.race,
            "lap": self.lap,
            "total_laps": self.total_laps,
            "regime_at_k": self.regime_at_k,
            "scenarios": self.scenarios,
            "model_version": self.model_version,
            "generated_at": self.generated_at,
            "drivers": rows,
            "matrix": [[round(float(x), 6) for x in row] for row in self.matrix],
            "ahead": [[round(float(x), 4) for x in row] for row in self.ahead],
        }


# -- state at lap k --------------------------------------------------------------


def car_states(race: RaceData, reference_seconds: float) -> List[CarState]:
    """Banked time, laps, tyres and retirement status of every car at the cutoff."""
    if race.truncated_at is None:
        raise ValueError("forecast needs a race truncated at lap k (truncate_at_lap); refusing to read the future")
    if race.result:
        raise ValueError("a truncated race must not carry the official result")
    laps = race.laps.dropna(subset=["t_end"])
    cutoff = laps["t_end"].max()
    last = laps.sort_values("lap_number").groupby("driver_number").tail(1).set_index("driver_number")
    tyres = tyre_state(race)
    out = []
    for d in race.driver_numbers:
        if d in last.index:
            row = last.loc[d]
            banked = (row["t_end"] - cutoff).total_seconds()
            done = int(row["lap_number"])
        else:
            banked, done = 0.0, 0
        # A car whose last crossing is more than two laps' time old has stopped.
        retired = done == 0 or banked < -2.0 * reference_seconds * 1.3
        compound, age = tyres.current.get(d, (None, 0))
        out.append(
            CarState(
                driver_number=d,
                driver_id=str(race.drivers.get(d, {}).get("name_acronym") or d),
                team=race.team_of(d),
                laps_done=done,
                banked=float(banked),
                retired=bool(retired),
                compound=compound,
                tyre_age=int(age or 0),
                used=[c for c in tyres.used.get(d, []) if c in DRY_COMPOUNDS],
                stops=tyres.stops.get(d, 0),
            )
        )
    return out


def pace_posterior(
    race: RaceData,
    prior: Prior,
    running: Sequence[CarState],
    level1: Level1Config,
    phi: float,
    sigma_eps: float,
    tyres: TyreModel,
    pooled_sigma_laps: float = 40.0,
):
    """Prior ⊕ the laps so far: a Kalman update with the field-mean contrasts as the measurement."""
    ids = [c.driver_id for c in running]
    mean, cov = prior.mean.copy(), prior.cov.copy()
    clean = clean_laps(race, level1)
    if race.wet or clean.frame.empty:
        return mean, cov, clean
    numbers = {c.driver_number: j for j, c in enumerate(running)}
    frame = clean.frame[clean.frame["driver_number"].isin(numbers)]
    fit = fit_race_pace(frame, level1, phi=phi, prior_means=tyre_prior_means(tyres), min_laps_per_driver=2,
                        trim=len(frame) > 40)
    if fit is None or len(fit.drivers) < 2:
        return mean, cov, clean
    m = fit.laps_used
    blended = (m * fit.resid_sd**2 + pooled_sigma_laps * sigma_eps**2) / (m + pooled_sigma_laps)
    R = fit.cov * (blended / max(fit.resid_sd**2, 1e-9))
    n = len(fit.drivers)
    H = np.zeros((n, len(ids)))
    for r, d in enumerate(fit.drivers):
        H[r, numbers[d]] = 1.0
    H = (np.eye(n) - 1.0 / n) @ H
    H, z, R = H[:-1], fit.theta[:-1], R[:-1, :-1] + np.eye(n - 1) * 1e-8
    S = H @ cov @ H.T + R
    K = cov @ H.T @ np.linalg.inv(S)
    mean = mean + K @ (z - H @ mean)
    cov = cov - K @ H @ cov
    return mean, (cov + cov.T) / 2, clean


def _tyre_cost(offset: np.ndarray, slope: np.ndarray, age0: np.ndarray, laps: np.ndarray) -> np.ndarray:
    """Σ over ``laps`` laps of (o + d·age), ages age0+1 … age0+laps (% of a lap)."""
    laps = np.maximum(laps, 0)
    return laps * offset + slope * (age0 * laps + laps * (laps + 1) / 2.0)


def plan_stops(
    running: Sequence[CarState],
    total_laps: int,
    tyres: TyreModel,
    pit_loss: float,
    sec: float,
    wet: bool,
    bias: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Each car's remaining strategy under the degradation model (math note §5).

    A car that still owes its second dry compound always stops. Any other car
    stops once more if the tyre time it saves by changing at s* exceeds the pit
    loss — the same trade-off teams make, and the one that keeps an owed stop
    from looking like a free set of new tyres when everyone else's tyres keep
    ageing. The new compound is the cheapest legal one for the laps left.
    """
    def offset(c):
        return tyres.offset.get(c, 0.0) if c in DRY_COMPOUNDS else 0.0

    def slope(c):
        return tyres.slope.get(c, 0.0) if c in DRY_COMPOUNDS else 0.0

    M = len(running)
    out = {name: np.zeros(M) for name in ("o1", "d1", "o2", "d2", "o3", "d3", "stop_lap", "stop_lap2")}
    out["owed"] = np.zeros(M, bool)
    out["stops"] = np.zeros(M, bool)
    out["stops2"] = np.zeros(M, bool)
    out["compound"] = np.array([None] * M, dtype=object)
    L = total_laps
    for j, car in enumerate(running):
        k, age, c1 = car.laps_done, car.tyre_age, car.compound
        n = L - k
        out["o1"][j], out["d1"][j] = offset(c1), slope(c1)
        out["stop_lap"][j] = np.inf
        out["stop_lap2"][j] = np.inf
        if wet or c1 not in DRY_COMPOUNDS or n <= 1:
            continue
        owed = len(set(car.used)) < 2
        stay = _tyre_cost(offset(c1), slope(c1), age, n) * sec
        best1 = None
        candidates = [c for c in DRY_COMPOUNDS if not owed or c not in car.used and c != c1] or [next_compound(c1, car.used)]
        for c2 in candidates:
            s = optimal_stop_lap(slope(c1), slope(c2), L, age, k) + bias
            s = float(np.clip(np.round(s), k + 1, L - 1))
            cost = (_tyre_cost(offset(c1), slope(c1), age, s - k) + _tyre_cost(offset(c2), slope(c2), 0.0, L - s)) * sec + pit_loss
            if best1 is None or cost < best1[0]:
                best1 = (cost, c2, s)
        cost1, c2, s1 = best1
        best2 = None
        if n >= 3:
            d1_eff = max(slope(c1), 1e-3)
            for ca in DRY_COMPOUNDS:
                da_eff = max(slope(ca), 1e-3)
                for cb in DRY_COMPOUNDS:
                    if owed and (ca in car.used or ca == c1) and (cb in car.used or cb == c1):
                        continue
                    db_eff = max(slope(cb), 1e-3)
                    w = (n + age) / (1.0 / d1_eff + 1.0 / da_eff + 1.0 / db_eff)
                    t1 = float(np.clip(np.round(k + w / d1_eff - age + bias), k + 1, L - 2))
                    t2 = float(np.clip(np.round(L - w / db_eff + bias), t1 + 1, L - 1))
                    c_tyre = (
                        _tyre_cost(offset(c1), slope(c1), age, t1 - k)
                        + _tyre_cost(offset(ca), slope(ca), 0.0, t2 - t1)
                        + _tyre_cost(offset(cb), slope(cb), 0.0, L - t2)
                    )
                    cost2 = c_tyre * sec + 2.0 * pit_loss
                    if best2 is None or cost2 < best2[0]:
                        best2 = (cost2, ca, cb, t1, t2)
        out["owed"][j] = owed
        if best2 is not None and best2[0] < cost1 and (owed or best2[0] < stay):
            _, ca, cb, t1, t2 = best2
            out["o2"][j], out["d2"][j], out["compound"][j] = offset(ca), slope(ca), ca
            out["o3"][j], out["d3"][j] = offset(cb), slope(cb)
            out["stops"][j] = True
            out["stops2"][j] = True
            out["stop_lap"][j] = t1
            out["stop_lap2"][j] = t2
        else:
            out["o2"][j], out["d2"][j], out["compound"][j] = offset(c2), slope(c2), c2
            out["stops"][j] = owed or cost1 < stay
            out["stop_lap"][j] = s1
    return out


def forecast(
    race: RaceData,
    prior: Prior,
    total_laps: int,
    *,
    level1: Level1Config,
    phi: float,
    sigma_eps: float,
    tyres: TyreModel,
    events: EventModel,
    config: ForecastConfig,
    rng: Optional[np.random.Generator] = None,
    model_version: Optional[str] = None,
    fallback_prior_variance: float = 0.5,
    transition: Optional["TransitionBaseline"] = None,
) -> ForecastResult:
    """The N×N finishing-position matrix of ``race`` as known at its truncation lap."""
    rng = rng or np.random.default_rng(0)
    k = int(race.truncated_at)
    L = int(max(total_laps, k))
    loc = race.ref.location
    pre_clean = clean_laps(race, level1)
    ref = pre_clean.reference_seconds
    if not np.isfinite(ref):
        durations = race.laps["lap_duration"].dropna()
        ref = float(durations.median()) if len(durations) else 90.0
    cars = car_states(race, ref)
    N = len(cars)
    run_idx = [i for i, c in enumerate(cars) if not c.retired]
    running = [cars[i] for i in run_idx]
    M = len(running)

    sub = prior.subset([c.driver_id for c in running], fallback_prior_variance)
    mean, cov, _ = pace_posterior(race, sub, running, level1, phi, sigma_eps, tyres, config.pooled_sigma_laps)

    S = config.scenarios
    sec = ref / 100.0
    chol = np.linalg.cholesky(cov + np.eye(M) * 1e-9)
    theta = mean + rng.standard_normal((S, M)) @ chol.T
    pace = ref + sec * theta  # seconds per lap

    laps_done = np.array([c.laps_done for c in running], float)
    banked = np.array([c.banked for c in running], float)
    down = np.maximum(0.0, k - laps_done)
    remaining = np.maximum(0.0, L - laps_done)

    plan = plan_stops(running, L, tyres, events.pit(loc), sec, race.wet, events.stop_lap_bias)
    o1, d1, o2, d2, o3, d3 = plan["o1"], plan["d1"], plan["o2"], plan["d2"], plan["o3"], plan["d3"]
    age1 = np.array([c.tyre_age for c in running], float)
    owed = plan["owed"]

    # Stop lap (car's own lap index; inf = no stop). The tyre cost uses the
    # planned lap; the noisy timing only decides how a stop meets a safety car.
    # The spread of real stops around s* is mostly error of the s* model, not
    # strategy that costs tyre time, so it must not enter the tyre term.
    planned = np.where(plan["stops"], plan["stop_lap"], np.inf) * np.ones((S, 1))
    planned2 = np.where(plan["stops2"], plan["stop_lap2"], np.inf) * np.ones((S, 1))
    noise_sd = events.stop_lap_sd if config.events else 0.0
    stop = np.round(plan["stop_lap"] + noise_sd * rng.standard_normal((S, M)))
    stop = np.clip(stop, laps_done + 1, np.maximum(laps_done + 1, L - 1))
    stop = np.where(plan["stops"], stop, np.inf)
    loss = events.pit(loc) + (events.pit_loss_sd * rng.standard_normal((S, M)) if config.events else 0.0)
    stop_cost = np.where(np.isfinite(stop), loss, 0.0)
    has_extra = np.broadcast_to(plan["stops2"][None, :], (S, M)).copy()
    extra_lap = np.round(plan["stop_lap2"] + noise_sd * rng.standard_normal((S, M)))
    extra_min = np.where(np.isfinite(stop), stop + 1, laps_done + 1)
    extra_lap = np.clip(extra_lap, extra_min, np.maximum(extra_min, L - 1))
    extra_lap = np.where(has_extra, extra_lap, np.inf)
    loss2 = events.pit(loc) + (events.pit_loss_sd * rng.standard_normal((S, M)) if config.events else 0.0)
    extra_cost = np.where(np.isfinite(extra_lap), loss2, 0.0)

    # neutralisations
    regime_now = current_regime(race.race_control)
    if config.events:
        h_sc, h_red, h_vsc = events.hazard(SC, loc), events.hazard(RED, loc), events.hazard("VSC", loc)
        h = h_sc + h_red
        first = k + rng.geometric(max(h, 1e-9), S).astype(float) if h > 0 else np.full(S, np.inf)
        is_red = rng.random(S) < (h_red / h if h > 0 else 0.0)
        if regime_now in (SC, RED):
            first[:] = k
            is_red[:] = regime_now == RED
        sc_len = np.asarray(events.sc_laps or [4], float)
        dur = np.where(is_red, 1.0, rng.choice(sc_len, S))
        if regime_now == SC:  # already out: how long has it run?
            started = max((e.start_lap for e in regime_events(race.race_control) if e.kind == SC and e.end_lap is None),
                          default=k)
            dur = np.maximum(1.0, dur - (k - started))
        first = np.where(first < L, first, np.inf)
        vsc = k + rng.geometric(max(h_vsc, 1e-9), S).astype(float) if h_vsc > 0 else np.full(S, np.inf)
        vsc = np.where(vsc < np.minimum(first, L), vsc, np.inf)
        h_dnf = events.dnf(loc, race.wet)
        retire_lap = k + rng.geometric(max(h_dnf, 1e-12), (S, M)).astype(float)
        retire_lap = np.where(retire_lap <= L, retire_lap, np.inf)
    else:
        first = np.full(S, np.inf)
        is_red = np.zeros(S, bool)
        dur = np.zeros(S)
        vsc = np.full(S, np.inf)
        retire_lap = np.full((S, M), np.inf)

    # cheap stops under VSC (no compression)
    # Event laps are compared on each car's own lap count. At the cutoff most
    # cars are part-way through lap k (laps_done = k-1) without being lapped,
    # so subtracting k - laps_done would put them a lap early. A car that is
    # genuinely lapped reaches lap e later, and so ranks behind, which is right.
    own_vsc = np.maximum(vsc[:, None], laps_done)
    u_vsc = rng.random((S, M))
    take_vsc = (
        np.isfinite(own_vsc) & np.isfinite(stop) & (stop > own_vsc)
        & (u_vsc < config.pit_under_vsc_probability)
    )
    take_vsc2 = (
        np.isfinite(own_vsc) & ~take_vsc & has_extra & np.isfinite(extra_lap) & (extra_lap > own_vsc)
        & (~np.isfinite(stop) | (stop <= own_vsc))
        & (u_vsc < config.pit_under_vsc_probability)
    )
    stop = np.where(take_vsc, own_vsc, stop)
    planned = np.where(take_vsc, own_vsc, planned)
    stop_cost = np.where(take_vsc, stop_cost * events.vsc_pit_factor, stop_cost)
    extra_lap = np.where(take_vsc2, own_vsc, extra_lap)
    planned2 = np.where(take_vsc2, own_vsc, planned2)
    extra_cost = np.where(take_vsc2, extra_cost * events.vsc_pit_factor, extra_cost)

    wet_mult = events.wet_noise_multiplier if (race.wet and config.events) else 1.0
    sigma_lap = config.noise_for(k, L) * wet_mult * sec * sigma_eps * np.sqrt((1 + phi) / (1 - phi))

    def tyre_upto(lap: np.ndarray, stop_lap: np.ndarray, stop_lap2: np.ndarray) -> np.ndarray:
        """Tyre cost (seconds) from the car's lap at k to its own lap ``lap``, changing at ``stop_lap`` and ``stop_lap2``."""
        lap = np.maximum(lap, laps_done)
        s1 = np.minimum(stop_lap, stop_lap2)
        s2 = np.maximum(stop_lap, stop_lap2)
        before = np.minimum(lap, s1) - laps_done
        mid = np.maximum(0.0, np.minimum(lap, s2) - s1)
        after = np.maximum(0.0, lap - s2)
        zeros = np.zeros_like(age1)
        return sec * (
            _tyre_cost(o1, d1, age1, before)
            + _tyre_cost(o2, d2, zeros, mid)
            + _tyre_cost(o3, d3, zeros, after)
        )

    finish = np.full((S, M), float(L))
    neutral = np.isfinite(first)
    # Random draws, drawn once so the totals below share them.
    eps_green = rng.standard_normal((S, M))
    eps_pre = rng.standard_normal((S, M))
    eps_post = rng.standard_normal((S, M))
    u_take = rng.random((S, M))

    def assemble(
        pace_arr: np.ndarray,
        noise: float,
        stop_arr: np.ndarray = stop,
        stop_cost_arr: np.ndarray = stop_cost,
        include_tyres: bool = True,
    ) -> np.ndarray:
        """Total time to the flag per scenario and car (green, or around one neutralisation)."""
        use_extra = has_extra if include_tyres else np.zeros((S, M), bool)
        e_cost = extra_cost if include_tyres else 0.0
        tyre_green = tyre_upto(finish, planned, planned2) if include_tyres else 0.0
        Z = (
            banked
            + remaining * pace_arr
            + tyre_green
            + stop_cost_arr
            + e_cost
            + noise * sigma_lap * np.sqrt(remaining) * eps_green
        )
        if not neutral.any():
            return Z
        # each car's own lap when the neutralisation starts (lapped cars are behind)
        e_own = np.where(neutral[:, None], np.maximum(first[:, None], laps_done), laps_done)
        n1 = e_own - laps_done
        tyre_pre = tyre_upto(e_own, planned, planned2) if include_tyres else 0.0
        pre = (
            banked
            + n1 * pace_arr
            + tyre_pre
            + np.where(stop_arr <= e_own, stop_cost_arr, 0.0)
            + np.where(use_extra & (extra_lap <= e_own), e_cost, 0.0)
            + noise * sigma_lap * np.sqrt(n1) * eps_pre
        )
        pending = np.isfinite(stop_arr) & (stop_arr > e_own)
        pending2 = ~pending & use_extra & np.isfinite(extra_lap) & (extra_lap > e_own)
        take = pending & (is_red[:, None] | (u_take < config.pit_under_sc_probability))
        take2 = pending2 & (is_red[:, None] | (u_take < config.pit_under_sc_probability))
        queue_cost = np.where(
            take,
            np.where(is_red[:, None], 0.0, stop_cost_arr * events.sc_pit_factor),
            np.where(take2, np.where(is_red[:, None], 0.0, e_cost * events.sc_pit_factor), 0.0),
        )
        alive = ~(retire_lap <= first[:, None])
        queue = np.where(alive, pre + queue_cost, np.inf)
        rank = np.argsort(np.argsort(queue, axis=1), axis=1).astype(float)
        # After the restart every car is on the lead lap (lapped cars wave by),
        # so all run the same n2 laps from a compressed queue.
        restart = np.minimum(first + dur, L)
        n2 = np.broadcast_to((L - restart)[:, None], (S, M))
        restart_own = np.maximum(restart[:, None], laps_done) * np.ones((1, M))
        stop_now = np.where(take, e_own, planned)
        stop2_now = np.where(take2, e_own, planned2)
        tyre_post = (
            tyre_upto(finish, stop_now, stop2_now) - tyre_upto(restart_own, stop_now, stop2_now)
            if include_tyres
            else 0.0
        )
        post = (
            config.restart_gap_seconds * rank
            + n2 * pace_arr
            + tyre_post
            + np.where(pending & ~take, stop_cost_arr, 0.0)
            + np.where(use_extra & (extra_lap > e_own) & ~take2, e_cost, 0.0)
            + noise * sigma_lap * np.sqrt(n2) * eps_post
        )
        return np.where(neutral[:, None], post, Z)

    Z = assemble(pace, 1.0)
    if config.events and config.pass_cost_seconds > 0.0 and M > 1:
        # Equal-pace reference keeping mandatory (owed) stops and SC queues only:
        # pit-cycle swaps from owed stops stay free, while on-track passes (from pace,
        # tyres or lap noise) pay circuit-scaled overtaking friction.
        equal_pace = np.full((S, M), ref)
        owed_stop = np.where(owed[None, :], stop, np.inf)
        owed_cost = np.where(owed[None, :], stop_cost, 0.0)
        Z_ref = assemble(equal_pace, 0.0, stop_arr=owed_stop, stop_cost_arr=owed_cost, include_tyres=False)
        rank_ref = np.argsort(np.argsort(Z_ref, axis=1), axis=1).astype(float)
        rank_raw = np.argsort(np.argsort(Z, axis=1), axis=1).astype(float)
        gained = np.maximum(0.0, rank_ref - rank_raw)
        Z = Z + (config.pass_cost_seconds * events.pass_difficulty(loc)) * gained
    # --- classification
    finished = ~np.isfinite(retire_lap)
    laps_completed = np.where(finished, L, retire_lap - 1 - down)
    key = np.where(finished, Z, 1e9 - laps_completed * 1e4 + rng.random((S, M)))
    full_key = np.empty((S, N))
    for j, i in enumerate(run_idx):
        full_key[:, i] = key[:, j]
    for i, c in enumerate(cars):
        if c.retired:  # already out: behind every running car, earliest retirement last
            full_key[:, i] = 2e9 - c.laps_done * 1e4 + c.banked * 1e-3
    order = np.argsort(full_key, axis=1)
    positions = np.empty_like(order)
    positions[np.arange(S)[:, None], order] = np.arange(N)[None, :]
    P = np.zeros((N, N))
    np.add.at(P, (np.tile(np.arange(N), S), positions.ravel()), 1.0)
    P /= S
    ahead = (positions[:, :, None] < positions[:, None, :]).mean(axis=0)

    if config.events and transition is not None:
        shrink_bucket, w = config.shrinkage_for(k, L)
        if w > 0.0 and shrink_bucket is not None and shrink_bucket in transition.counts:
            from f1_race_intelligence.race_model.baselines import pairwise_ahead_conditional

            P_trans = transition.predict_running(shrink_bucket, cars)
            P = (1.0 - w) * P + w * P_trans
            ahead = (1.0 - w) * ahead + w * pairwise_ahead_conditional(P_trans)

    p_dnf = np.ones(N)
    for j, i in enumerate(run_idx):
        p_dnf[i] = 1.0 - finished[:, j].mean()
    pace_mean = np.full(N, np.nan)
    pace_sd = np.full(N, np.nan)
    for j, i in enumerate(run_idx):
        pace_mean[i], pace_sd[i] = mean[j], np.sqrt(max(cov[j, j], 0))

    drivers = [
        {
            "driver_number": c.driver_number,
            "driver": c.driver_id,
            "team": c.team,
            "laps_done": c.laps_done,
            "gap_seconds": round(-c.banked, 3),
            "retired_at_k": c.retired,
            "compound": c.compound,
            "tyre_age": c.tyre_age,
            "owes_stop": bool(owed[run_idx.index(i)]) if i in run_idx else False,
        }
        for i, c in enumerate(cars)
    ]
    return ForecastResult(
        session_key=race.ref.session_key,
        race=race.tag,
        lap=k,
        total_laps=L,
        drivers=drivers,
        matrix=P,
        p_dnf=p_dnf,
        ahead=ahead,
        pace_mean=pace_mean,
        pace_sd=pace_sd,
        scenarios=S,
        model_version=model_version,
        regime_at_k=regime_now,
    )


def pairwise_closed_form(mean_total: np.ndarray, cov_total: np.ndarray) -> np.ndarray:
    """Pr(i ahead of j) = Φ(−E[Z_i − Z_j] / sd) for jointly Gaussian totals (§4.2)."""
    diff = mean_total[:, None] - mean_total[None, :]
    var = np.diag(cov_total)[:, None] + np.diag(cov_total)[None, :] - 2 * cov_total
    with np.errstate(divide="ignore", invalid="ignore"):
        out = norm.cdf(-diff / np.sqrt(var))
    np.fill_diagonal(out, 0.0)
    return out
