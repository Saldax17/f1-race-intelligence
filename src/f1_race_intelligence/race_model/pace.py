"""Level 1: pace within one race, from clean laps (math note §2–3, §3.3 xi).

Lap times are expressed in percent of the race's median clean lap, so pace
transfers across circuits. For one race,

    y_il = lap effect_l + θ_i + o_c + d_c · age_il + ε_il,   ε AR(1) with φ

fitted by ridge least squares. The compound terms carry weak Gaussian priors
so that a compound run by only one or two cars cannot trade off against those
cars' pace (2023 Melbourne). Standard errors come from the coefficient
covariance, contrasted to the field mean and inflated by √((1+φ)/(1−φ)) for
the AR(1) errors.

The same fit serves two purposes: the full-race θ̂ that feeds the season
filter (level 2), and, on a race truncated at lap k, the in-race measurement
that updates the pace prior.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from f1_race_intelligence.race_model.racedata import DRY_COMPOUNDS, RaceData, finite
from f1_race_intelligence.race_model.regime import neutralised_laps

COMPOUND_ORDER = ("HARD", "MEDIUM", "SOFT")


@dataclass(frozen=True)
class Level1Config:
    free_air_seconds: float = 1.5
    slow_lap_factor: float = 1.04
    min_clean_laps: int = 5
    ridge_sigma: float = 0.4
    compound_offset_prior_sd: float = 1.0
    compound_slope_prior_sd: float = 0.1
    min_drivers: int = 8

    @classmethod
    def from_settings(cls, settings: Any) -> "Level1Config":
        return cls(**{name: getattr(settings, name) for name in cls.__dataclass_fields__ if hasattr(settings, name)})


@dataclass
class TyreModel:
    """Compound offsets (vs HARD) and degradation slopes, % of a lap per lap of age."""

    offset: Dict[str, float] = field(default_factory=lambda: {"HARD": 0.0, "MEDIUM": -0.3, "SOFT": -0.6})
    slope: Dict[str, float] = field(default_factory=lambda: {"HARD": 0.02, "MEDIUM": 0.04, "SOFT": 0.08})

    def to_dict(self) -> Dict[str, Any]:
        return {"offset": dict(self.offset), "slope": dict(self.slope)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TyreModel":
        return cls(offset=dict(data["offset"]), slope=dict(data["slope"]))


@dataclass
class TyreState:
    """Per car: compound and tyre age on every lap, in/out laps, compounds used."""

    compound: Dict[Tuple[int, int], str]
    age: Dict[Tuple[int, int], int]
    inout: Set[Tuple[int, int]]
    used: Dict[int, List[str]]
    stops: Dict[int, int]
    current: Dict[int, Tuple[Optional[str], int]]
    """driver → (compound, age) on the last lap of their latest stint."""


def tyre_state(race: RaceData) -> TyreState:
    compound: Dict[Tuple[int, int], str] = {}
    age: Dict[Tuple[int, int], int] = {}
    inout: Set[Tuple[int, int]] = set()
    used: Dict[int, List[str]] = defaultdict(list)
    stops: Dict[int, int] = {}
    current: Dict[int, Tuple[Optional[str], int]] = {}
    by_driver: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for stint in race.stints:
        if stint.get("driver_number") is not None:
            by_driver[int(stint["driver_number"])].append(stint)
    for driver, stints in by_driver.items():
        stints = sorted(stints, key=lambda s: s.get("stint_number") or 0)
        valid = [s for s in stints if s.get("lap_start") is not None]
        for j, stint in enumerate(valid):
            start = int(stint["lap_start"])
            end = int(stint["lap_end"]) if stint.get("lap_end") is not None else start
            base = int(stint.get("tyre_age_at_start") or 0)
            name = stint.get("compound")
            for lap in range(start, end + 1):
                compound[(driver, lap)] = name
                age[(driver, lap)] = base + lap - start
            if j > 0:
                inout.add((driver, start))
            if j < len(valid) - 1:
                inout.add((driver, end))
            if name and name not in used[driver]:
                used[driver].append(name)
            current[driver] = (name, base + end - start)
        stops[driver] = max(0, len(valid) - 1)
    for stop in race.pit:
        if stop.get("driver_number") is None or stop.get("lap_number") is None:
            continue
        inout.add((int(stop["driver_number"]), int(stop["lap_number"])))
        inout.add((int(stop["driver_number"]), int(stop["lap_number"]) + 1))
    return TyreState(compound, age, inout, dict(used), stops, current)


@dataclass
class CleanLaps:
    frame: pd.DataFrame
    """Clean laps: driver_number, lap_number, compound, age, y (% of ref)."""
    reference_seconds: float
    """Median clean lap time of the race, seconds (NaN if no clean laps)."""
    green_reference_seconds: float
    """Median of all green-flag laps (free air or not), seconds."""


def clean_laps(race: RaceData, config: Level1Config, tyres: Optional[TyreState] = None) -> CleanLaps:
    """Green, free-air, dry-tyre laps that are neither in- nor out-laps."""
    tyres = tyres or tyre_state(race)
    laps = race.laps.dropna(subset=["lap_duration", "t_end"]).copy()
    empty = CleanLaps(pd.DataFrame(columns=["driver_number", "lap_number", "compound", "age", "y"]), np.nan, np.nan)
    if laps.empty:
        return empty
    last_lap = int(laps["lap_number"].max())
    neutral = neutralised_laps(race.race_control, last_lap)
    keys = list(zip(laps["driver_number"], laps["lap_number"]))
    laps["compound"] = [tyres.compound.get(key) for key in keys]
    laps["age"] = [tyres.age.get(key) for key in keys]
    laps["inout"] = [key in tyres.inout for key in keys]

    ok = (
        (laps["lap_number"] > 1)
        & ~laps["is_pit_out_lap"]
        & ~laps["inout"]
        & ~laps["lap_number"].isin(neutral)
        & laps["compound"].isin(DRY_COMPOUNDS)
        & laps["age"].notna()
    )
    green = laps[ok].copy()
    if green.empty:
        return empty
    ref = float(green["lap_duration"].median())
    lap_median = green.groupby("lap_number")["lap_duration"].median()
    slow = set(lap_median[lap_median > config.slow_lap_factor * ref].index)
    green = green[~green["lap_number"].isin(slow)]
    green_ref = float(green["lap_duration"].median()) if len(green) else ref

    iv = race.intervals
    if len(iv):
        merged = pd.merge_asof(
            green.sort_values("t_end"),
            iv.sort_values("date")[["date", "driver_number", "interval", "gap_to_leader"]],
            left_on="t_end",
            right_on="date",
            by="driver_number",
            direction="backward",
        )
        free = (merged["interval"] >= config.free_air_seconds) | (merged["gap_to_leader"] == 0)
        clean = merged[free.fillna(False)].copy()
    else:  # no interval feed: cannot tell traffic apart, keep the green laps
        clean = green.copy()
    if clean.empty:
        return CleanLaps(empty.frame, np.nan, green_ref)
    ref = float(clean["lap_duration"].median())
    clean["y"] = 100.0 * (clean["lap_duration"] / ref - 1.0)
    clean["age"] = clean["age"].astype(float)
    frame = clean[["driver_number", "lap_number", "compound", "age", "y"]].reset_index(drop=True)
    return CleanLaps(frame, ref, green_ref)


@dataclass
class RacePace:
    """Result of the per-race ridge fit."""

    drivers: List[int]
    theta: np.ndarray
    """Driver pace, centred on the field mean of the fitted drivers (% of a lap)."""
    cov: np.ndarray
    """Covariance of the field-mean contrasts, AR-inflated (% squared)."""
    n_clean: Dict[int, int]
    resid_sd: float
    compound_terms: Dict[str, float]
    phi_pairs: Tuple[np.ndarray, np.ndarray]
    laps_used: int

    def se(self) -> Dict[int, float]:
        return dict(zip(self.drivers, np.sqrt(np.clip(np.diag(self.cov), 0, None))))

    def theta_map(self) -> Dict[int, float]:
        return dict(zip(self.drivers, self.theta))


def _design(df: pd.DataFrame, config: Level1Config, prior_means: Optional[Mapping[str, float]]):
    X = pd.concat(
        [
            pd.get_dummies(df["lap_number"], prefix="L", dtype=float),
            pd.get_dummies(df["driver_number"], prefix="D", dtype=float),
        ],
        axis=1,
    )
    compounds = [c for c in COMPOUND_ORDER if c in set(df["compound"])]
    for c in compounds:
        indicator = (df["compound"] == c).astype(float)
        if c != compounds[0]:
            X["C_" + c] = indicator
        X["A_" + c] = indicator * df["age"]
    s0 = config.ridge_sigma
    penalty = np.array(
        [
            s0**2 / config.compound_offset_prior_sd**2
            if col.startswith("C_")
            else s0**2 / config.compound_slope_prior_sd**2
            if col.startswith("A_")
            else 1e-6
            for col in X.columns
        ]
    )
    prior = np.zeros(len(X.columns))
    if prior_means:
        base = compounds[0] if compounds else "HARD"
        for j, col in enumerate(X.columns):
            if col.startswith("A_"):
                prior[j] = prior_means.get(col, 0.0)
            elif col.startswith("C_"):
                name = col[2:]
                prior[j] = prior_means.get("O_" + name, 0.0) - prior_means.get("O_" + base, 0.0)
    return X, penalty, prior


def fit_race_pace(
    clean: pd.DataFrame,
    config: Level1Config,
    phi: float,
    prior_means: Optional[Mapping[str, float]] = None,
    min_laps_per_driver: int = 1,
    trim: bool = True,
) -> Optional[RacePace]:
    """Ridge fit of one race's clean laps. ``None`` if too little data."""
    counts = clean.groupby("driver_number").size()
    clean = clean[clean["driver_number"].isin(counts[counts >= min_laps_per_driver].index)]
    if clean.empty or clean["driver_number"].nunique() < 2:
        return None

    def solve(df: pd.DataFrame):
        X, penalty, prior = _design(df, config, prior_means)
        A = X.values.T @ X.values + np.diag(penalty)
        A_inv = np.linalg.inv(A)
        beta = A_inv @ (X.values.T @ df["y"].values + penalty * prior)
        resid = df["y"].values - X.values @ beta
        return X, beta, resid, A_inv

    X, beta, resid, A_inv = solve(clean)
    if trim and len(resid) > 10:
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if mad > 0:
            keep = np.abs(resid) <= 3 * mad
            clean = clean[keep].copy()
            X, beta, resid, A_inv = solve(clean)
    dof = max(1, len(clean) - np.linalg.matrix_rank(X.values))
    resid_sd = float(np.sqrt((resid**2).sum() / dof))

    cols = list(X.columns)
    drivers = sorted(int(c[2:]) for c in cols if c.startswith("D_"))
    idx = [cols.index(f"D_{d}") for d in drivers]
    raw = beta[idx]
    n = len(drivers)
    C = np.eye(n) - 1.0 / n
    inflation = (1 + phi) / (1 - phi)
    cov = inflation * resid_sd**2 * (C @ A_inv[np.ix_(idx, idx)] @ C.T)

    ordered = clean.assign(resid=resid).sort_values(["driver_number", "lap_number"])
    prev = ordered.groupby("driver_number")[["lap_number", "resid", "compound"]].shift(1)
    pair = (ordered["lap_number"] - prev["lap_number"] == 1) & (ordered["compound"] == prev["compound"])
    pairs = (prev.loc[pair, "resid"].to_numpy(float), ordered.loc[pair, "resid"].to_numpy(float))

    terms = {c: float(b) for c, b in zip(cols, beta) if c[:2] in ("C_", "A_")}
    return RacePace(
        drivers=drivers,
        theta=C @ raw,
        cov=cov,
        n_clean=clean.groupby("driver_number").size().astype(int).to_dict(),
        resid_sd=resid_sd,
        compound_terms=terms,
        phi_pairs=pairs,
        laps_used=len(clean),
    )


@dataclass
class Level1Result:
    """Full-race pace for every driver-race, plus the pooled lap-noise and tyre model."""

    obs: pd.DataFrame
    phi: float
    sigma_eps: float
    tyres: TyreModel
    tyres_by_location: Dict[str, Dict[str, Any]]
    reference_seconds: Dict[int, float]
    race_stats: List[Dict[str, Any]]


def estimate_phi(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], default: float = 0.47) -> float:
    a = np.concatenate([p[0] for p in pairs]) if pairs else np.array([])
    b = np.concatenate([p[1] for p in pairs]) if pairs else np.array([])
    if len(a) < 30:
        return default
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.9))


def pool_tyres(per_race: Sequence[Tuple[str, Dict[str, float]]]) -> Tuple[TyreModel, Dict[str, Dict[str, Any]]]:
    """Median slopes and HARD-relative offsets across races; per-location values shrunk to them."""
    slopes: Dict[str, List[float]] = defaultdict(list)
    offsets: Dict[str, List[float]] = defaultdict(list)
    by_loc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for location, terms in per_race:
        for name, value in terms.items():
            if name.startswith("A_"):
                slopes[name[2:]].append(value)
                by_loc[location]["A_" + name[2:]].append(value)
        if "A_HARD" in terms:  # offsets are relative to the first compound present: HARD
            for name, value in terms.items():
                if name.startswith("C_"):
                    offsets[name[2:]].append(value)
                    by_loc[location]["O_" + name[2:]].append(value)
    model = TyreModel()
    for c in COMPOUND_ORDER:
        if slopes.get(c):
            model.slope[c] = float(np.median(slopes[c]))
        if c != "HARD" and offsets.get(c):
            model.offset[c] = float(np.median(offsets[c]))
    model.offset["HARD"] = 0.0
    locations: Dict[str, Dict[str, Any]] = {}
    for location, values in by_loc.items():
        entry = {"offset": dict(model.offset), "slope": dict(model.slope)}
        for c in COMPOUND_ORDER:
            xs = values.get("A_" + c, [])
            if xs:  # one race weighs as much as the pooled value
                n = len(xs)
                entry["slope"][c] = float((n * np.mean(xs) + model.slope[c]) / (n + 1))
            os_ = values.get("O_" + c, [])
            if os_ and c != "HARD":
                n = len(os_)
                entry["offset"][c] = float((n * np.mean(os_) + model.offset[c]) / (n + 1))
        locations[location] = entry
    return model, locations


def tyre_prior_means(tyres: TyreModel) -> Dict[str, float]:
    means = {f"A_{c}": v for c, v in tyres.slope.items()}
    means.update({f"O_{c}": v for c, v in tyres.offset.items()})
    return means


def fit_level1(races: Sequence[RaceData], config: Level1Config, tyre_races: Optional[Set[int]] = None) -> Level1Result:
    """Full-race θ̂ and SE for every driver-race (flat prior: laps only, no double counting).

    ``tyre_races`` restricts the pooled tyre model to those session keys
    (the training races), so validation degradation never leaks into it.
    """
    from f1_race_intelligence.race_model.racedata import final_order

    fits: List[Tuple[RaceData, CleanLaps, Optional[RacePace]]] = []
    pairs = []
    for race in races:
        clean = clean_laps(race, config)
        fit = None
        if not race.wet and clean.frame["driver_number"].nunique() >= config.min_drivers:
            fit = fit_race_pace(clean.frame, config, phi=0.0)
        if fit is not None:
            pairs.append(fit.phi_pairs)
        fits.append((race, clean, fit))
    phi = estimate_phi(pairs)
    inflation = (1 + phi) / (1 - phi)

    rows, stats, tyre_terms, refs = [], [], [], {}
    for ridx, (race, clean, fit) in enumerate(fits):
        grid = {g["driver_number"]: g for g in race.grid if g.get("lap_duration")}
        q_med = np.median([g["lap_duration"] for g in grid.values()]) if grid else np.nan
        order = final_order(race.result) if race.result else None
        theta: Dict[int, float] = {}
        se: Dict[int, float] = {}
        if fit is not None:
            keep = [d for d in fit.drivers if fit.n_clean.get(d, 0) >= config.min_clean_laps]
            if keep:
                full = fit.theta_map()
                centre = float(np.median([full[d] for d in keep]))
                theta = {d: full[d] - centre for d in keep}
                se = {d: s * np.sqrt(inflation) for d, s in fit.se().items()}
            if tyre_races is None or race.ref.session_key in tyre_races:
                tyre_terms.append((race.ref.location, fit.compound_terms))
        refs[race.ref.session_key] = clean.reference_seconds
        for driver in race.driver_numbers:
            g = grid.get(driver)
            rows.append(
                dict(
                    ridx=ridx,
                    year=race.ref.year,
                    race=race.tag,
                    location=race.ref.location,
                    session_key=race.ref.session_key,
                    date=race.ref.date_start,
                    driver=driver,
                    name=race.acronym_of(driver),
                    team=race.team_of(driver),
                    wet=race.wet,
                    n_clean=int(fit.n_clean.get(driver, 0)) if fit else 0,
                    theta_hat=theta.get(driver, np.nan),
                    se=se.get(driver, np.nan) if driver in theta else np.nan,
                    quali_pct=100 * (g["lap_duration"] / q_med - 1) if g and np.isfinite(q_med) else np.nan,
                    grid=g.get("position") if g else None,
                    final_position=order.position.get(driver) if order else None,
                    retired=bool(order and driver in order.retired),
                )
            )
        stats.append(
            dict(
                race=race.tag,
                session_key=race.ref.session_key,
                wet=race.wet,
                clean_laps=int(len(clean.frame)),
                resid_sd=fit.resid_sd if fit else np.nan,
                reference_seconds=clean.reference_seconds,
            )
        )
    obs = pd.DataFrame(rows)
    tyres, by_location = pool_tyres(tyre_terms)
    sigma = finite(s["resid_sd"] for s in stats)
    return Level1Result(
        obs=obs,
        phi=phi,
        sigma_eps=float(np.median(sigma)) if len(sigma) else 0.4,
        tyres=tyres,
        tyres_by_location=by_location,
        reference_seconds=refs,
        race_stats=stats,
    )
