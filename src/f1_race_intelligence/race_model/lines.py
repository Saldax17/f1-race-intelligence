"""Racing lines from car position (OpenF1 ``location``) and telemetry (``car_data``).

Exploratory: nothing here feeds the forecast yet. It turns ~3.7 Hz x/y samples
into track-relative coordinates, so laps, cars and circuits can be compared:

* a **reference line** per race: the median path of the fastest clean laps,
  resampled every ``step`` units of arc length;
* for every sample: **progress** ``s`` along that line (distance from the
  line's start) and signed **lateral offset** ``d`` (left of travel positive);
* a **circuit profile** from the reference laps' speed/throttle/brake as a
  function of ``s``: lap length, full-throttle straights, braking zones,
  corners (speed minima).

Units: x/y are OpenF1's circuit coordinates. Their scale is checked against
the distance a car covers by integrating its speed (``scale_to_metres``),
so profiles are reported in metres.

Cleaning: x = y = z = 0 marks a car not yet on track (seen before the start);
``throttle``/``brake`` above 100 are sentinels (see the validation specs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from f1_race_intelligence.race_model.racedata import RaceData
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFileError
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

ALIGN_TOLERANCE = pd.Timedelta(milliseconds=300)


# -- loading ------------------------------------------------------------------


def load_samples(catalog: RawDataCatalog, race: RaceData, endpoint: str) -> pd.DataFrame:
    """Every per-driver file of ``endpoint`` (``location`` or ``car_data``) for one race."""
    records: List[Dict[str, Any]] = []
    for raw_file in catalog.discover([endpoint], year=race.ref.year, meeting_key=race.ref.meeting_key,
                                     session_key=race.ref.session_key):
        try:
            records.extend(catalog.load(raw_file).records)
        except RawFileError as exc:
            logger.warning("raw_file_unreadable", extra={"path": raw_file.uri, "error": str(exc)})
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], format="ISO8601", utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    frame["driver_number"] = frame["driver_number"].astype(int)
    return frame.dropna(subset=["date"])


def positions_with_telemetry(location: pd.DataFrame, car_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Clean x/y samples, each with the nearest telemetry sample within 300 ms."""
    pos = location[~((location["x"] == 0) & (location["y"] == 0) & (location["z"] == 0))]
    pos = pos[["date", "driver_number", "x", "y"]].astype({"x": float, "y": float}).sort_values("date")
    if car_data is None or car_data.empty:
        return pos.reset_index(drop=True)
    tel = car_data[(car_data["throttle"] <= 100) & (car_data["brake"] <= 100)]
    tel = tel[["date", "driver_number", "speed", "throttle", "brake", "drs"]].sort_values("date")
    merged = pd.merge_asof(pos, tel, on="date", by="driver_number", direction="nearest", tolerance=ALIGN_TOLERANCE)
    return merged.reset_index(drop=True)


def tag_laps(samples: pd.DataFrame, race: RaceData) -> pd.DataFrame:
    """Add ``lap_number`` to each sample from the laps' start and end times."""
    laps = race.laps.dropna(subset=["t_start", "t_end"])[["driver_number", "lap_number", "t_start", "t_end"]]
    samples = samples.assign(date=samples["date"].astype("datetime64[ns, UTC]"))
    tagged = pd.merge_asof(samples.sort_values("date"), laps.sort_values("t_start"), left_on="date",
                           right_on="t_start", by="driver_number", direction="backward")
    keep = tagged["lap_number"].notna() & (tagged["date"] < tagged["t_end"])
    tagged = tagged[keep].copy()
    tagged["lap_number"] = tagged["lap_number"].astype(int)
    tagged["t_lap"] = (tagged["date"] - tagged["t_start"]).dt.total_seconds()
    return tagged.drop(columns=["t_start", "t_end"]).reset_index(drop=True)


# -- the reference line ----------------------------------------------------------------


def resample_path(xy: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` units of arc length along a polyline."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.concatenate([[True], seg > 0])
    xy = xy[keep]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    grid = np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])])


@dataclass
class ReferenceLine:
    points: np.ndarray
    """(n, 2) points every ``step`` along the line, in lap order."""
    step: float
    laps_used: int

    @property
    def length(self) -> float:
        return float(len(self.points) * self.step)

    def project(self, xy: np.ndarray) -> Dict[str, np.ndarray]:
        """Progress ``s`` along the line and signed lateral offset ``d`` (left of travel positive)."""
        pts = self.points
        tangent = np.roll(pts, -1, axis=0) - pts
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
        idx = cKDTree(pts).query(xy)[1]
        rel = xy - pts[idx]
        along = (rel * tangent[idx]).sum(axis=1)
        normal = np.column_stack([-tangent[idx, 1], tangent[idx, 0]])
        return {"s": (idx * self.step + along) % self.length, "d": (rel * normal).sum(axis=1), "index": idx}


def reference_line(samples: pd.DataFrame, clean_laps: pd.DataFrame, n_laps: int = 10, step: float = 10.0) -> Optional[ReferenceLine]:
    """Median path of the ``n_laps`` fastest clean laps, aligned on the fastest one.

    ``clean_laps`` has driver_number, lap_number and lap_duration. Each lap's
    samples are resampled by arc length; the others are matched to the
    fastest lap's points by nearest neighbour, and the median taken.
    """
    fastest = clean_laps.nsmallest(n_laps, "lap_duration")
    paths = []
    for d, l in fastest[["driver_number", "lap_number"]].itertuples(index=False):
        lap = samples[(samples["driver_number"] == d) & (samples["lap_number"] == l)].sort_values("date")
        if len(lap) < 30:
            continue
        paths.append(resample_path(lap[["x", "y"]].to_numpy(float), step))
    if not paths:
        return None
    base = paths[0]
    stack = [base]
    for path in paths[1:]:
        stack.append(path[cKDTree(path).query(base)[1]])
    return ReferenceLine(points=np.median(np.stack(stack), axis=0), step=step, laps_used=len(paths))


def scale_to_metres(samples: pd.DataFrame, line: ReferenceLine) -> float:
    """Metres per coordinate unit: speed-integrated distance per lap over the line's length."""
    laps = samples.dropna(subset=["speed"]).groupby(["driver_number", "lap_number"])
    distances = []
    for _, lap in laps:
        lap = lap.sort_values("date")
        dt = lap["date"].diff().dt.total_seconds().fillna(0).to_numpy()
        if len(lap) < 30 or dt.max() > 2:
            continue
        distances.append(float((lap["speed"].to_numpy() / 3.6 * dt).sum()))
    if not distances:
        return float("nan")
    return float(np.median(distances) / line.length)


# -- circuit profile ------------------------------------------------------------------------


@dataclass
class CircuitProfile:
    location: str
    lap_metres: float
    full_throttle_share: float
    longest_straight_m: float
    straights_over_500m: int
    braking_zones: int
    heavy_braking_zones: int
    """Braking from above 250 km/h to below 150 km/h."""
    corners: int
    """Local speed minima below 250 km/h."""
    min_speed_kmh: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def circuit_profile(location: str, samples: pd.DataFrame, line: ReferenceLine, laps: pd.DataFrame,
                    metres_per_unit: float, bins: int = 400) -> CircuitProfile:
    """Speed/throttle/brake as a function of progress on the reference laps, summarised."""
    ref = samples.merge(laps[["driver_number", "lap_number"]], on=["driver_number", "lap_number"])
    ref = ref.dropna(subset=["speed", "throttle", "brake"])
    proj = line.project(ref[["x", "y"]].to_numpy(float))
    ref = ref.assign(s=proj["s"], b=(proj["s"] / line.length * bins).astype(int).clip(0, bins - 1))
    per = ref.groupby("b").agg(speed=("speed", "median"), throttle=("throttle", "median"),
                               brake=("brake", "mean")).reindex(range(bins)).interpolate(limit_direction="both")
    seg = line.length * metres_per_unit / bins
    full = (per["throttle"] >= 98).to_numpy()
    runs, run = [], 0
    for f in np.concatenate([full, full]):  # the lap wraps around
        run = run + 1 if f else 0
        runs.append(run)
    longest = min(max(runs), bins) * seg
    straights = 0
    run = 0
    for f in np.concatenate([full, [False]]):
        if f:
            run += 1
        else:
            straights += run * seg >= 500
            run = 0
    braking = (per["brake"] >= 0.5).to_numpy()
    starts = np.flatnonzero(braking & ~np.roll(braking, 1))
    speed = per["speed"].to_numpy()
    heavy = 0
    for a in starts:
        window = speed[[(a + j) % bins for j in range(-3, 15)]]
        heavy += window[:4].max() > 250 and window.min() < 150
    smooth = pd.Series(np.concatenate([speed[-5:], speed, speed[:5]])).rolling(5, center=True).mean().to_numpy()[5:-5]
    minima = (smooth < np.roll(smooth, 1)) & (smooth <= np.roll(smooth, -1)) & (smooth < 250)
    return CircuitProfile(
        location=location,
        lap_metres=float(line.length * metres_per_unit),
        full_throttle_share=float(full.mean()),
        longest_straight_m=float(longest),
        straights_over_500m=int(straights),
        braking_zones=int(len(starts)),
        heavy_braking_zones=int(heavy),
        corners=int(minima.sum()),
        min_speed_kmh=float(speed.min()),
        extra={"heavy_braking_s_m": [float(((a / bins) * line.length * metres_per_unit)) for a in starts
                                     if speed[[(a + j) % bins for j in range(-3, 4)]].max() > 250]},
    )


# -- on-track passes from progress ------------------------------------------------------------


STABLE_SECONDS = 10.0
"""A swap must hold this long, and not be preceded by the reverse swap this recently."""

PIT_LANE_OFFSET_M = 5.0
"""Lateral offset beyond which a sample is in the pit lane (the line is snapped; the lane sits ~15 m off)."""


def race_distance(samples: pd.DataFrame, line: ReferenceLine, metres_per_unit: float) -> pd.DataFrame:
    """Add ``s_m`` (progress, m), ``d_m`` (offset, m) and ``R`` (unwrapped race distance, m) per sample."""
    proj = line.project(samples[["x", "y"]].to_numpy(float))
    out = samples.assign(s_m=proj["s"] * metres_per_unit, d_m=proj["d"] * metres_per_unit)
    out = out.sort_values(["driver_number", "date"]).reset_index(drop=True)
    period = line.length * metres_per_unit
    R = np.empty(len(out))
    for _, idx in out.groupby("driver_number", sort=False).indices.items():
        phase = out["s_m"].to_numpy()[idx] / period * 2 * np.pi
        R[idx] = np.unwrap(phase) / (2 * np.pi) * period
    out["R"] = R
    return out


def on_track_passes(samples: pd.DataFrame, race: RaceData, lap_length_m: float, grid_seconds: float = 0.5,
                    max_gap_s: float = 2.0) -> pd.DataFrame:
    """Every change of order between two cars on track: when, on which lap, and where (``s_m``).

    Cars are compared on a common clock by interpolating race distance ``R``.
    A swap counts if both cars are out of the pit lane, on a green lap after
    lap 1, within half a lap of each other (a fight for position, not
    lapping), neither has a data gap longer than ``max_gap_s``, and it is not
    undone within 3 s (jitter, or a moment side by side).
    """
    from f1_race_intelligence.race_model.pace import tyre_state
    from f1_race_intelligence.race_model.regime import neutralised_laps

    last = int(race.laps["lap_number"].max()) if len(race.laps) else 0
    neutral = neutralised_laps(race.race_control, last) | {1}
    # Pit-lane positions are often snapped onto the track line too, so a car in
    # its box looks like a stopped car being passed: drop in- and out-laps.
    inout = tyre_state(race).inout
    t0 = samples["date"].min()
    samples = samples.assign(t=(samples["date"] - t0).dt.total_seconds())
    grid = np.arange(samples["t"].min(), samples["t"].max(), grid_seconds)
    tracks = {}
    for d, g in samples.groupby("driver_number"):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        continuous = np.interp(grid, t, np.r_[np.diff(t) <= max_gap_s, True].astype(float)) > 0.99
        inside = (grid >= t[0]) & (grid <= t[-1]) & continuous
        R = np.where(inside, np.interp(grid, t, g["R"].to_numpy()), np.nan)
        pit = np.interp(grid, t, (g["d_m"].abs() > PIT_LANE_OFFSET_M).astype(float)) > 0
        lap = np.interp(grid, t, g["lap_number"].to_numpy(float))
        s = np.interp(grid, t, g["s_m"].to_numpy())
        tracks[d] = (R, pit, lap, s)
    rows = []
    drivers = sorted(tracks)
    for i, a in enumerate(drivers):
        for b in drivers[i + 1:]:
            Ra, pa, la, sa = tracks[a]
            Rb, pb, lb, sb = tracks[b]
            diff = Ra - Rb
            valid = ~np.isnan(diff) & (np.abs(diff) < lap_length_m / 2) & ~pa & ~pb
            sign = np.sign(diff)
            for c in np.flatnonzero(valid[:-1] & valid[1:] & (sign[:-1] * sign[1:] < 0)):
                lap = int(round(max(la[c], lb[c])))
                if lap in neutral:
                    continue
                if any((car, l) in inout for car in (a, b) for l in (lap - 1, lap, lap + 1)):
                    continue
                passer, passed = (a, b) if diff[c + 1] > 0 else (b, a)
                rows.append({"race": race.tag, "location": race.ref.location, "t": float(grid[c]), "lap": lap,
                             "passer": int(passer), "passed": int(passed), "s_m": float((sa[c] + sb[c]) / 2)})
    passes = pd.DataFrame(rows)
    if passes.empty:
        return passes
    pair = passes[["passer", "passed"]].min(axis=1).astype(str) + "-" + passes[["passer", "passed"]].max(axis=1).astype(str)
    passes = passes.assign(pair=pair).sort_values(["pair", "t"])
    prev = passes.groupby("pair")["t"].shift(1)
    nxt = passes.groupby("pair")["t"].shift(-1)
    # Swaps undone within this window are positioning glitches (GPS loss in
    # tunnels: Monaco) or side-by-side moments, not passes.
    stable = ((passes["t"] - prev).fillna(np.inf) > STABLE_SECONDS) & ((nxt - passes["t"]).fillna(np.inf) > STABLE_SECONDS)
    return passes[stable].drop(columns="pair").sort_values("t").reset_index(drop=True)


def line_passes(race: RaceData) -> pd.DataFrame:
    """Ground truth from timing alone: net order changes between consecutive crossings of the line.

    For each green lap l after lap 1, take the cars that completed laps l−1 and l
    (none on an in- or out-lap around l) and count every pair whose order at the
    line flipped. A pass undone within the lap is invisible here, by design.
    """
    from f1_race_intelligence.race_model.pace import tyre_state
    from f1_race_intelligence.race_model.regime import neutralised_laps

    laps = race.laps.dropna(subset=["t_end"])
    if laps.empty:
        return pd.DataFrame(columns=["lap", "passer", "passed"])
    last = int(laps["lap_number"].max())
    neutral = neutralised_laps(race.race_control, last) | {1}
    inout = tyre_state(race).inout
    ends = laps.pivot_table(index="lap_number", columns="driver_number", values="t_end", aggfunc="first")
    rows = []
    for lap in range(2, last + 1):
        if lap in neutral or lap - 1 not in ends.index or lap not in ends.index:
            continue
        both = ends.loc[[lap - 1, lap]].dropna(axis=1)
        cars = [c for c in both.columns if not any((c, l) in inout for l in (lap - 1, lap, lap + 1))]
        before = both.loc[lap - 1, cars].sort_values().index.tolist()
        after = {c: i for i, c in enumerate(both.loc[lap, cars].sort_values().index)}
        for i, a in enumerate(before):
            for b in before[i + 1:]:
                if after[b] < after[a]:
                    rows.append({"lap": lap, "passer": int(b), "passed": int(a)})
    return pd.DataFrame(rows, columns=["lap", "passer", "passed"])
