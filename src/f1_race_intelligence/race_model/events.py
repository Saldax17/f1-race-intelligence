"""Discrete events and pit stops, fitted on training races (math note §5–7).

* Safety car, VSC and red flag: a per-lap hazard on green laps from lap 2,
  partially pooled by circuit, ``h_c = (e_c + κ h̄) / (n_c + κ)``; durations
  from their empirical distribution.
* Retirements (unclassified cars, convention (a)): a per driver-lap hazard,
  pooled the same way, with one multiplier for wet races.
* Pit loss: time lost on the in- and out-laps against the field median of the
  same laps, so stops under a safety car are measured on the same footing;
  per circuit, shrunk to the pooled median. Its ratio under SC/VSC gives the
  cheap-stop factors.
* Stop timing: the degradation-optimal lap s* (§5.3), plus the empirical bias
  and spread of real one-stop races around it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from f1_race_intelligence.race_model.pace import TyreModel, tyre_state
from f1_race_intelligence.race_model.racedata import DRY_COMPOUNDS, RaceData, final_order, is_excluded, is_retirement
from f1_race_intelligence.race_model.regime import RED, SC, VSC, neutralised_laps, regime_by_lap, regime_events

HAZARD_POOLING_LAPS = 60.0
"""κ for the SC/VSC/red hazards: a circuit needs about one race of laps to move halfway."""
DNF_POOLING_DRIVER_LAPS = 2000.0
"""κ for retirements: about 1.5 races of driver-laps."""


@dataclass
class EventModel:
    sc_hazard: float = 0.012
    vsc_hazard: float = 0.008
    red_hazard: float = 0.002
    sc_by_location: Dict[str, float] = field(default_factory=dict)
    vsc_by_location: Dict[str, float] = field(default_factory=dict)
    red_by_location: Dict[str, float] = field(default_factory=dict)
    sc_laps: List[int] = field(default_factory=lambda: [3, 4, 4, 5, 5, 6])
    vsc_laps: List[int] = field(default_factory=lambda: [1, 2, 2, 3])
    dnf_hazard: float = 0.0015
    dnf_by_location: Dict[str, float] = field(default_factory=dict)
    wet_dnf_multiplier: float = 1.5
    pit_loss: float = 22.0
    pit_loss_by_location: Dict[str, float] = field(default_factory=dict)
    pit_loss_sd: float = 2.0
    sc_pit_factor: float = 0.5
    vsc_pit_factor: float = 0.7
    stop_lap_bias: float = 0.0
    stop_lap_sd: float = 5.0
    extra_stop_rate: float = 0.3
    extra_stop_by_location: Dict[str, float] = field(default_factory=dict)
    circuit_laps: Dict[str, int] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)

    def hazard(self, kind: str, location: str) -> float:
        table = {SC: self.sc_by_location, VSC: self.vsc_by_location, RED: self.red_by_location}[kind]
        default = {SC: self.sc_hazard, VSC: self.vsc_hazard, RED: self.red_hazard}[kind]
        return float(table.get(location, default))

    def dnf(self, location: str, wet: bool) -> float:
        h = float(self.dnf_by_location.get(location, self.dnf_hazard))
        return h * (self.wet_dnf_multiplier if wet else 1.0)

    def pit(self, location: str) -> float:
        return float(self.pit_loss_by_location.get(location, self.pit_loss))

    def extra_stop(self, location: str) -> float:
        return float(self.extra_stop_by_location.get(location, self.extra_stop_rate))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventModel":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def optimal_stop_lap(d1: float, d2: float, total_laps: int, age_now: float, lap_now: int) -> float:
    """s* = (d2·L − d1·(a_k − k)) / (d1 + d2): where staying out stops paying (§5.3)."""
    d1, d2 = max(d1, 1e-3), max(d2, 1e-3)
    return (d2 * total_laps - d1 * (age_now - lap_now)) / (d1 + d2)


def next_compound(current: Optional[str], used: Sequence[str]) -> str:
    """The compound an owed stop goes to: the hardest dry one not used yet."""
    for name in ("HARD", "MEDIUM", "SOFT"):
        if name != current and name not in used:
            return name
    return "HARD" if current != "HARD" else "MEDIUM"


def _shrunk(events: Mapping[str, float], exposure: Mapping[str, float], pooled: float, kappa: float) -> Dict[str, float]:
    return {
        loc: float((events.get(loc, 0.0) + kappa * pooled) / (exposure[loc] + kappa))
        for loc in exposure
        if exposure[loc] > 0
    }


def fit_events(races: Sequence[RaceData], tyres: TyreModel) -> EventModel:
    """Fit every event component, race distances per circuit included, on ``races``
    (the training races only)."""
    model = EventModel()
    ev = {SC: defaultdict(float), VSC: defaultdict(float), RED: defaultdict(float)}
    green_laps: Dict[str, float] = defaultdict(float)
    sc_len, vsc_len = [], []
    dnf_events, dnf_exposure = defaultdict(float), defaultdict(float)
    wet_events = wet_exposure = dry_events = dry_exposure = 0.0
    losses: Dict[str, List[float]] = defaultdict(list)
    sc_ratio, vsc_ratio = [], []
    stop_resid: List[float] = []
    extra = defaultdict(lambda: [0.0, 0.0])

    for race in races:
        loc = race.ref.location
        order = final_order(race.result)
        total = max((r.get("number_of_laps") or 0) for r in race.result) if race.result else race.leader_laps()
        if total <= 2:
            continue
        events = regime_events(race.race_control)
        regimes = regime_by_lap(race.race_control, total)
        green_laps[loc] += sum(1 for lap, kind in regimes.items() if lap >= 2 and kind == "GREEN")
        for e in events:
            if e.start_lap >= 2:
                ev[e.kind][loc] += 1
            if e.laps:
                (sc_len if e.kind == SC else vsc_len if e.kind == VSC else []).append(e.laps)

        # retirements (racedata.is_retirement: not classified, not DSQ/DNS), after lap 1
        for r in race.result:
            d = r.get("driver_number")
            if d is None or is_excluded(r):
                continue
            laps = int(r.get("number_of_laps") or 0)
            retired = is_retirement(r)
            exposure = max(0, min(laps, total) - 1) + (1 if retired else 0)
            dnf_exposure[loc] += exposure
            hit = 1.0 if (retired and laps >= 1) else 0.0
            dnf_events[loc] += hit
            if race.wet:
                wet_events, wet_exposure = wet_events + hit, wet_exposure + exposure
            else:
                dry_events, dry_exposure = dry_events + hit, dry_exposure + exposure

        # pit loss vs the field median of the same laps
        laps = race.laps.dropna(subset=["lap_duration"])
        if not laps.empty:
            table = laps.pivot_table(index="lap_number", columns="driver_number", values="lap_duration")
            pitting = defaultdict(set)
            for stop in race.pit:
                if stop.get("lap_number") is not None:
                    pitting[int(stop["lap_number"])].add(stop["driver_number"])
                    pitting[int(stop["lap_number"]) + 1].add(stop["driver_number"])
            for stop in race.pit:
                d, lap = stop.get("driver_number"), stop.get("lap_number")
                if d is None or lap is None or d not in table.columns or lap < 2 or lap + 1 > total:
                    continue
                loss = 0.0
                ok = True
                for l in (int(lap), int(lap) + 1):
                    if l not in table.index or np.isnan(table.at[l, d]):
                        ok = False
                        break
                    others = table.loc[l].drop(labels=[c for c in pitting[l] if c in table.columns], errors="ignore")
                    loss += table.at[l, d] - float(np.nanmedian(others.values)) if others.notna().any() else np.nan
                if not ok or not np.isfinite(loss) or loss <= 0 or loss > 80:
                    continue
                kind = regimes.get(int(lap), "GREEN")
                if kind == "GREEN" and regimes.get(int(lap) + 1, "GREEN") == "GREEN":
                    losses[loc].append(loss)
                elif kind == SC:
                    sc_ratio.append(loss)
                elif kind == VSC:
                    vsc_ratio.append(loss)

        # stop timing of one-stop dry races, and how often cars stop more than they must
        if not race.wet:
            ts = tyre_state(race)
            for d, stops in ts.stops.items():
                if order.position.get(d) is None or d in order.retired or d in order.classified_dnf:
                    continue
                extra[loc][0] += 1
                extra[loc][1] += 1 if stops >= 2 else 0
                if stops != 1:
                    continue
                stints = sorted((s for s in race.stints if s.get("driver_number") == d), key=lambda s: s.get("stint_number") or 0)
                if len(stints) != 2 or stints[0].get("compound") not in DRY_COMPOUNDS or stints[1].get("compound") not in DRY_COMPOUNDS:
                    continue
                d1, d2 = tyres.slope.get(stints[0]["compound"], 0.05), tyres.slope.get(stints[1]["compound"], 0.05)
                s_star = optimal_stop_lap(d1, d2, total, float(stints[0].get("tyre_age_at_start") or 0), 0)
                actual = stints[0].get("lap_end")
                if actual is not None:
                    stop_resid.append(float(actual) - float(np.clip(s_star, 1, total - 1)))

    green_total = sum(green_laps.values())
    if green_total > 0:
        for kind, attr in ((SC, "sc"), (VSC, "vsc"), (RED, "red")):
            pooled = sum(ev[kind].values()) / green_total
            setattr(model, f"{attr}_hazard", float(pooled))
            setattr(model, f"{attr}_by_location", _shrunk(ev[kind], green_laps, pooled, HAZARD_POOLING_LAPS))
    if sc_len:
        model.sc_laps = [int(x) for x in sc_len]
    if vsc_len:
        model.vsc_laps = [int(x) for x in vsc_len]
    exposure_total = sum(dnf_exposure.values())
    if exposure_total > 0:
        model.dnf_hazard = float(sum(dnf_events.values()) / exposure_total)
        model.dnf_by_location = _shrunk(dnf_events, dnf_exposure, model.dnf_hazard, DNF_POOLING_DRIVER_LAPS)
        if dry_exposure > 0 and wet_exposure > 0:
            dry_rate = dry_events / dry_exposure
            model.wet_dnf_multiplier = float((wet_events + 2) / (wet_exposure * dry_rate + 2))
    green_losses = [x for xs in losses.values() for x in xs]
    if green_losses:
        model.pit_loss = float(np.median(green_losses))
        model.pit_loss_by_location = {
            loc: float((len(xs) * np.median(xs) + 5 * model.pit_loss) / (len(xs) + 5)) for loc, xs in losses.items() if xs
        }
        resid = [x - np.median(xs) for xs in losses.values() for x in xs]
        model.pit_loss_sd = float(1.4826 * np.median(np.abs(resid))) if resid else model.pit_loss_sd
        if len(sc_ratio) >= 5:
            model.sc_pit_factor = float(np.clip(np.median(sc_ratio) / model.pit_loss, 0.2, 1.0))
        if len(vsc_ratio) >= 5:
            model.vsc_pit_factor = float(np.clip(np.median(vsc_ratio) / model.pit_loss, 0.2, 1.0))
    if len(stop_resid) >= 10:
        model.stop_lap_bias = float(np.median(stop_resid))
        model.stop_lap_sd = float(max(1.0, 1.4826 * np.median(np.abs(np.array(stop_resid) - model.stop_lap_bias))))
    finishers = sum(v[0] for v in extra.values())
    if finishers:
        model.extra_stop_rate = float(sum(v[1] for v in extra.values()) / finishers)
        model.extra_stop_by_location = {
            loc: float((v[1] + 20 * model.extra_stop_rate) / (v[0] + 20)) for loc, v in extra.items() if v[0]
        }
    for race in races:
        winner = [r.get("number_of_laps") for r in race.result if r.get("position") == 1]
        if winner and winner[0]:
            model.circuit_laps[race.ref.location] = max(int(winner[0]), model.circuit_laps.get(race.ref.location, 0))
    model.counts = {
        "races": len(races),
        "green_laps": green_total,
        "sc": int(sum(ev[SC].values())),
        "vsc": int(sum(ev[VSC].values())),
        "red": int(sum(ev[RED].values())),
        "retirements": int(sum(dnf_events.values())),
        "green_pit_stops": len(green_losses),
        "sc_pit_stops": len(sc_ratio),
        "one_stop_drivers": len(stop_resid),
    }
    return model
