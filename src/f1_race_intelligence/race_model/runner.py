"""The race model's pipeline stages: train, evaluate (+ promote), update, forecast.

``train``     fits level 1 on every race, level 2 / events / baselines on the
              training races, picks the lap-noise multiplier, replays the
              season filter chronologically through every race, and writes a
              new immutable version.
``evaluate``  scores a version on the validation (or test) races at the
              configured k-fractions against both baselines, writes the
              report, and with ``promote`` moves ``latest`` to it if it beats
              the required baseline.
``update``    after a race weekend: re-runs level 1 and the filter replay with
              the hyperparameters frozen, so the next race's prior includes the
              last result. Writes a new version and promotes it if its parent
              was the promoted one.
``forecast``  one race at one lap, from the raw tree or live from OpenF1.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from f1_race_intelligence.race_model.artifact import (
    ModelArtifact,
    ModelRegistry,
    forecasts_prefix,
    new_version,
    reports_prefix,
)
from f1_race_intelligence.race_model.baselines import TransitionBaseline, order_at_k, persist_baseline, ahead_from_matrix
from f1_race_intelligence.race_model.contracts import (
    build_provenance,
    diagnose_race_model_generalization,
    extract_interpretability,
    log_race_model_to_mlflow,
    render_evaluation_markdown,
    validate_forecast_response,
)
from f1_race_intelligence.race_model.events import EventModel, fit_events
from f1_race_intelligence.race_model.forecast import ForecastConfig, ForecastResult, car_states, forecast
from f1_race_intelligence.race_model.pace import Level1Config, TyreModel, fit_level1
from f1_race_intelligence.race_model.racedata import (
    RaceCatalog,
    RaceData,
    RaceRef,
    final_order,
    load_live_race,
    scheduled_laps,
    truncate_at_lap,
)
from f1_race_intelligence.race_model.scoring import (
    brier,
    calibration_table,
    event_probabilities,
    paired_bootstrap,
    pairwise_log_loss,
    rps_rows,
)
from f1_race_intelligence.race_model.season import Prior, SeasonFilter, SeasonParams, fit_season, observations, replay
from f1_race_intelligence.storage.backends.base import join_key
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


# -- splits -----------------------------------------------------------------------


TRAIN, VALIDATION, TEST, FUTURE = "train", "validation", "test", "future"


def split_of(ref: RaceRef, train_years: Sequence[int], validation_until: str, test_until: str) -> str:
    """The split of one race, from fixed dates: it never depends on which other races exist."""
    if ref.year in set(train_years):
        return TRAIN
    day = (ref.date_start or "")[:10]
    if day < validation_until:
        return VALIDATION
    if day < test_until:
        return TEST
    return FUTURE


def assign_splits(refs: Sequence[RaceRef], train_years: Sequence[int], validation_until: str,
                  test_until: str) -> Dict[int, str]:
    return {r.session_key: split_of(r, train_years, validation_until, test_until) for r in refs}


class SealedTestError(PermissionError):
    """A code path tried to read a test race while the test set is sealed."""


def reference_lap(race: RaceData) -> float:
    durations = race.laps["lap_duration"].dropna()
    return float(durations.median()) if len(durations) else 90.0


def k_laps(total_laps: int, fractions: Sequence[float]) -> List[Tuple[str, int]]:
    out = []
    for f in fractions:
        k = int(min(total_laps - 1, max(1, round(f * total_laps))))
        out.append((f"{int(round(f * 100))}%", k))
    return out


# -- assembling a model from an artifact -------------------------------------------


@dataclass
class LoadedModel:
    artifact: ModelArtifact
    level1: Level1Config
    phi: float
    sigma_eps: float
    tyres: TyreModel
    tyres_by_location: Dict[str, Dict[str, Any]]
    events: EventModel
    config: ForecastConfig
    season: SeasonFilter
    priors: Dict[int, Prior]
    transition: TransitionBaseline

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact, scenarios: Optional[int] = None) -> "LoadedModel":
        l1 = artifact.level1
        fc = dict(artifact.forecast)
        if scenarios:
            fc["scenarios"] = scenarios
        return cls(
            artifact=artifact,
            level1=Level1Config(**l1["config"]),
            phi=float(l1["phi"]),
            sigma_eps=float(l1["sigma_eps"]),
            tyres=TyreModel.from_dict(l1["tyres"]),
            tyres_by_location=dict(l1.get("tyres_by_location") or {}),
            events=EventModel.from_dict(artifact.events),
            config=ForecastConfig(**{k: v for k, v in fc.items() if k in ForecastConfig.__dataclass_fields__}),
            season=SeasonFilter.from_dict(artifact.season["state"]),
            priors={int(k): Prior.from_dict(v) for k, v in artifact.season["priors"].items()},
            transition=TransitionBaseline.from_dict(artifact.transition),
        )

    def tyres_for(self, location: str) -> TyreModel:
        entry = self.tyres_by_location.get(location)
        return TyreModel.from_dict(entry) if entry else self.tyres

    def prior_for(self, race: RaceData) -> Prior:
        """The replayed pre-race prior, or a preview from the filter's latest state (a new race)."""
        if race.ref.session_key in self.priors:
            return self.priors[race.ref.session_key]
        drivers, teams, quali = [], [], []
        grid = {g["driver_number"]: g for g in race.grid if g.get("lap_duration")}
        q_med = np.median([g["lap_duration"] for g in grid.values()]) if grid else np.nan
        for d in race.driver_numbers:
            team = race.team_of(d)
            if team is None:
                continue
            drivers.append(race.acronym_of(d))
            teams.append(team)
            g = grid.get(d)
            quali.append(100 * (g["lap_duration"] / q_med - 1) if g and np.isfinite(q_med) else np.nan)
        return self.season.preview(race.ref.year, teams, drivers, np.asarray(quali, float))

    def total_laps(self, race: RaceData, override: Optional[int] = None) -> int:
        return int(override) if override else scheduled_laps(race, self.events.circuit_laps)

    def run(self, race_k: RaceData, prior: Prior, total_laps: int, seed: int = 0, events: bool = True) -> ForecastResult:
        config = self.config if events else ForecastConfig(**{**asdict(self.config), "events": False})
        return forecast(
            race_k,
            prior,
            total_laps,
            level1=self.level1,
            phi=self.phi,
            sigma_eps=self.sigma_eps,
            tyres=self.tyres_for(race_k.ref.location),
            events=self.events,
            config=config,
            rng=np.random.default_rng(seed),
            model_version=self.artifact.version,
            fallback_prior_variance=float(self.season.params.p0_gamma + self.season.params.p0_delta),
        )


# -- scoring one forecast --------------------------------------------------------------


def score_forecast(result: ForecastResult, race: RaceData, model: LoadedModel, bucket: str) -> Dict[str, Any]:
    """Model and baseline scores for one (race, k), over cars running at k."""
    order = final_order(race.result)
    numbers = [d["driver_number"] for d in result.drivers]
    actual = [order.position.get(n) for n in numbers]
    running = [not d["retired_at_k"] and a is not None for d, a in zip(result.drivers, actual)]
    scored = [a if r else None for a, r in zip(actual, running)]

    race_k = truncate_at_lap(race, result.lap)
    cars = car_states(race_k, reference_lap(race_k))
    by_number = {c.driver_number: c for c in cars}
    cars = [by_number[n] for n in numbers]
    for car, d in zip(cars, result.drivers):  # the same retirement call as the forecast
        car.retired = bool(d["retired_at_k"])
    owed = [d["owes_stop"] for d in result.drivers]
    persist = persist_baseline(cars, owed, model.events.pit(race.ref.location))
    transition = model.transition.predict(bucket, order_at_k(cars))

    def summary(P: np.ndarray, ahead: Optional[np.ndarray]) -> Dict[str, float]:
        rps = rps_rows(P, scored)
        out = {"rps": float(np.nanmean(rps)) if np.isfinite(rps).any() else float("nan")}
        if ahead is not None:
            out["pairwise_log_loss"] = pairwise_log_loss(ahead, scored)
        return out

    probs = event_probabilities(result.matrix)
    outcomes = {
        "win": [a == 1 for a in scored],
        "podium": [a is not None and a <= 3 for a in scored],
        "points": [a is not None and a <= 10 for a in scored],
    }
    mask = [a is not None for a in scored]
    events_rows = {
        name: {"p": [float(p) for p, m in zip(probs[name], mask) if m], "y": [bool(y) for y, m in zip(outcomes[name], mask) if m]}
        for name in probs
    }
    dnf_truth = [n in order.retired for n, d in zip(numbers, result.drivers) if not d["retired_at_k"] and n not in order.excluded]
    dnf_p = [float(p) for p, d, n in zip(result.p_dnf, result.drivers, numbers) if not d["retired_at_k"] and n not in order.excluded]
    return {
        "session_key": race.ref.session_key,
        "race": race.tag,
        "bucket": bucket,
        "lap": result.lap,
        "total_laps": result.total_laps,
        "cars_scored": int(sum(mask)),
        "model": summary(result.matrix, result.ahead),
        "persist": summary(persist, None),
        "transition": summary(transition, ahead_from_matrix(transition)),
        "events": events_rows,
        "dnf": {"p": dnf_p, "y": dnf_truth},
    }


def aggregate(rows: Sequence[Mapping[str, Any]], baseline: str = "persist") -> Dict[str, Any]:
    """Per-bucket means, race-paired bootstrap vs each baseline, and calibration."""
    frame = pd.DataFrame(
        [
            {
                "session_key": r["session_key"],
                "bucket": r["bucket"],
                **{f"{m}_rps": r[m]["rps"] for m in ("model", "persist", "transition")},
                "model_pll": r["model"].get("pairwise_log_loss"),
                "transition_pll": r["transition"].get("pairwise_log_loss"),
            }
            for r in rows
        ]
    )
    if frame.empty:
        return {"rows": 0}
    out: Dict[str, Any] = {"rows": int(len(frame)), "races": int(frame["session_key"].nunique()), "by_bucket": {}}
    for bucket, g in list(frame.groupby("bucket", sort=False)) + [("all", frame)]:
        entry = {c: float(g[c].mean()) for c in g.columns if c.endswith(("_rps", "_pll"))}
        for base in ("persist", "transition"):
            per_race = (g.groupby("session_key")["model_rps"].mean() - g.groupby("session_key")[f"{base}_rps"].mean()).to_dict()
            entry[f"delta_rps_vs_{base}"] = paired_bootstrap(per_race)
        out["by_bucket"][bucket] = entry
    calib: Dict[str, Any] = {}
    for name in ("win", "podium", "points"):
        p = [x for r in rows for x in r["events"][name]["p"]]
        y = [x for r in rows for x in r["events"][name]["y"]]
        calib[name] = {"brier": brier(p, y), "table": calibration_table(p, y)}
    p = [x for r in rows for x in r["dnf"]["p"]]
    y = [x for r in rows for x in r["dnf"]["y"]]
    calib["dnf"] = {"brier": brier(p, y), "mean_forecast": float(np.mean(p)) if p else None, "observed": float(np.mean(y)) if y else None}
    out["calibration"] = calib
    out["beats_" + baseline] = bool(out["by_bucket"]["all"][f"delta_rps_vs_{baseline}"]["mean"] < 0)
    return out


# -- the stages ----------------------------------------------------------------------


@dataclass
class StageResult:
    ok: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


class RaceModelRunner:
    def __init__(self, settings: Any, input_store: DataStore, output_store: DataStore, client_factory: Optional[Callable[[], Any]] = None) -> None:
        self.settings = settings
        self.cfg = settings.race_model
        self.input_store = input_store
        self.output_store = output_store
        self.catalog = RaceCatalog(input_store, self.cfg.raw_path)
        self.registry = ModelRegistry(output_store, self.cfg.output_path)
        self.client_factory = client_factory

    # -- data -------------------------------------------------------------------

    def splits_of(self, refs: Sequence[RaceRef]) -> Dict[int, str]:
        return assign_splits(refs, self.cfg.train_years, self.cfg.validation_until, self.cfg.test_until)

    def _load(self, ref: RaceRef, split: str, unsealed: bool = False) -> RaceData:
        """The only place a race's raw files are opened: test races need an explicit unseal."""
        if split == TEST and self.cfg.test_sealed and not unsealed:
            raise SealedTestError(f"{ref.tag} is a test race and the test set is sealed")
        return self.catalog.load(ref)

    def load_races(self, include_test: bool = False) -> Tuple[List[RaceRef], List[RaceData]]:
        """Races in calendar order. Test races are dropped *before* their raw files are read,
        unless ``include_test`` (the final, ledgered test evaluation)."""
        all_refs = self.catalog.races(self.cfg.years, exclude=self.cfg.exclude_races)
        splits = self.splits_of(all_refs)
        refs = [r for r in all_refs if include_test or splits[r.session_key] != TEST]
        races = []
        for ref in refs:
            try:
                race = self._load(ref, splits[ref.session_key], unsealed=include_test)
            except LookupError as exc:
                logger.warning("race_skipped", extra={"race": ref.tag, "reason": str(exc)})
                continue
            if not race.result:
                logger.warning("race_without_result", extra={"race": ref.tag})
            races.append(race)
        return [r.ref for r in races], races

    def _report(self, name: str, payload: Mapping[str, Any]) -> str:
        prefix = reports_prefix(self.output_store, self.cfg.output_path)
        return self.output_store.backend.write_json_unique(prefix, name, payload)

    # -- shared fitting ------------------------------------------------------------

    def _fit_all(self, races: Sequence[RaceData], splits: Mapping[int, str], params: Optional[SeasonParams] = None,
                 events: Optional[EventModel] = None, forecast_cfg: Optional[Dict[str, Any]] = None,
                 transition: Optional[TransitionBaseline] = None,
                 frozen_level1: Optional[Mapping[str, Any]] = None) -> Tuple[ModelArtifact, Dict[str, Any]]:
        cfg = self.cfg
        level1 = Level1Config.from_settings(cfg)
        train_keys = {k for k, s in splits.items() if s == TRAIN}
        train_races = [r for r in races if r.ref.session_key in train_keys]
        started = time.monotonic()
        # Every pooled level-1 quantity comes from training races (or is frozen).
        l1 = fit_level1(
            races, level1, tyre_races=train_keys, noise_races=train_keys,
            phi=frozen_level1["phi"] if frozen_level1 else None,
            sigma_eps=frozen_level1["sigma_eps"] if frozen_level1 else None,
        )
        obs = observations(l1.obs)
        logger.info("m7_level1_fitted", extra={"races": len(races), "phi": l1.phi, "sigma_eps": l1.sigma_eps})

        fitted_ll = None
        if params is None:
            base = SeasonParams(p0_gamma=cfg.initial_team_variance, p0_delta=cfg.initial_driver_variance)
            params, fitted_ll = fit_season(obs, train_keys, burn_in=cfg.burn_in_races, fit_rho=cfg.fit_rho,
                                           base=base, quali_max_abs_pct=cfg.quali_max_abs_pct)
            logger.info("m7_season_fitted", extra={"params": params.to_dict(), "loglik": fitted_ll})
        state, priors, _ = replay(obs, params, cfg.quali_max_abs_pct)
        if events is None:
            events = fit_events(train_races, l1.tyres)

        settings_dump = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
        val_count = sum(1 for r in races if splits.get(r.ref.session_key) == VALIDATION)
        provenance = build_provenance(
            settings_dump,
            train_races,
            validation_races_count=val_count,
            test_sealed=cfg.test_sealed,
            random_seed=cfg.random_seed,
        )
        artifact = ModelArtifact(
            version=new_version(),
            created_at=datetime.now(timezone.utc).isoformat(),
            settings=settings_dump,
            level1={
                "config": asdict(level1),
                "phi": l1.phi,
                "sigma_eps": l1.sigma_eps,
                "tyres": l1.tyres.to_dict(),
                "tyres_by_location": l1.tyres_by_location,
            },
            season={"state": state.to_dict(), "priors": {str(k): p.to_dict() for k, p in priors.items()},
                    "train_loglik": fitted_ll},
            events=events.to_dict(),
            forecast=forecast_cfg or {
                "scenarios": cfg.scenarios,
                "restart_gap_seconds": cfg.restart_gap_seconds,
                "pit_under_sc_probability": cfg.pit_under_sc_probability,
                "pit_under_vsc_probability": cfg.pit_under_vsc_probability,
                "noise_scale": cfg.noise_scales[0] if cfg.noise_scales else 1.0,
            },
            transition=(transition or TransitionBaseline()).to_dict(),
            races=[{**r.ref.to_dict(), "split": splits.get(r.ref.session_key)} for r in races],
            provenance=provenance,
        )
        prior_check = self._prior_check(l1.obs, priors, splits)
        metrics = {
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "races": len(races),
            "splits": pd.Series(list(splits.values())).value_counts().to_dict(),
            "level1": {
                "phi": l1.phi,
                "sigma_eps_median": l1.sigma_eps,
                "driver_races_with_theta": int(l1.obs["theta_hat"].notna().sum()),
                "wet_races": int(sum(1 for r in races if r.wet)),
            },
            "season_params": params.to_dict(),
            "prior_check": prior_check,
            "events": events.counts,
        }
        artifact.metrics = metrics
        return artifact, metrics

    @staticmethod
    def _prior_check(obs: pd.DataFrame, priors: Mapping[int, Prior], splits: Mapping[int, str]) -> Dict[str, Any]:
        """Pre-race prior vs full-race θ̂ (math note §3.3 xi): RMSE, corrected τ², calibration z²."""
        rows = []
        frame = obs[obs["team"].notna() & obs["theta_hat"].notna()].copy()
        frame["driver_id"] = frame["name"].fillna(frame["driver"].astype(str)).astype(str)
        for key, g in frame.groupby("session_key"):
            prior = priors.get(int(key))
            if prior is None:
                continue
            idx = {d: i for i, d in enumerate(prior.drivers)}
            for _, r in g.iterrows():
                i = idx.get(r["driver_id"])
                if i is None:
                    continue
                rows.append((splits.get(int(key)), r["theta_hat"] - prior.mean[i], prior.cov[i, i], r["se"]))
        out = {}
        # Test races are scored only by an explicit, ledgered `m7-evaluate --split test`.
        for split in (TRAIN, VALIDATION):
            sel = [x for x in rows if x[0] == split]
            if not sel:
                continue
            err = np.array([x[1] for x in sel])
            tau2 = np.array([x[2] for x in sel])
            se2 = np.array([x[3] for x in sel]) ** 2
            out[split] = {
                "n": len(sel),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "tau2_corrected": float(np.mean(err**2) - np.mean(se2)),
                "tau2_model": float(np.mean(tau2)),
                "z2": float(np.mean(err**2 / (tau2 + se2))),
            }
        return out

    def _transition_counts(self, races: Sequence[RaceData]) -> TransitionBaseline:
        base = TransitionBaseline()
        for race in races:
            order = final_order(race.result)
            total = scheduled_laps(race)
            for bucket, k in k_laps(total, self.cfg.k_fractions):
                try:
                    race_k = truncate_at_lap(race, k)
                    cars = car_states(race_k, reference_lap(race_k))
                except ValueError:
                    continue
                running = order_at_k(cars)
                base.add(bucket, [running.index(i) + 1 for i in range(len(cars))],
                         [order.position.get(c.driver_number) for c in cars])
        return base

    def _choose_noise(self, model: LoadedModel, races: Sequence[RaceData]) -> Tuple[float, Dict[str, Any]]:
        """Pick the lap-noise multiplier by RPS on the latest training races.

        With ``noise_scale_per_phase`` one multiplier is chosen per k-bucket
        (stored as ``noise_scale_by_fraction``); otherwise one for all.
        """
        cfg = self.cfg
        if len(cfg.noise_scales) <= 1:
            return (cfg.noise_scales[0] if cfg.noise_scales else 1.0), {}
        sample = [r for r in races if not r.wet][-cfg.calibration_races:]
        fractions = {f"{int(round(f * 100))}%": f for f in cfg.k_fractions}
        by_bucket: Dict[str, Dict[str, List[float]]] = {}
        model.config.noise_scale_by_fraction = None
        model.config.scenarios = min(cfg.scenarios, 1000)
        for scale in cfg.noise_scales:
            model.config.noise_scale = scale
            for race in sample:
                for bucket, k in k_laps(model.total_laps(race), cfg.k_fractions):
                    try:
                        result = model.run(truncate_at_lap(race, k), model.prior_for(race), model.total_laps(race), seed=cfg.random_seed)
                    except ValueError:
                        continue
                    rps = score_forecast(result, race, model, bucket)["model"]["rps"]
                    by_bucket.setdefault(bucket, {}).setdefault(str(scale), []).append(rps)
        overall = {str(s): float(np.nanmean([x for b in by_bucket.values() for x in b.get(str(s), [])])) for s in cfg.noise_scales}
        best = float(min(overall, key=overall.get))
        detail: Dict[str, Any] = {"overall": overall, "by_bucket": {
            b: {s: float(np.nanmean(v)) for s, v in scores.items()} for b, scores in by_bucket.items()}}
        if cfg.noise_scale_per_phase:
            detail["chosen_by_fraction"] = {
                str(fractions[b]): float(min(scores, key=scores.get)) for b, scores in detail["by_bucket"].items()}
        return best, detail

    # -- stages -------------------------------------------------------------------

    def train(self) -> StageResult:
        refs, races = self.load_races()
        if not races:
            return StageResult(False, reason="no races with raw data were found")
        splits = self.splits_of(refs)
        assert TEST not in splits.values(), "a test race reached training"
        train_races = [r for r in races if splits[r.ref.session_key] == TRAIN and r.result]
        if len(train_races) <= self.cfg.burn_in_races:
            return StageResult(False, reason=f"only {len(train_races)} training races; need more than the burn-in")
        transition = self._transition_counts(train_races)
        artifact, metrics = self._fit_all(races, splits, transition=transition)
        model = LoadedModel.from_artifact(artifact)
        scale, scores = self._choose_noise(model, train_races)
        artifact.forecast["noise_scale"] = scale
        artifact.forecast["noise_scale_by_fraction"] = scores.get("chosen_by_fraction")
        artifact.metrics["noise_scale_rps"] = scores
        interpretability = extract_interpretability(artifact)
        uri = self.registry.save(artifact)
        verification = self.registry.verify(artifact.version)
        report = {
            "stage": "m7-train",
            "version": artifact.version,
            "model_uri": uri,
            "provenance": artifact.provenance,
            "interpretability": interpretability,
            "package_verification": verification,
            **artifact.metrics,
        }
        self._report("train", report)
        return StageResult(True, {
            "version": artifact.version,
            "model_uri": uri,
            "noise_scale": scale,
            "prior_check": metrics["prior_check"],
            "input_fingerprint": artifact.provenance.get("input_fingerprint"),
            "manifest_sha256": verification.get("manifest_sha256"),
        })

    # -- the sealed test set ----------------------------------------------------------

    @property
    def _ledger_key(self) -> str:
        return join_key(reports_prefix(self.output_store, self.cfg.output_path), "test_ledger.json")

    def test_ledger(self) -> List[Dict[str, Any]]:
        backend = self.output_store.backend
        if not backend.exists(self._ledger_key):
            return []
        import json

        return list(json.loads(backend.read_bytes(self._ledger_key)).get("entries", []))

    def _open_test(self, artifact: ModelArtifact, allow_repeat: bool) -> Tuple[LoadedModel, List[RaceData]]:
        """Unseal once: record it, then replay the season filter online through the test races.

        The entry is written *before* any test race is read, so even a crashed
        run counts as a look. Development artifacts never contain test races,
        so their pre-race priors are rebuilt here the way a live forecaster
        would have had them: every hyperparameter (level 1, season filter,
        events) frozen from the artifact, and each test race's prior using
        only races that finished before it.
        """
        entries = self.test_ledger()
        if entries and not allow_repeat:
            raise SealedTestError(
                f"the test set was already evaluated ({len(entries)} time(s), last {entries[-1]['at']} on version "
                f"{entries[-1]['version']}); a second look turns it into validation. Pass --allow-repeat-test to "
                "override; the ledger keeps the count."
            )
        entry = {"version": artifact.version, "at": datetime.now(timezone.utc).isoformat(), "look": len(entries) + 1}
        self.output_store.backend.write_json(self._ledger_key, {"entries": entries + [entry]})
        logger.warning("test_set_unsealed", extra=entry)

        refs, races = self.load_races(include_test=True)
        splits = self.splits_of(refs)
        params = SeasonParams.from_dict(artifact.season["state"]["params"])
        replayed, _ = self._fit_all(
            races, splits, params=params, events=EventModel.from_dict(artifact.events),
            forecast_cfg=artifact.forecast, transition=TransitionBaseline.from_dict(artifact.transition),
            frozen_level1=artifact.level1,
        )
        model = LoadedModel.from_artifact(artifact)
        model.priors = {int(k): Prior.from_dict(v) for k, v in replayed.season["priors"].items()}
        return model, [r for r in races if splits[r.ref.session_key] == TEST]

    def evaluate(self, version: Optional[str] = None, split: str = "validation", promote: bool = False,
                 scenarios: Optional[int] = None, unseal_test: bool = False,
                 allow_repeat_test: bool = False, log_mlflow: bool = False) -> StageResult:
        artifact = self.registry.load(version) if version else self._newest()
        if split == TEST:
            if promote:
                return StageResult(False, reason="promotion is decided on validation, never on test")
            if self.cfg.test_sealed and not unseal_test:
                return StageResult(False, reason="the test set is sealed: pass --unseal-test for the final, "
                                                 "recorded evaluation (it is refused a second time)")
            try:
                model, races = self._open_test(artifact, allow_repeat_test)
            except SealedTestError as exc:
                return StageResult(False, reason=str(exc))
            if scenarios:
                model.config.scenarios = scenarios
        elif split == VALIDATION:
            model = LoadedModel.from_artifact(artifact, scenarios)
            refs = [RaceRef.from_dict(r) for r in artifact.races if r.get("split") == VALIDATION]
            races = [self._load(ref, self.splits_of([ref])[ref.session_key]) for ref in refs]
        else:
            return StageResult(False, reason=f"unknown split {split!r}")
        if not races:
            return StageResult(False, reason=f"no {split} races to evaluate for version {artifact.version}")
        rows = []
        for race in races:
            if not race.result:
                continue
            total = model.total_laps(race)
            for bucket, k in k_laps(total, self.cfg.k_fractions):
                try:
                    result = model.run(truncate_at_lap(race, k), model.prior_for(race), total, seed=self.cfg.random_seed)
                except ValueError as exc:
                    logger.warning("forecast_skipped", extra={"race": race.tag, "lap": k, "reason": str(exc)})
                    continue
                rows.append(score_forecast(result, race, model, bucket))
        summary = aggregate(rows, self.cfg.promote_requires_beating)
        interpretability = extract_interpretability(artifact)
        verification = self.registry.verify(artifact.version)
        # Generalization diagnosis is strictly train + validation (never test).
        diagnosis = (
            diagnose_race_model_generalization(
                prior_check=(artifact.metrics or {}).get("prior_check") or {},
                validation_summary=summary,
                noise_scale_rps=(artifact.metrics or {}).get("noise_scale_rps"),
                baseline_key=self.cfg.promote_requires_beating,
            )
            if split == VALIDATION
            else None
        )
        promoted = False
        if promote:
            if summary.get(f"beats_{self.cfg.promote_requires_beating}"):
                promoted = True
        report: Dict[str, Any] = {
            "stage": "m7-evaluate",
            "version": artifact.version,
            "split": split,
            "provenance": artifact.provenance,
            "generalization_diagnosis": diagnosis,
            "interpretability": interpretability,
            "package_verification": verification,
            **summary,
            "rows_detail": [{k: v for k, v in r.items() if k not in ("events", "dnf")} for r in rows],
        }
        report_uri = self._report(f"evaluate_{split}", report)
        if promoted:
            self.registry.promote(artifact.version, {"reason": "evaluation", "report": report_uri,
                                                     "delta_rps": summary["by_bucket"]["all"]})
        md_text = render_evaluation_markdown(
            report=report,
            artifact=artifact,
            promoted=promoted,
            verification=verification,
            mlflow_info=None,
        )
        mlflow_info = None
        if log_mlflow:
            mlflow_info = log_race_model_to_mlflow(
                settings=self.settings,
                artifact=artifact,
                evaluation_report=report,
                markdown_report=md_text,
                model_bytes=self.registry.read_model_bytes(artifact.version),
                promoted=promoted,
                verification=verification,
                register_model=(split == VALIDATION),
            )
            md_text = render_evaluation_markdown(
                report=report,
                artifact=artifact,
                promoted=promoted,
                verification=verification,
                mlflow_info=mlflow_info,
            )
        md_key = join_key(reports_prefix(self.output_store, self.cfg.output_path), f"evaluate_{split}.md")
        markdown_uri = self.output_store.backend.write_bytes(
            md_key, md_text.encode("utf-8"), content_type="text/markdown; charset=utf-8"
        )
        overall = summary.get("by_bucket", {}).get("all", {})
        out: Dict[str, Any] = {
            "version": artifact.version,
            "split": split,
            "races": summary.get("races"),
            "promoted": promoted,
            "model_rps": overall.get("model_rps"),
            "persist_rps": overall.get("persist_rps"),
            "transition_rps": overall.get("transition_rps"),
            "generalization_status": diagnosis.get("status") if diagnosis else None,
            "package_verified": verification.get("ok"),
            "report": report_uri,
            "markdown_report": markdown_uri,
        }
        if mlflow_info:
            out["mlflow"] = mlflow_info
        if promote and not promoted:
            return StageResult(False, out, reason=f"version does not beat the {self.cfg.promote_requires_beating} baseline; latest unchanged")
        return StageResult(True, out)

    def verify(self, version: Optional[str] = None) -> StageResult:
        """Read-only SHA-256 verification of a saved race-model version package."""
        target = version or self.registry.latest_version()
        if target is None:
            versions = self.registry.versions()
            target = versions[-1] if versions else None
        result = self.registry.verify(target)
        reason = None if result["ok"] else "; ".join(result["errors"])
        return StageResult(result["ok"], result, reason=reason)

    def update(self) -> StageResult:
        """Refresh level 1 and the filter state with frozen hyperparameters."""
        parent = self.registry.load()
        # While the test set is sealed the update skips test races too; the
        # filter then carries form across that gap with its random-walk variance.
        refs, races = self.load_races()
        known = {r["session_key"] for r in parent.races}
        new_refs = [r for r in refs if r.session_key not in known]
        splits = self.splits_of(refs)
        params = SeasonParams.from_dict(parent.season["state"]["params"])
        artifact, _ = self._fit_all(
            races, splits, params=params, events=EventModel.from_dict(parent.events), forecast_cfg=parent.forecast,
            transition=TransitionBaseline.from_dict(parent.transition), frozen_level1=parent.level1,
        )
        # level-1 tyre/noise terms stay those of the trained parent
        artifact.level1 = {**artifact.level1, "phi": parent.level1["phi"], "sigma_eps": parent.level1["sigma_eps"],
                           "tyres": parent.level1["tyres"], "tyres_by_location": parent.level1["tyres_by_location"]}
        artifact.parent_version = parent.version
        uri = self.registry.save(artifact)
        self.registry.promote(artifact.version, {"reason": "update", "parent": parent.version})
        self._report("update", {"stage": "m7-update", "version": artifact.version, "parent": parent.version,
                                "new_races": [r.tag for r in new_refs]})
        return StageResult(True, {"version": artifact.version, "parent": parent.version, "model_uri": uri,
                                  "new_races": [r.tag for r in new_refs]})

    def forecast(self, session_key: int, lap: Optional[int] = None, source: str = "raw",
                 version: Optional[str] = None, total_laps: Optional[int] = None,
                 year: Optional[int] = None) -> StageResult:
        artifact = self.registry.load(version)
        model = LoadedModel.from_artifact(artifact)
        if source == "live":
            if self.client_factory is None:
                return StageResult(False, reason="live forecasts need an OpenF1 client")
            race = load_live_race(self.client_factory(), session_key)
            if self.cfg.test_sealed and self.splits_of([race.ref])[race.ref.session_key] == TEST:
                return StageResult(False, reason=f"{race.tag} is a test race and the test set is sealed")
        else:
            ref = next((r for r in self.catalog.races([year] if year else self.cfg.years) if r.session_key == session_key), None)
            if ref is None:
                return StageResult(False, reason=f"session {session_key} is not a Grand Prix in the raw tree")
            try:  # the result is only a label; truncate_at_lap drops it
                race = self._load(ref, self.splits_of([ref])[ref.session_key])
            except SealedTestError as exc:
                return StageResult(False, reason=str(exc))
        k = int(lap or race.leader_laps())
        if k < 1:
            return StageResult(False, reason="the race has not completed a lap yet")
        race_k = truncate_at_lap(race, k)
        result = model.run(race_k, model.prior_for(race), model.total_laps(race, total_laps), seed=self.cfg.random_seed)
        payload = result.to_dict()
        validate_forecast_response(payload)
        prefix = join_key(forecasts_prefix(self.output_store, self.cfg.output_path),
                          f"year={race.ref.year}", f"session={race.ref.session_key}")
        key = join_key(prefix, f"lap={k:03d}.json")
        uri = self.output_store.backend.write_json(key, payload)
        self.output_store.backend.write_json(join_key(prefix, "latest.json"), payload)
        top = sorted(payload["drivers"], key=lambda d: -d["p_win"])[:3]
        return StageResult(True, {"forecast_uri": uri, "race": race.tag, "lap": k, "total_laps": result.total_laps,
                                  "version": artifact.version, "top_win": {d["driver"]: d["p_win"] for d in top}})

    def _newest(self) -> ModelArtifact:
        versions = self.registry.versions()
        if not versions:
            raise LookupError("no model versions; run m7-train first")
        return self.registry.load(versions[-1])
