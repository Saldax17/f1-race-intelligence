"""Race model (M7): regime parsing, leakage guard, sampler vs closed form, filter, scores, registry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from f1_race_intelligence.race_model.artifact import ModelArtifact, ModelRegistry, new_version
from f1_race_intelligence.race_model.events import EventModel, optimal_stop_lap
from f1_race_intelligence.race_model.forecast import (
    ForecastConfig,
    car_states,
    forecast,
    pace_posterior,
    pairwise_closed_form,
)
from f1_race_intelligence.race_model.pace import Level1Config, TyreModel, clean_laps, fit_level1
from f1_race_intelligence.race_model.racedata import RaceRef, build_race, final_order, truncate_at_lap
from f1_race_intelligence.race_model.regime import RED, SC, VSC, neutralised_laps, regime_events
from f1_race_intelligence.race_model.runner import assign_splits, k_laps
from f1_race_intelligence.race_model.scoring import paired_bootstrap, pairwise_log_loss, rps_rows
from f1_race_intelligence.race_model.season import Prior, RaceObservation, SeasonFilter, SeasonParams, replay
from f1_race_intelligence.storage.store import DataStore

T0 = datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)
OFFSETS = [-0.6, -0.3, 0.0, 0.2, 0.5, 0.8]  # % of a lap
TEAMS = ["A", "A", "B", "B", "C", "C"]


def iso(t: datetime) -> str:
    return t.isoformat()


def synthetic_race(total=20, seed=1, session_key=9001, year=2024, location="Testville", messages=()):
    rng = np.random.default_rng(seed)
    base = 90.0
    laps, intervals, stints, result = [], [], [], []
    ends = {}
    for i, off in enumerate(OFFSETS):
        d = i + 1
        t = T0 + timedelta(seconds=0.3 * i)
        for lap in range(1, total + 1):
            duration = base * (1 + off / 100) + rng.normal(0, 0.2) + (3.0 if lap == 1 else 0.0)
            laps.append({"driver_number": d, "lap_number": lap, "lap_duration": duration,
                         "date_start": iso(t), "is_pit_out_lap": lap == 9})
            t = t + timedelta(seconds=duration)
            intervals.append({"date": iso(t), "driver_number": d, "interval": 2.0 if i else None,
                              "gap_to_leader": 0 if i == 0 else 2.0 * i})
        ends[d] = t
        stints += [
            {"driver_number": d, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 8, "tyre_age_at_start": 0},
            {"driver_number": d, "stint_number": 2, "compound": "HARD", "lap_start": 9, "lap_end": total, "tyre_age_at_start": 0},
        ]
    order = sorted(ends, key=ends.get)
    for pos, d in enumerate(order, 1):
        result.append({"driver_number": d, "position": pos, "number_of_laps": total, "dnf": False, "dns": False, "dsq": False})
    drivers = [{"driver_number": i + 1, "name_acronym": f"D{i + 1}", "team_name": TEAMS[i]} for i in range(len(OFFSETS))]
    ref = RaceRef(year=year, meeting_key=1, session_key=session_key, location=location, date_start=iso(T0))
    return build_race(ref, {"drivers": drivers, "laps": laps, "stints": stints, "pit": [], "race_control": list(messages),
                            "intervals": intervals, "session_result": result, "starting_grid": []})


LEVEL1 = Level1Config(min_drivers=4, min_clean_laps=3)


# -- regime ----------------------------------------------------------------------------


def test_red_flag_closes_an_open_safety_car():
    messages = [
        {"date": "2023-11-05T17:10:00Z", "lap_number": 1, "message": "SAFETY CAR DEPLOYED", "category": "SafetyCar"},
        {"date": "2023-11-05T17:12:00Z", "lap_number": 2, "message": "RED FLAG", "flag": "RED"},
        {"date": "2023-11-05T18:00:00Z", "lap_number": 30, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "2023-11-05T18:02:00Z", "lap_number": 31, "message": "VIRTUAL SAFETY CAR ENDING"},
    ]
    events = regime_events(messages)
    assert [(e.kind, e.start_lap, e.end_lap) for e in events] == [(SC, 1, 2), (RED, 2, 3), (VSC, 30, 31)]
    assert {1, 2, 3, 4, 30, 31, 32} <= neutralised_laps(messages, 71)
    assert 40 not in neutralised_laps(messages, 71)


# -- the leakage guard -----------------------------------------------------------------------


def test_truncation_removes_the_future_and_the_label():
    race = synthetic_race()
    cut = truncate_at_lap(race, 10)
    assert cut.result == [] and cut.truncated_at == 10
    assert cut.laps["lap_number"].max() == 10
    assert cut.laps["t_end"].max() <= race.laps.loc[race.laps.lap_number == 10, "t_end"].min()
    assert all(s["lap_end"] <= 10 for s in cut.stints)
    with pytest.raises(ValueError):
        car_states(race, 90.0)  # an untruncated race is refused


def test_final_order_convention_a():
    order = final_order([
        {"driver_number": 1, "position": 1, "number_of_laps": 50},
        {"driver_number": 2, "position": None, "number_of_laps": 10},
        {"driver_number": 3, "position": None, "number_of_laps": 30},
        {"driver_number": 4, "position": None, "number_of_laps": 50, "dsq": True},
        {"driver_number": 5, "position": 2, "number_of_laps": 49, "dnf": True},
    ])
    assert order.position == {1: 1, 5: 2, 3: 3, 2: 4}
    assert order.excluded == {4} and order.retired == {2, 3, 5}


# -- the sampler -------------------------------------------------------------------------------


def run_forecast(k=10, events=False, scenarios=20000, prior=None):
    race = synthetic_race()
    cut = truncate_at_lap(race, k)
    prior = prior or Prior([f"D{i + 1}" for i in range(6)], np.zeros(6), np.eye(6) * 0.25)
    config = ForecastConfig(scenarios=scenarios, events=events, noise_scale=1.0)
    result = forecast(cut, prior, 20, level1=LEVEL1, phi=0.4, sigma_eps=0.3, tyres=TyreModel(),
                      events=EventModel(), config=config, rng=np.random.default_rng(7))
    return race, cut, prior, result


def test_matrix_is_doubly_stochastic():
    *_, result = run_forecast(events=True, scenarios=2000)
    assert np.allclose(result.matrix.sum(axis=1), 1) and np.allclose(result.matrix.sum(axis=0), 1)
    assert np.all((result.p_dnf >= 0) & (result.p_dnf <= 1))


def test_simulated_head_to_heads_match_the_closed_form():
    """With events off the totals are jointly Gaussian: pairwise frequencies must equal Φ (§4.2)."""
    race, cut, prior, result = run_forecast()
    tyres = TyreModel()
    ref = clean_laps(cut, LEVEL1).reference_seconds
    cars = car_states(cut, ref)
    mean, cov, _ = pace_posterior(cut, prior.subset([c.driver_id for c in cars], 0.5), cars, LEVEL1, 0.4, 0.3, tyres)
    sec = ref / 100
    n = np.array([20 - c.laps_done for c in cars], float)
    banked = np.array([c.banked for c in cars])
    tyre = np.array([sec * (n_i * tyres.offset["HARD"] + tyres.slope["HARD"] * (c.tyre_age * n_i + n_i * (n_i + 1) / 2))
                     for c, n_i in zip(cars, n)])
    mu = banked + n * (ref + sec * mean) + tyre
    sigma_lap = sec * 0.3 * np.sqrt(1.4 / 0.6)
    total_cov = np.outer(n * sec, n * sec) * cov + np.diag(n * sigma_lap**2)
    expected = pairwise_closed_form(mu, total_cov)
    assert np.abs(result.ahead - expected).max() < 0.02


def test_the_data_overrides_a_wrong_prior():
    wrong = Prior([f"D{i + 1}" for i in range(6)], np.array(OFFSETS[::-1]), np.eye(6) * 0.05)
    *_, result = run_forecast(k=15, prior=wrong, scenarios=2000)
    posterior = result.pace_mean - result.pace_mean.mean()
    truth = np.array(OFFSETS) - np.mean(OFFSETS)
    assert np.corrcoef(posterior, truth)[0, 1] > 0.9


def test_a_safety_car_queues_cars_by_track_position_not_lap_count():
    """At the cutoff the others are part-way through lap k: they are behind, not a lap ahead in time."""
    race = synthetic_race()
    cut = truncate_at_lap(race, 10)
    assert sorted({c.laps_done for c in car_states(cut, 90.0)}) == [9, 10]
    prior = Prior([f"D{i + 1}" for i in range(6)], np.array(OFFSETS), np.eye(6) * 1e-4)
    always_sc = EventModel(sc_hazard=0.9, vsc_hazard=0.0, red_hazard=0.0, dnf_hazard=0.0, sc_laps=[3], stop_lap_sd=0.0)
    result = forecast(cut, prior, 20, level1=LEVEL1, phi=0.4, sigma_eps=0.3, tyres=TyreModel(), events=always_sc,
                      config=ForecastConfig(scenarios=4000), rng=np.random.default_rng(3))
    leader = [d["driver_number"] for d in result.drivers].index(1)
    assert result.matrix[leader, 0] > 0.6  # fastest car, leading, restarts at the front of the queue


def test_stop_lap_formula():
    assert optimal_stop_lap(0.05, 0.05, 50, 0, 0) == pytest.approx(25)


# -- level 1 / level 2 ---------------------------------------------------------------------------


def test_level1_recovers_the_pace_order():
    races = [synthetic_race(seed=s, session_key=9000 + s) for s in range(3)]
    l1 = fit_level1(races, LEVEL1)
    one = l1.obs[l1.obs.session_key == 9000].sort_values("driver")
    assert np.corrcoef(one.theta_hat, OFFSETS)[0, 1] > 0.99
    assert (one.se > 0).all()


def test_season_filter_learns_and_round_trips():
    races = [
        RaceObservation(2024, [f"D{i + 1}" for i in range(6)], TEAMS, np.array(OFFSETS) * 0.97,
                        np.array(OFFSETS), np.full(6, 0.1), key=k)
        for k in range(8)
    ]
    state, priors, ll = replay(races, SeasonParams())
    assert np.isfinite(ll)
    first, last = priors[0], priors[7]
    assert np.abs(last.mean - OFFSETS).mean() < np.abs(first.mean - OFFSETS).mean() + 1e-9
    assert np.diag(last.cov).mean() < np.diag(first.cov).mean()
    clone = SeasonFilter.from_dict(state.to_dict())
    before = clone.x.copy()
    clone.preview(2025, TEAMS, [f"D{i + 1}" for i in range(6)], np.full(6, np.nan))
    assert np.array_equal(before, clone.x)  # preview never changes the state


# -- scores, splits ----------------------------------------------------------------------------------


def test_rps_is_zero_for_a_perfect_forecast_and_pairwise_loss_is_small():
    P = np.eye(4)
    assert np.allclose(rps_rows(P, [1, 2, 3, 4]), 0)
    assert rps_rows(np.full((4, 4), 0.25), [1, None, 3, 4])[1] != rps_rows(np.full((4, 4), 0.25), [1, None, 3, 4])[1]
    ahead = np.triu(np.ones((4, 4)), 1) * 0.999
    assert pairwise_log_loss(ahead, [1, 2, 3, 4]) < 0.01
    assert paired_bootstrap({1: -0.1, 2: -0.2, 3: -0.15})["hi"] < 0


def test_splits_and_k_grid():
    refs = [RaceRef(year, 1, k, f"L{k}", f"{year}-0{(k % 9) + 1}-01") for k, year in
            [(1, 2023), (2, 2024), (3, 2025), (4, 2025), (5, 2025)]]
    splits = assign_splits(refs, [2023, 2024], 0.5)
    assert [splits[k] for k in range(1, 6)] == ["train", "train", "validation", "validation", "test"]
    assert k_laps(50, [0.1, 0.5, 0.99]) == [("10%", 5), ("50%", 25), ("99%", 49)]


# -- registry ---------------------------------------------------------------------------------------


def test_registry_saves_promotes_and_refuses_overwrites(tmp_path):
    registry = ModelRegistry(DataStore.local(str(tmp_path)), "data/models/m7")
    artifact = ModelArtifact(version=new_version(), created_at="now", settings={}, level1={}, season={},
                             events={}, forecast={}, transition={}, races=[])
    registry.save(artifact)
    with pytest.raises(FileExistsError):
        registry.save(artifact)
    assert registry.latest_version() is None
    registry.promote(artifact.version, {"reason": "test"})
    assert registry.latest_version() == artifact.version
    assert registry.load().version == artifact.version
