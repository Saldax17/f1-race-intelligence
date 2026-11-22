"""Race model (M7): regime parsing, leakage guard, sampler vs closed form, filter, scores, registry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
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


def synthetic_race(total=20, seed=1, session_key=9001, year=2024, location="Testville", messages=(),
                   offsets=OFFSETS, teams=TEAMS, date_start=None):
    rng = np.random.default_rng(seed)
    base = 90.0
    laps, intervals, stints, result = [], [], [], []
    ends = {}
    for i, off in enumerate(offsets):
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
    drivers = [{"driver_number": i + 1, "name_acronym": f"D{i + 1}", "team_name": teams[i]} for i in range(len(offsets))]
    ref = RaceRef(year=year, meeting_key=1, session_key=session_key, location=location, date_start=date_start or iso(T0))
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
    # retired = not classified (convention (a)); car 5 is a classified DNF: it finished 2nd
    assert order.excluded == {4} and order.retired == {2, 3}
    assert order.classified_dnf == {5}


# -- one DNF definition for training and evaluation (review of PR #22) ----------------------


DNF_RESULT_CASES = [
    # driver, row, expected is_retirement
    (1, {"position": 1, "number_of_laps": 20}, False),
    (2, {"position": None, "number_of_laps": 10, "dnf": True}, True),     # retired, unclassified
    (3, {"position": 6, "number_of_laps": 19, "dnf": True}, False),       # classified despite dnf=True
    (4, {"position": None, "number_of_laps": 20, "dsq": True}, False),    # DSQ: excluded, not a retirement
    (5, {"position": None, "number_of_laps": 0, "dns": True}, False),     # DNS: excluded
    (6, {"position": None, "number_of_laps": 12}, True),                  # unclassified without the flag
]


@pytest.mark.parametrize("driver, row, expected", DNF_RESULT_CASES)
def test_is_retirement_is_not_classified_and_not_excluded(driver, row, expected):
    from f1_race_intelligence.race_model.racedata import is_retirement

    assert is_retirement({"driver_number": driver, **row}) is expected


def test_evaluation_and_hazard_fit_count_the_same_retirements():
    """final_order (evaluation), fit_events (hazard) and the level-1 table agree on who retired."""
    from f1_race_intelligence.race_model.events import fit_events
    from f1_race_intelligence.race_model.racedata import is_retirement

    race = synthetic_race(total=20)
    result = [{"driver_number": d, **row} for d, row, _ in DNF_RESULT_CASES]
    race.result = result
    expected = {r["driver_number"] for r in result if is_retirement(r)}

    order = final_order(result)
    assert set(order.retired) == expected == {2, 6}
    events = fit_events([race], TyreModel())
    assert events.counts["retirements"] == len(expected)  # both retired after lap 1
    obs = fit_level1([race], LEVEL1).obs.set_index("driver")
    assert {d for d in obs.index if obs.loc[d, "retired"]} == expected

    # a classified DNF counts as exposure (laps run) but never as an event
    only_classified_dnf = [{"driver_number": 3, "position": 1, "number_of_laps": 19, "dnf": True}]
    race.result = only_classified_dnf
    assert fit_events([race], TyreModel()).counts["retirements"] == 0
    assert final_order(only_classified_dnf).retired == frozenset()


# -- pooled level-1 noise comes from training races only (review of PR #22) -----------------


def _wreck(race, scale=0.2, seed=0):
    """The reviewers' dynamic check: validation lap times made wildly noisy (sigma_eps ~10 if pooled)."""
    rng = np.random.default_rng(seed)
    race.laps["lap_duration"] = race.laps["lap_duration"] * (1 + rng.normal(0, scale, len(race.laps))).clip(0.5)
    return race


def test_restricting_tyre_races_alone_restricts_phi_and_sigma_too():
    """Regression for the second review of PR #22: passing only tyre_races=train_keys must already
    keep validation data out of phi and sigma_eps (no separate noise_races needed)."""
    # One training and one validation race, as in the review: sigma_eps is a median over races, and
    # the median of two is their mean, so pooling a wrecked validation race moves it ~50x.
    train = [synthetic_race(seed=1, session_key=400)]
    keys = {r.ref.session_key for r in train}
    clean_val = [synthetic_race(seed=60, session_key=500)]
    wrecked_val = [_wreck(synthetic_race(seed=60, session_key=500))]
    a = fit_level1(train + clean_val, LEVEL1, tyre_races=keys)
    b = fit_level1(train + wrecked_val, LEVEL1, tyre_races=keys)
    assert (a.phi, a.sigma_eps, a.tyres.to_dict()) == (b.phi, b.sigma_eps, b.tyres.to_dict())
    assert a.sigma_eps == fit_level1(train, LEVEL1).sigma_eps  # exactly the training-only estimate
    pooled = fit_level1(train + wrecked_val, LEVEL1).sigma_eps  # the old behaviour
    assert pooled > 10 * a.sigma_eps


def test_level1_noise_is_fitted_on_the_given_races_only():
    train = [synthetic_race(seed=s, session_key=200 + s) for s in range(3)]
    other = [synthetic_race(seed=50, session_key=300)]
    noisy = [synthetic_race(seed=51, session_key=300)]
    noisy[0].laps["lap_duration"] = noisy[0].laps["lap_duration"] * (1 + np.random.default_rng(0).normal(0, 0.01, len(noisy[0].laps)))
    keys = {r.ref.session_key for r in train}
    a = fit_level1(train + other, LEVEL1, tyre_races=keys, noise_races=keys)
    b = fit_level1(train + noisy, LEVEL1, tyre_races=keys, noise_races=keys)
    assert (a.phi, a.sigma_eps) == (b.phi, b.sigma_eps)
    assert a.tyres.to_dict() == b.tyres.to_dict()
    pooled = fit_level1(train + noisy, LEVEL1)  # without the restriction the noisy race moves sigma
    assert pooled.sigma_eps != a.sigma_eps


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


def test_splits_are_fixed_dates_and_never_move_when_races_are_added():
    refs = [RaceRef(2023, 1, 1, "A", "2023-03-01"), RaceRef(2025, 1, 2, "B", "2025-04-01"),
            RaceRef(2025, 1, 3, "C", "2025-09-01"), RaceRef(2026, 1, 4, "D", "2026-03-01")]
    splits = assign_splits(refs, [2023, 2024], "2025-08-01", "2026-01-01")
    assert [splits[k] for k in range(1, 5)] == ["train", "validation", "test", "future"]
    more = refs + [RaceRef(2026, 1, k, f"E{k}", f"2026-0{k - 1}-15") for k in range(5, 9)]
    assert assign_splits(more, [2023, 2024], "2025-08-01", "2026-01-01")[3] == "test"
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
