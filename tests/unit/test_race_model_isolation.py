"""The test set is sealed: no development path may read a test race, and the final look is single-use.

Every test race in the fake catalog is "poisoned": loading it while it is not
explicitly unsealed fails the test. The stages run end to end on synthetic
races, so a new code path that opens a test race's raw data is caught here.
"""

from __future__ import annotations

import pytest

from f1_race_intelligence.config.settings import AppSettings
from f1_race_intelligence.race_model.racedata import RaceRef
from f1_race_intelligence.race_model.runner import TEST, RaceModelRunner
from f1_race_intelligence.storage.store import DataStore

from tests.unit.test_race_model import _wreck, synthetic_race

OFFSETS8 = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
TEAMS8 = ["A", "A", "B", "B", "C", "C", "D", "D"]

CALENDAR = [  # (session_key, year, date)
    (1, 2023, "2023-03-05"), (2, 2023, "2023-05-07"), (3, 2023, "2023-07-09"),
    (4, 2024, "2024-03-02"), (5, 2024, "2024-05-05"), (6, 2024, "2024-07-07"),
    (7, 2025, "2025-03-16"), (8, 2025, "2025-05-18"),          # validation
    (9, 2025, "2025-09-07"), (10, 2025, "2025-11-09"),         # test
]
TEST_KEYS = {9, 10}


VALIDATION_KEYS = {7, 8}


class PoisonedCatalog:
    """Serves synthetic races; a test race may be read only while ``unsealed``.

    ``altered_validation`` serves completely different validation races
    (reversed pace order, other noise), for the leakage regression test.
    """

    def __init__(self, altered_validation: bool = False):
        self.refs = [RaceRef(y, k, k, f"Track{k % 4}", f"{d}T13:00:00+00:00") for k, y, d in CALENDAR]
        self.unsealed = False
        self.reads = []
        self.altered_validation = altered_validation

    def races(self, years=None, exclude=()):
        return [r for r in self.refs if years is None or r.year in years]

    def load(self, ref):
        if ref.session_key in TEST_KEYS and not self.unsealed:
            pytest.fail(f"test race {ref.tag} was read while the test set is sealed")
        self.reads.append(ref.session_key)
        altered = self.altered_validation and ref.session_key in VALIDATION_KEYS
        race = synthetic_race(seed=ref.session_key + (1000 if altered else 0), session_key=ref.session_key,
                              year=ref.year, location=ref.location,
                              offsets=OFFSETS8[::-1] if altered else OFFSETS8, teams=TEAMS8,
                              date_start=ref.date_start)
        # as in the review's dynamic check: wildly noisy validation lap times
        return _wreck(race, seed=ref.session_key) if altered else race


@pytest.fixture
def runner(tmp_path):
    settings = AppSettings.model_validate({"race_model": {
        "years": [2023, 2024, 2025], "train_years": [2023, 2024],
        "validation_until": "2025-08-01", "test_until": "2026-01-01",
        "burn_in_races": 1, "scenarios": 200, "noise_scales": [1.0], "min_clean_laps": 3,
        "output_path": "models",
    }})
    run = RaceModelRunner(settings, DataStore.local(str(tmp_path)), DataStore.local(str(tmp_path)))
    run.catalog = PoisonedCatalog()
    return run


def test_development_never_reads_a_test_race(runner):
    trained = runner.train()
    assert trained.ok, trained.reason
    artifact = runner.registry.load(trained.summary["version"])
    assert {r["split"] for r in artifact.races} == {"train", "validation"}
    assert not TEST_KEYS & {int(k) for k in artifact.season["priors"]}
    assert not TEST_KEYS & set(artifact.season["state"]["races_seen"])

    validated = runner.evaluate(split="validation", promote=True)
    assert validated.summary["races"] == 2
    runner.registry.promote(artifact.version, {"reason": "test"})  # synthetic races need not beat the baseline
    assert runner.update().ok
    assert not TEST_KEYS & set(runner.catalog.reads)


def test_changing_validation_data_does_not_change_any_fitted_parameter(tmp_path):
    """Regression for the PR #22 review: phi and sigma_eps (and every other fitted
    quantity) must come from training races only. Test races are never read at all."""
    settings = AppSettings.model_validate({"race_model": {
        "years": [2023, 2024, 2025], "train_years": [2023, 2024],
        "validation_until": "2025-08-01", "test_until": "2026-01-01",
        "burn_in_races": 1, "scenarios": 200, "noise_scales": [1.0, 2.0], "calibration_races": 2,
        "min_clean_laps": 3, "output_path": "models",
    }})

    def fitted(altered, sub):
        store = DataStore.local(str(tmp_path / sub))
        run = RaceModelRunner(settings, store, store)
        run.catalog = PoisonedCatalog(altered_validation=altered)
        result = run.train()
        assert result.ok, result.reason
        a = run.registry.load(result.summary["version"])
        return {
            "phi": a.level1["phi"], "sigma_eps": a.level1["sigma_eps"], "tyres": a.level1["tyres"],
            "tyres_by_location": a.level1["tyres_by_location"],
            "season_params": a.season["state"]["params"], "events": a.events, "transition": a.transition,
            "noise": (a.forecast["noise_scale"], a.forecast.get("noise_scale_by_fraction")),
        }, a

    original, art_o = fitted(False, "original")
    altered, art_a = fitted(True, "altered")
    for name in original:
        assert original[name] == altered[name], f"{name} changed when only validation data changed"
    # the online replay does see validation results, so their pre-race priors may differ: that is allowed
    assert set(art_o.season["priors"]) == set(art_a.season["priors"])


def test_the_test_set_refuses_every_entry_but_the_explicit_one(runner):
    assert runner.train().ok
    sealed = runner.evaluate(split=TEST)
    assert not sealed.ok and "sealed" in sealed.reason
    assert not runner.evaluate(split=TEST, unseal_test=True, promote=True).ok  # never promotes on test
    runner.registry.promote(runner.registry.versions()[-1], {"reason": "test"})
    forecast = runner.forecast(9, lap=10, year=2025)
    assert not forecast.ok and "sealed" in forecast.reason
    assert runner.test_ledger() == []


def test_the_final_look_is_ledgered_and_single_use(runner):
    assert runner.train().ok
    runner.catalog.unsealed = True  # only from here on may the fake catalog serve test races
    first = runner.evaluate(split=TEST, unseal_test=True)
    assert first.ok, first.reason
    assert first.summary["races"] == 2
    assert [e["look"] for e in runner.test_ledger()] == [1]

    second = runner.evaluate(split=TEST, unseal_test=True)
    assert not second.ok and "already evaluated" in second.reason
    assert len(runner.test_ledger()) == 1
    assert runner.evaluate(split=TEST, unseal_test=True, allow_repeat_test=True).ok
    assert [e["look"] for e in runner.test_ledger()] == [1, 2]


def test_test_priors_use_only_races_that_finished_before(runner):
    """The online replay at unseal time: race 10's prior includes race 9, race 9's does not include 10."""
    assert runner.train().ok
    runner.catalog.unsealed = True
    artifact = runner._newest()
    model, races = runner._open_test(artifact, allow_repeat=False)
    assert [r.ref.session_key for r in races] == [9, 10]
    assert {9, 10} <= set(model.priors)
    # frozen hyperparameters: the replay never refits on test races
    assert model.phi == artifact.level1["phi"] and model.sigma_eps == artifact.level1["sigma_eps"]
    assert model.season.params.to_dict() == artifact.season["state"]["params"]
    assert model.events.to_dict() == artifact.events
