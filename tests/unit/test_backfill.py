"""R2: historical backfill by season, with the M6 completeness gate.

Everything runs through the real CLI (plan with F1_EXTRACTION_YEARS, then M2->M5
per planned session, then M6 --require-complete-history) against a fake OpenF1
client that serves three synthetic seasons. Nothing is downloaded and no stage
report is fabricated.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import load_settings
from f1_race_intelligence.ingestion.exceptions import OpenF1HTTPError
from f1_race_intelligence.ingestion.openf1 import F1Client
from f1_race_intelligence.pipelines import incremental as inc
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.storage.store import DataStore
from factories import (
    MEETING_RECORD,
    SESSION_RECORD,
    car_data_record,
    driver_record,
    interval_record,
    lap_record,
    position_record,
    stint_record,
    weather_record,
)

SEASONS = (2023, 2024, 2025)
DRIVERS = (1, 44)
ENDPOINTS = ["drivers", "laps", "car_data", "position", "intervals", "stints", "pit", "weather", "race_control"]
EXIT_DEFERRED = cli.EXIT_DEFERRED


def meeting_key(year):
    return year * 10


def session_keys(year):
    return (year * 10 + 1, year * 10 + 2)


def shift(record, days):
    shifted = dict(record)
    for name in ("date", "date_start", "date_end"):
        if isinstance(shifted.get(name), str):
            shifted[name] = (datetime.fromisoformat(shifted[name]) + timedelta(days=days)).isoformat()
    return shifted


def rekey(record, year, session_key):
    days = (year - 2023) * 365 + (session_key - year * 10 - 1) * 14
    shifted = shift(record, days)
    if "session_key" in shifted:
        shifted["session_key"] = session_key
    if "meeting_key" in shifted:
        shifted["meeting_key"] = meeting_key(year)
    return shifted


class SeasonClient:
    """A fake OpenF1 serving two synthetic race sessions per season."""

    def __init__(self, failing_sessions=()):
        self.failing_sessions = set(failing_sessions)
        self.calls = []
        self.client = MagicMock(spec=F1Client)
        c = self.client
        c.get_meetings.side_effect = self._meetings
        c.get_sessions.side_effect = self._sessions
        c.get_drivers.side_effect = lambda session_key: self._rows(session_key, [driver_record(d, f"D{d}", f"Team {d}") for d in DRIVERS])
        c.get_laps.side_effect = self._laps
        c.get_stints.side_effect = lambda session_key: self._rows(session_key, [stint_record(d, 1, 1, 3) for d in DRIVERS])
        c.get_positions.side_effect = lambda session_key: self._rows(session_key, [position_record(d, -10, i + 1) for i, d in enumerate(DRIVERS)])
        c.get_weather.side_effect = lambda session_key: self._rows(session_key, [weather_record(o, air_temperature=18.0 + session_key % 13, track_temperature=25.0 + session_key % 17) for o in (-30, 70, 170)])
        c.get_car_data.side_effect = self._car_data
        c.get_intervals.side_effect = lambda session_key: self._rows(
            session_key,
            [interval_record(d, offset, gap_to_leader=1.5 + (session_key % 11) * 0.2 + i, interval=0.8 + (session_key % 7) * 0.1)
             for i, d in enumerate(DRIVERS) for offset in (10, 110, 210)],
        )
        c.get_pit.return_value = []
        c.get_race_control.return_value = []

    @staticmethod
    def _year_of(session_key):
        return session_key // 10

    def _rows(self, session_key, records):
        self.calls.append(session_key)
        return [rekey(record, self._year_of(session_key), session_key) for record in records]

    def _car_data(self, session_key, driver_number):
        seed = session_key % 89
        return self._rows(
            session_key,
            [car_data_record(driver_number, offset, speed=230 + seed % 40 + (offset // 100) * 3 + driver_number % 7,
                             throttle=60 + seed % 30, rpm=10000 + seed * 11)
             for offset in (10, 60, 110, 160, 210, 260)],
        )

    def _laps(self, session_key):
        if session_key in self.failing_sessions:
            raise OpenF1HTTPError(status_code=503, message="unavailable", endpoint="laps")
        # Lap times differ per session, driver and lap, as real data does: identical sessions
        # would make the anti-leakage statistics degenerate (and the guard rightly refuses that).
        seed = session_key % 97
        return self._rows(
            session_key,
            [lap_record(d, lap, lap_duration=88.0 + lap * 1.7 + seed * 0.31 + d * 0.13 + (lap * seed) % 5 * 0.2)
             for d in DRIVERS for lap in (1, 2, 3)],
        )

    def _meetings(self, year):
        record = {**MEETING_RECORD, "meeting_key": meeting_key(year), "year": year}
        return [shift(record, (year - 2023) * 365)]

    def _sessions(self, year, meeting_key):
        return [
            rekey({**SESSION_RECORD, "year": year}, year, key) for key in session_keys(year)
        ]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path_factory):
    for name in ("F1_ENVIRONMENT", "F1_STORAGE_BACKEND", "F1_STORAGE_BUCKET", "F1_STORAGE_LAYOUT",
                 "F1_CONFIG_PATH", "F1_EXTRACTION_YEARS"):
        monkeypatch.delenv(name, raising=False)
    cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(cwd)
    # M6 records the revision it ran at; pin it so reports are valid and reuse is possible.
    import f1_race_intelligence.modeling.runner as runner_module

    monkeypatch.setattr(runner_module, "source_revision", lambda: ("a" * 40, False))
    monkeypatch.setattr(runner_module, "dependency_lock_sha256", lambda: "b" * 64)
    yield cwd
    assert not (cwd / "manifests").exists()


@pytest.fixture
def config(tmp_path):
    raw = tmp_path / "raw"
    data = {
        "historical_extraction": {"years": list(SEASONS), "session_types": ["Race"], "endpoints": ENDPOINTS,
                                  "output_path": str(raw), "manifest_path": str(tmp_path / "manifests_m2")},
        "validation": {"raw_path": str(raw), "manifest_path": str(tmp_path / "manifests_m2"), "report_path": str(tmp_path / "validation")},
        "consolidation": {"raw_path": str(raw), "output_path": str(tmp_path / "processed"),
                          "report_path": str(tmp_path / "consolidation"), "validation_path": str(tmp_path / "validation")},
        "features": {"input_path": str(tmp_path / "processed"), "output_path": str(tmp_path / "features"),
                     "report_path": str(tmp_path / "feature_reports")},
        "modeling": {"input_path": str(tmp_path / "features"), "output_path": str(tmp_path / "modeling"),
                     "report_path": str(tmp_path / "modeling_reports"), "years": list(SEASONS),
                     "split_strategy": "fraction", "train_fraction": 1 / 3, "validation_fraction": 1 / 3,
                     "min_sessions_per_split": 1},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# -- helpers ------------------------------------------------------------------------


def plan_phase(config, tmp_path, client, years, monkeypatch):
    """The plan task of one phase: F1_EXTRACTION_YEARS limits it, exactly as the ASL sets it."""
    if years is not None:
        monkeypatch.setenv("F1_EXTRACTION_YEARS", years)
    output = tmp_path / f"plan-{years or 'default'}.json"
    code = cli.main(["--stage", "plan", "--config", str(config), "--plan-output", str(output)],
                    client_factory=lambda settings: client.client)
    monkeypatch.delenv("F1_EXTRACTION_YEARS", raising=False)
    assert code == 0
    return json.loads(output.read_text(encoding="utf-8"))


def run_session(config, client, unit):
    scope = ["--year", str(unit["year"]), "--meeting-key", str(unit["meeting_key"]), "--session-key", str(unit["session_key"]), "--config", str(config)]
    codes = [cli.main(["--stage", "m2", *scope], client_factory=lambda settings: client.client)]
    if codes[0] != 0:
        return codes
    for stage in ("m3", "m4", "m5"):
        extra = ["--overwrite-derived"] if stage in ("m4", "m5") else []
        codes.append(cli.main(["--stage", stage, *scope, *extra]))
    return codes


def run_phase(config, tmp_path, client, years, monkeypatch):
    plan = plan_phase(config, tmp_path, client, years, monkeypatch)
    results = {unit["session_key"]: run_session(config, client, unit) for unit in plan["sessions"]}
    return plan, results


def history(config, years=SEASONS):
    settings = load_settings(str(config))
    return inc.history_completeness(DataStore.from_settings(settings), settings, list(years))


def m6(config, *flags):
    return cli.main(["--stage", "m6", "--config", str(config), *flags])


def modeling_reports(tmp_path):
    return sorted((tmp_path / "modeling_reports").rglob("modeling_report_*.json")) if (tmp_path / "modeling_reports").exists() else []


def tree_digest(tmp_path, *folders):
    digest = {}
    for folder in folders:
        for path in sorted((tmp_path / folder).rglob("*")):
            if path.is_file():
                digest[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def complete_history(config, tmp_path, client, monkeypatch):
    for year in SEASONS:
        _, results = run_phase(config, tmp_path, client, str(year), monkeypatch)
        assert all(codes == [0, 0, 0, 0] for codes in results.values()), results


# -- A/E: one season, scoped through F1_EXTRACTION_YEARS --------------------------------


def test_single_season_phase_plans_and_processes_only_that_season(config, tmp_path, monkeypatch):
    client = SeasonClient()
    plan, results = run_phase(config, tmp_path, client, "2024", monkeypatch)

    assert {unit["year"] for unit in plan["sessions"]} == {2024}
    assert sorted(results) == list(session_keys(2024))
    assert all(codes == [0, 0, 0, 0] for codes in results.values())

    state = history(config)
    assert state["seasons"]["2024"]["status"] == "complete" and state["seasons"]["2024"]["sessions"] == 2
    assert state["seasons"]["2023"]["status"] == "incomplete"
    assert state["seasons"]["2025"]["status"] == "incomplete"
    assert state["complete"] is False
    # nothing of the other seasons was requested from OpenF1
    assert {key // 10 for key in client.calls} == {2024}


def test_default_plan_without_scope_still_covers_the_configured_seasons(config, tmp_path, monkeypatch):
    plan = plan_phase(config, tmp_path, SeasonClient(), None, monkeypatch)
    assert sorted({unit["year"] for unit in plan["sessions"]}) == list(SEASONS)


# -- F/G/B: M6 gate -----------------------------------------------------------------------


def test_m6_is_deferred_while_any_season_is_missing_and_nothing_is_modelled(config, tmp_path, monkeypatch, capsys):
    client = SeasonClient()
    run_phase(config, tmp_path, client, "2023", monkeypatch)
    run_phase(config, tmp_path, client, "2024", monkeypatch)
    capsys.readouterr()

    assert m6(config, "--require-complete-history") == EXIT_DEFERRED
    err = capsys.readouterr().err
    assert "m6: DEFERRED" in err and "2025" in err
    assert modeling_reports(tmp_path) == []  # no modelling artefacts were produced
    assert not (tmp_path / "modeling").exists()


def test_sequential_phases_complete_the_history_and_only_then_m6_runs(config, tmp_path, monkeypatch, capsys):
    client = SeasonClient()
    outcomes = []
    for year in SEASONS:
        _, results = run_phase(config, tmp_path, client, str(year), monkeypatch)
        assert all(codes == [0, 0, 0, 0] for codes in results.values())
        outcomes.append(m6(config, "--require-complete-history"))
    capsys.readouterr()

    assert outcomes == [EXIT_DEFERRED, EXIT_DEFERRED, 0]
    reports = modeling_reports(tmp_path)
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["verdict"] in {"VALID", "VALID_WITH_WARNINGS"}
    assert report["source"]["sessions"] == 6  # the whole consolidated history, not the last season
    assert history(config)["complete"] is True


def test_m6_after_complete_history_is_idempotent_and_keeps_its_run_id(config, tmp_path, monkeypatch, capsys):
    complete_history(config, tmp_path, SeasonClient(), monkeypatch)
    assert m6(config, "--require-complete-history") == 0
    first = json.loads(modeling_reports(tmp_path)[0].read_text(encoding="utf-8"))["run_id"]
    capsys.readouterr()

    assert m6(config, "--require-complete-history") == 0
    assert "reused_from_run_id" in capsys.readouterr().err
    assert len(modeling_reports(tmp_path)) == 1
    assert json.loads(modeling_reports(tmp_path)[0].read_text(encoding="utf-8"))["run_id"] == first


def test_without_the_flag_m6_keeps_its_previous_behaviour(config, tmp_path, monkeypatch):
    run_phase(config, tmp_path, SeasonClient(), "2023", monkeypatch)
    assert m6(config) != EXIT_DEFERRED  # the gate is opt-in; the ASL always opts in


def test_the_history_flag_is_only_valid_for_m6(config):
    assert cli.main(["--stage", "m3", "--require-complete-history", "--config", str(config)]) == cli.EXIT_USAGE


# -- B/C/D: sequential, resumable, idempotent ---------------------------------------------


def test_a_failed_phase_resumes_without_touching_completed_sessions(config, tmp_path, monkeypatch):
    first_key, second_key = session_keys(2024)
    failing = SeasonClient(failing_sessions={second_key})
    _, results = run_phase(config, tmp_path, failing, "2024", monkeypatch)
    assert results[first_key] == [0, 0, 0, 0] and results[second_key][0] == cli.EXIT_FAILED

    assert history(config, [2024])["seasons"]["2024"]["status"] == "incomplete"  # failed state is never "processed"
    before = {k: v for k, v in tree_digest(tmp_path, "raw", "processed", "features", "manifests_m2/incremental").items() if str(first_key) in k}

    healthy = SeasonClient()
    plan, results = run_phase(config, tmp_path, healthy, "2024", monkeypatch)
    assert [unit["session_key"] for unit in plan["sessions"]] == [second_key]  # only the failed one is replanned
    assert {d["session_key"]: d["decision"] for d in plan["decisions"]} == {first_key: "omit", second_key: "include"}
    assert results == {second_key: [0, 0, 0, 0]}
    assert history(config, [2024])["seasons"]["2024"]["status"] == "complete"
    after = {k: v for k, v in tree_digest(tmp_path, "raw", "processed", "features", "manifests_m2/incremental").items() if str(first_key) in k}
    assert after == before  # the completed session was not rewritten
    assert first_key not in healthy.calls


def test_rerunning_a_completed_season_changes_nothing_and_duplicates_nothing(config, tmp_path, monkeypatch):
    run_phase(config, tmp_path, SeasonClient(), "2023", monkeypatch)
    before = tree_digest(tmp_path, "raw", "processed", "features", "manifests_m2/incremental", "consolidation", "feature_reports", "validation")

    again = SeasonClient()
    plan, results = run_phase(config, tmp_path, again, "2023", monkeypatch)

    assert plan["sessions"] == [] and results == {}
    assert {d["decision"] for d in plan["decisions"]} == {"omit"}
    assert tree_digest(tmp_path, "raw", "processed", "features", "manifests_m2/incremental", "consolidation", "feature_reports", "validation") == before
    assert set(again.calls) == set()  # not even session data was requested


def test_phase_state_is_shared_whatever_seasons_a_run_names(config, tmp_path, monkeypatch):
    """years selects seasons; it must not split the state recorded by different phases."""
    settings = load_settings(str(config))
    store = DataStore.from_settings(settings)
    narrow = load_settings(str(config))
    narrow.historical_extraction.years = [2024]
    assert inc._config_fingerprint(settings, store) == inc._config_fingerprint(narrow, store)
    changed = load_settings(str(config))
    changed.features.rolling_windows = [2]
    assert inc._config_fingerprint(settings, store) != inc._config_fingerprint(changed, store)


# -- H: missing / invalid / altered evidence ------------------------------------------------


def _tamper_targets(tmp_path):
    key = session_keys(2024)[0]
    return {
        "state_removed": lambda: next((tmp_path / "manifests_m2" / "incremental").rglob(f"session={key}/state.json")).unlink(),
        "state_invalid_json": lambda: next((tmp_path / "manifests_m2" / "incremental").rglob(f"session={key}/state.json")).write_text("{not json", encoding="utf-8"),
        "stages_removed": lambda: next((tmp_path / "manifests_m2" / "incremental").rglob(f"session={key}/stages.json")).unlink(),
        "raw_altered": lambda: _append(next(p for p in (tmp_path / "raw").rglob(f"*{key}*.json") if "laps" in str(p))),
        "feature_altered": lambda: _append(next((tmp_path / "features" / "2024").glob(f"session_{key}.parquet"))),
        "m2_manifest_removed": lambda: [p.unlink() for p in (tmp_path / "manifests_m2").rglob("manifest_*.json") if _has_session(p, key)],
        "m3_report_altered": lambda: _append(next(p for p in (tmp_path / "validation").glob("validation_report_*.json") if _validation_session(p) == key)),
    }


def _append(path):
    path.write_bytes(path.read_bytes() + b" ")


def _has_session(path, key):
    try:
        return any(e.get("session_key") == key and e.get("endpoint") == "laps" for e in json.loads(path.read_text(encoding="utf-8")).get("entries", []))
    except ValueError:
        return False


def _validation_session(path):
    return json.loads(path.read_text(encoding="utf-8")).get("source", {}).get("filter", {}).get("session_key")


@pytest.mark.parametrize("tamper", ["state_removed", "state_invalid_json", "stages_removed", "raw_altered",
                                    "feature_altered", "m2_manifest_removed", "m3_report_altered"])
def test_missing_invalid_or_altered_evidence_defers_m6_and_marks_the_season_incomplete(config, tmp_path, monkeypatch, capsys, tamper):
    complete_history(config, tmp_path, SeasonClient(), monkeypatch)
    assert history(config)["complete"] is True

    _tamper_targets(tmp_path)[tamper]()

    state = history(config)
    assert state["complete"] is False and state["seasons"]["2024"]["status"] == "incomplete", tamper
    assert [item["session_key"] for item in state["seasons"]["2024"]["pending"]] == [session_keys(2024)[0]]
    capsys.readouterr()
    assert m6(config, "--require-complete-history") == EXIT_DEFERRED
    assert modeling_reports(tmp_path) == []


def test_a_missing_catalogue_is_unknown_not_empty(config, tmp_path):
    state = history(config, [2023])
    assert state["complete"] is False
    assert "catalogue" in state["seasons"]["2023"]["reason"]


def test_history_verification_never_reaches_openf1(config, tmp_path, monkeypatch):
    run_phase(config, tmp_path, SeasonClient(), "2023", monkeypatch)
    settings = load_settings(str(config))
    with HistoricalExtractionPipeline(settings, client=inc._NoClient(), store=DataStore.from_settings(settings)) as pipeline:
        units, reason = pipeline.cached_work_units(2023)
        assert reason is None and len(units) == 2
        assert pipeline.cached_work_units(2025)[1] is not None
    with pytest.raises(RuntimeError, match="must not call OpenF1"):
        inc._NoClient().get_laps
