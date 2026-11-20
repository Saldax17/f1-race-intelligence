"""End-to-end incremental planning over the REAL stages M2->M5 (CLI), with a fake OpenF1 client.

No stage report is fabricated: M2, M3, M4 and M5 write their own manifests and
reports, and the planner is asked afterwards whether the session may be omitted.
"""

import json
from pathlib import Path

import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import load_settings
from f1_race_intelligence.ingestion.exceptions import OpenF1Error, OpenF1HTTPError
from f1_race_intelligence.pipelines import incremental as inc
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.layout import Layer
from f1_race_intelligence.storage.store import DataStore
from factories import MEETING_KEY, SESSION_KEY, YEAR, car_data_record
from test_consolidation import build_raw  # noqa: F401 - fixture
from test_historical_extraction import make_client

DEFAULT_ENDPOINTS = ["drivers", "laps", "car_data", "position", "intervals", "stints", "pit", "weather", "race_control"]
NO_CAR_DATA = [name for name in DEFAULT_ENDPOINTS if name != "car_data"]
SCOPE = ["--year", str(YEAR), "--meeting-key", str(MEETING_KEY), "--session-key", str(SESSION_KEY)]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path_factory):
    for name in ("F1_ENVIRONMENT", "F1_STORAGE_BACKEND", "F1_STORAGE_BUCKET", "F1_STORAGE_LAYOUT", "F1_CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)
    # Any state written relative to the process directory would land here.
    cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(cwd)
    yield cwd
    assert not (cwd / "manifests").exists(), "incremental state escaped the configured store"


def write_config(tmp_path: Path, raw_root: Path, endpoints) -> Path:
    config = {
        "historical_extraction": {"years": [YEAR], "session_types": ["Race"], "endpoints": endpoints,
                                  "output_path": str(raw_root), "manifest_path": str(tmp_path / "manifests_m2")},
        "validation": {"raw_path": str(raw_root), "manifest_path": str(tmp_path / "manifests_m2"),
                       "report_path": str(tmp_path / "validation")},
        "consolidation": {"raw_path": str(raw_root), "output_path": str(tmp_path / "processed"),
                          "report_path": str(tmp_path / "consolidation"), "validation_path": str(tmp_path / "validation")},
        "features": {"input_path": str(tmp_path / "processed"), "output_path": str(tmp_path / "features"),
                     "report_path": str(tmp_path / "feature_reports")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def fake_client(pit_error=None):
    client = make_client()
    client.get_car_data.side_effect = lambda *a, **kw: [car_data_record(kw.get("driver_number", 1), 10)]
    if pit_error is not None:
        client.get_pit.side_effect = pit_error
    return client


def run_stages(config: Path, client, stages=("m2", "m3", "m4", "m5")):
    codes = []
    for stage in stages:
        extra = ["--overwrite-derived"] if stage in ("m4", "m5") else []
        kwargs = {"client_factory": lambda settings: client} if stage == "m2" else {}
        codes.append(cli.main(["--stage", stage, *SCOPE, "--config", str(config), *extra], **kwargs))
    return codes


def context(config: Path):
    settings = load_settings(str(config))
    return DataStore.from_settings(settings), settings, SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=SESSION_KEY)


def reusable(config: Path):
    return inc.session_reuse_decision(*context(config))


def plan(config: Path):
    settings = load_settings(str(config))
    store = DataStore.from_settings(settings)
    pipeline = HistoricalExtractionPipeline(settings, client=make_client(), store=store)
    units = pipeline.plan()
    return units, pipeline.plan_decisions


def m2_manifest(tmp_path: Path) -> Path:
    """The session run's manifest (planning writes its own, entry-less manifests)."""
    candidates = [
        path
        for path in sorted((tmp_path / "manifests_m2").rglob("manifest_*.json"))
        if any(e.get("session_key") == SESSION_KEY and e.get("endpoint") == "laps" for e in json.loads(path.read_text(encoding="utf-8")).get("entries", []))
    ]
    return candidates[-1]


@pytest.mark.parametrize("endpoints", [DEFAULT_ENDPOINTS, NO_CAR_DATA], ids=["default", "without_car_data"])
def test_real_chain_registers_completion_and_second_plan_omits_the_session(build_raw, raw_root, tmp_path, endpoints):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, endpoints)

    units, _ = plan(config)
    assert [u["session_key"] for u in units] == [SESSION_KEY]  # first plan: pending

    assert run_stages(config, fake_client()) == [0, 0, 0, 0]

    store, settings, scope = context(config)
    m2_key, manifest = inc._latest_session_report(
        store, store.root(Layer.EXTRACTION_MANIFESTS, settings.historical_extraction.manifest_path), scope, "historical_extraction"
    )
    raw, absent = inc._raw_artifacts(store, settings, scope, m2_key, manifest)
    keys = [item["key"] for item in raw]
    assert keys == sorted(set(keys)) and absent == []  # each storage key exactly once, in a stable order
    if "car_data" in endpoints:
        assert sum("drivers" in key for key in keys) == 1
        assert sum("car_data" in key for key in keys) == 2

    assert reusable(config) == (True, "all M2-M5 artifacts and provenance match")
    units, decisions = plan(config)
    assert units == []
    assert [d["decision"] for d in decisions] == ["omit"]

    state = list((tmp_path / "manifests_m2" / "incremental").rglob("state.json"))
    assert len(state) == 1  # state lives under the configured store, not the working directory


def test_second_complete_chain_keeps_the_session_omitted(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, DEFAULT_ENDPOINTS)
    run_stages(config, fake_client())
    assert run_stages(config, fake_client()) == [0, 0, 0, 0]
    assert reusable(config)[0] is True


def test_legitimate_404_is_recorded_accepted_omitted_and_evidence_loss_reincludes_it(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=None)
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    error = OpenF1HTTPError(status_code=404, message="No results found.", endpoint="pit")
    assert run_stages(config, fake_client(error)) == [0, 0, 0, 0]

    entries = [e for e in json.loads(m2_manifest(tmp_path).read_text(encoding="utf-8"))["entries"] if e["endpoint"] == "pit"]
    assert [(e["status"], e["status_code"], e["error_type"], e["session_key"]) for e in entries] == [
        ("failed", 404, "OpenF1HTTPError", SESSION_KEY)
    ]
    assert reusable(config)[0] is True
    assert plan(config)[0] == []

    store, settings, scope = context(config)
    state = json.loads(next((tmp_path / "manifests_m2" / "incremental").rglob("state.json")).read_text(encoding="utf-8"))
    assert [(a["endpoint"], a["status_code"], a["retryable"]) for a in state["absent_endpoints"]] == [("pit", 404, False)]

    manifest_path = m2_manifest(tmp_path)
    original = manifest_path.read_text(encoding="utf-8")
    mutations = {
        "status 500": lambda d: [e.update(status_code=500) for e in d["entries"] if e["endpoint"] == "pit"],
        "no status (timeout)": lambda d: [e.update(status_code=None) for e in d["entries"] if e["endpoint"] == "pit"],
        "entry removed": lambda d: d.update(entries=[e for e in d["entries"] if e["endpoint"] != "pit"]),
        "other session": lambda d: [e.update(session_key=1) for e in d["entries"] if e["endpoint"] == "pit"],
    }
    for label, mutate in mutations.items():
        data = json.loads(original)
        mutate(data)
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        assert reusable(config)[0] is False, label
        assert [u["session_key"] for u in plan(config)[0]] == [SESSION_KEY], label
    manifest_path.unlink()
    assert reusable(config)[0] is False
    assert [u["session_key"] for u in plan(config)[0]] == [SESSION_KEY]


@pytest.mark.parametrize(
    "error",
    [
        OpenF1Error("read timeout"),
        OpenF1HTTPError(status_code=403, message="forbidden", endpoint="pit"),
        OpenF1HTTPError(status_code=429, message="rate limited", endpoint="pit"),
        OpenF1HTTPError(status_code=500, message="server error", endpoint="pit"),
    ],
    ids=["timeout", "403", "429", "500"],
)
def test_non_404_failures_never_count_as_a_legitimate_absence(build_raw, raw_root, tmp_path, error):
    build_raw(car_data=None, pit=None)
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    assert run_stages(config, fake_client(error))[0] == cli.EXIT_FAILED
    assert reusable(config)[0] is False
    assert inc.record_session_completion(*context(config))[0] is False


def test_missing_raw_file_without_manifest_evidence_is_not_an_absence(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    run_stages(config, fake_client())
    assert reusable(config)[0] is True
    weather = next(p for p in raw_root.rglob("*.json") if "weather" in str(p) and str(SESSION_KEY) in p.name)
    weather.unlink()  # the manifest says "downloaded/skipped", not 404
    assert reusable(config)[0] is False


def test_altered_raw_input_reincludes_the_session(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, DEFAULT_ENDPOINTS)
    run_stages(config, fake_client())
    laps = next(p for p in raw_root.rglob("*.json") if "laps" in str(p) and str(SESSION_KEY) in p.name)
    laps.write_bytes(laps.read_bytes().replace(b"}", b"} ", 1))
    decision = reusable(config)
    assert decision[0] is False and "raw" in decision[1]
    assert [u["session_key"] for u in plan(config)[0]] == [SESSION_KEY]


def test_manual_m5_after_a_raw_change_cannot_complete_the_session_without_fresh_m3(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    run_stages(config, fake_client())
    laps = next(p for p in raw_root.rglob("*.json") if "laps" in str(p) and str(SESSION_KEY) in p.name)
    laps.write_bytes(laps.read_bytes().replace(b"}", b"} ", 1))

    assert run_stages(config, fake_client(), stages=("m5",)) == [0]
    assert reusable(config)[0] is False
    assert inc.record_session_completion(*context(config))[0] is False
    assert run_stages(config, fake_client(), stages=("m4", "m5")) == [0, 0]
    assert reusable(config)[0] is False  # M3 is still stale
    assert run_stages(config, fake_client(), stages=("m3", "m4", "m5")) == [0, 0, 0]
    assert reusable(config)[0] is True


def _mutate_json(path: Path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("target", ["m3_report", "m4_report_path", "m4_report_bytes", "m5_report", "stages", "features", "consolidated"])
def test_mutating_reports_hashes_paths_or_outputs_reincludes_the_session(build_raw, raw_root, tmp_path, target):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    run_stages(config, fake_client())
    assert reusable(config)[0] is True
    files = {
        "m3_report": next((tmp_path / "validation").rglob("*.json")),
        "m4_report_path": next((tmp_path / "consolidation").rglob("*.json")),
        "m4_report_bytes": next((tmp_path / "consolidation").rglob("*.json")),
        "m5_report": next((tmp_path / "feature_reports").rglob("*.json")),
        "stages": next((tmp_path / "manifests_m2" / "incremental").rglob("stages.json")),
        "features": next((tmp_path / "features").rglob("*.parquet")),
        "consolidated": next((tmp_path / "processed").rglob("*.parquet")),
    }
    path = files[target]
    if path.suffix == ".parquet":
        path.write_bytes(path.read_bytes() + b"\0")
    elif target == "m4_report_path":
        _mutate_json(path, lambda d: [s.update(output_path="elsewhere.parquet") for s in d["sessions"]])
    elif target == "stages":
        _mutate_json(path, lambda d: d["stages"]["m3"]["report"].update(sha256="0" * 64))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    assert reusable(config)[0] is False, target
    assert [u["session_key"] for u in plan(config)[0]] == [SESSION_KEY]


def test_two_independent_local_roots_do_not_share_state(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    run_stages(config, fake_client())
    assert reusable(config)[0] is True

    other = tmp_path / "other"
    other.mkdir()
    other_config = other / "config.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["historical_extraction"]["manifest_path"] = str(other / "manifests_m2")
    other_config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert reusable(other_config) == (False, "no verified completion provenance")


def test_a_different_configuration_does_not_reuse_the_decision(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, NO_CAR_DATA)
    run_stages(config, fake_client())
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["features"]["rolling_windows"] = [2]
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert reusable(config)[0] is False
