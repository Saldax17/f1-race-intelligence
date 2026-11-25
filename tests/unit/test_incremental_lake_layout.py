"""Incremental provenance over the REAL stages M2->M5 on the data lake layout.

The lake layout files every per-session record under
``manifests/<stage>/year=<y>/session=<s>/`` (``layout.records_prefix``). M2 and M3 always
did; M4 and M5 wrote their reports at the root of ``manifests/<stage>/`` instead, so the
provenance resolver (which lists the session partition) could not find them: M4 logged
``incremental_stage_provenance_not_recorded``, ``stages.json`` kept only M3 and
``state.json`` was never written, so the planner re-included a fully processed session.

No stage report is fabricated here except where a test says so: the stages write their
own reports through the CLI, the same entry point the Fargate tasks use.
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import load_settings
from f1_race_intelligence.consolidation.runner import ConsolidationRunner
from f1_race_intelligence.features.runner import FeatureRunner
from f1_race_intelligence.pipelines import incremental as inc
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends import LocalStorageBackend
from f1_race_intelligence.storage.layout import LakeLayout, Layer
from f1_race_intelligence.storage.migration import copy_raw
from f1_race_intelligence.storage.store import DataStore
from factories import MEETING_KEY, SESSION_KEY, YEAR, car_data_record, driver_record
from test_consolidation import DRIVERS, build_raw  # noqa: F401 - fixture
from test_historical_extraction import make_client
from test_incremental_e2e import (  # noqa: F401 - clean_environment is an autouse fixture
    DEFAULT_ENDPOINTS,
    clean_environment,
    fake_client,
    run_commands,
    run_stages,
    state_machine_session_commands,
    write_config,
)

OTHER_SESSION = SESSION_KEY + 1
UNIT = {"year": YEAR, "meeting_key": MEETING_KEY, "session_key": SESSION_KEY}

M4_REPORT = "consolidation_report_"
M5_REPORT = "feature_engineering_report_"


# -- harness ------------------------------------------------------------------


def write_lake_config(tmp_path: Path) -> Path:
    config = {
        "storage": {"backend": "local", "layout": "lake", "local_root": str(tmp_path / "lake")},
        "historical_extraction": {"years": [YEAR], "session_types": ["Race"], "endpoints": DEFAULT_ENDPOINTS},
    }
    path = tmp_path / "lake-config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def lake_with_raw(build_raw, raw_root: Path, tmp_path: Path, **sources) -> Path:
    """Raw data for the test session, already inside the lake, plus the config that points at it."""
    build_raw(**{"car_data": None, "pit": [], **sources})
    lake = DataStore(backend=LocalStorageBackend(str(tmp_path / "lake")), layout=LakeLayout())
    copy_raw(DataStore.local(), lake, raw_path=str(raw_root), include_manifests=False)
    return write_lake_config(tmp_path)


def context(config: Path, session_key: int = SESSION_KEY):
    settings = load_settings(str(config))
    scope = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=session_key)
    return DataStore.from_settings(settings), settings, scope


def plan(config: Path):
    settings = load_settings(str(config))
    pipeline = HistoricalExtractionPipeline(settings, client=make_client(), store=DataStore.from_settings(settings))
    return pipeline.plan(), pipeline.plan_decisions


def run_chain(config: Path):
    """The per-session commands the state machine would send, through the real CLI."""
    commands = state_machine_session_commands(UNIT)
    return run_commands(config, commands, fake_client())


def run_later_stages(config: Path, *stages: str):
    commands = state_machine_session_commands(UNIT)
    return run_commands(config, {stage: commands[stage] for stage in stages}, fake_client())


def partition(tmp_path: Path, stage_dir: str, session_key: int = SESSION_KEY) -> Path:
    return tmp_path / "lake" / "manifests" / stage_dir / f"year={YEAR}" / f"session={session_key}"


def root_reports(tmp_path: Path, stage_dir: str, prefix: str):
    return sorted((tmp_path / "lake" / "manifests" / stage_dir).glob(f"{prefix}*.json"))


def incremental(tmp_path: Path, name: str):
    return sorted((tmp_path / "lake" / "manifests" / "incremental").rglob(name))


def stage_names(tmp_path: Path):
    files = incremental(tmp_path, "stages.json")
    return set(json.loads(files[0].read_text(encoding="utf-8"))["stages"]) if files else set()


def m3_report(config: Path) -> dict:
    store, settings, scope = context(config)
    _, report = inc._latest_session_report(
        store, store.root(Layer.VALIDATION_REPORTS, settings.validation.report_path), scope, "data_validation"
    )
    return report


# -- where the reports go ---------------------------------------------------------


def test_m4_and_m5_reports_are_filed_under_their_session_partition(build_raw, raw_root, tmp_path):
    config = lake_with_raw(build_raw, raw_root, tmp_path)

    assert run_chain(config) == [0, 0, 0, 0]

    for stage_dir, prefix in (("consolidation", M4_REPORT), ("features", M5_REPORT)):
        assert len(list(partition(tmp_path, stage_dir).glob(f"{prefix}*.json"))) == 1, stage_dir
        assert root_reports(tmp_path, stage_dir, prefix) == [], f"{stage_dir} report leaked to the stage root"
    # Same convention as M2 and M3.
    assert list(partition(tmp_path, "extraction").glob("manifest_*.json"))
    assert list((tmp_path / "lake" / "validated" / f"year={YEAR}" / f"session={SESSION_KEY}").glob("validation_report_*.json"))


def test_the_resolver_finds_the_reports_the_stages_wrote(build_raw, raw_root, tmp_path):
    config = lake_with_raw(build_raw, raw_root, tmp_path)
    assert run_chain(config) == [0, 0, 0, 0]

    store, settings, scope = context(config)
    for stage in ("m3", "m4", "m5"):
        root, pipeline_name = inc._stage_report_location(settings, store, stage)
        key, payload = inc._latest_session_report(store, root, scope, pipeline_name)
        assert f"year={YEAR}/session={SESSION_KEY}/" in key
        assert payload["pipeline_name"] == pipeline_name


# -- provenance and completion ---------------------------------------------------------


def test_real_chain_on_the_lake_records_provenance_and_completion(build_raw, raw_root, tmp_path):
    config = lake_with_raw(build_raw, raw_root, tmp_path)
    units, _ = plan(config)
    assert [u["session_key"] for u in units] == [SESSION_KEY]  # pending before the run

    assert run_chain(config) == [0, 0, 0, 0]

    assert stage_names(tmp_path) == {"m3", "m4", "m5"}
    assert len(incremental(tmp_path, "state.json")) == 1
    assert inc.session_reuse_decision(*context(config)) == (True, "all M2-M5 artifacts and provenance match")
    units, decisions = plan(config)
    assert units == []
    assert [d["decision"] for d in decisions] == ["omit"]


def test_m3_warning_allows_provenance_and_completion(build_raw, raw_root, tmp_path):
    """The Abu Dhabi case: every driver's country_code is null, the only finding is that WARNING."""
    drivers = [{**driver_record(d, f"D{d}", f"Team {d}"), "country_code": None} for d in DRIVERS]
    config = lake_with_raw(build_raw, raw_root, tmp_path, drivers=drivers)

    assert run_chain(config) == [0, 0, 0, 0]

    report = m3_report(config)
    assert report["global_status"] == "WARNING"
    assert report["summary"]["errors"] == 0
    warnings = [r for r in report["results"] if r["severity"] == "WARNING"]
    # The synthetic session already carries a few WARNINGs of its own; the null country_code is one more.
    assert ("missing_values", "drivers", "country_code") in [
        (w["rule"], w["endpoint"], (w.get("details") or {}).get("field")) for w in warnings
    ]
    assert stage_names(tmp_path) == {"m3", "m4", "m5"}
    assert len(incremental(tmp_path, "state.json")) == 1
    assert inc.session_reuse_decision(*context(config))[0] is True


def test_m3_fail_never_registers_the_session_as_completed(build_raw, raw_root, tmp_path):
    bad_gear = {d: [car_data_record(d, 10.0, n_gear=20), car_data_record(d, 20.0)] for d in DRIVERS}
    config = lake_with_raw(build_raw, raw_root, tmp_path, car_data=bad_gear)

    # M3 runs without --strict, so FAIL is recorded and the task exits 0 (unchanged behaviour).
    assert run_chain(config) == [0, 0, 0, 0]

    assert m3_report(config)["global_status"] == "FAIL"
    assert incremental(tmp_path, "state.json") == []
    assert stage_names(tmp_path) == set()  # no stage provenance: M4/M5 cannot build on a failed M3
    assert inc.session_reuse_decision(*context(config))[0] is False
    units, decisions = plan(config)
    assert [u["session_key"] for u in units] == [SESSION_KEY]
    assert [d["decision"] for d in decisions] == ["include"]


# -- isolation between sessions -------------------------------------------------------------


def test_the_resolver_never_attributes_another_sessions_report(tmp_path):
    store = DataStore(backend=LocalStorageBackend(str(tmp_path)), layout=LakeLayout())
    root = "manifests/consolidation"

    def report(*sessions, started):
        return {
            "pipeline_name": "data_consolidation",
            "started_at": started,
            "sessions": [{"year": YEAR, "session_key": s, "output_path": f"out-{s}"} for s in sessions],
        }

    for session in (SESSION_KEY, OTHER_SESSION):
        prefix = store.layout.records_prefix(root, year=YEAR, session_key=session)
        store.backend.write_json(f"{prefix}/consolidation_report_{session}.json", report(session, started="t"))
    # A report left at the stage root (what the old writer produced), naming both sessions.
    store.backend.write_json(f"{root}/consolidation_report_stale.json", report(SESSION_KEY, OTHER_SESSION, started="old"))

    own = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=SESSION_KEY)
    other = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=OTHER_SESSION)
    unprocessed = SessionScope(year=YEAR, meeting_key=MEETING_KEY, session_key=OTHER_SESSION + 1)

    key, payload = inc._latest_session_report(store, root, own, "data_consolidation")
    assert key.endswith(f"year={YEAR}/session={SESSION_KEY}/consolidation_report_{SESSION_KEY}.json")
    assert [s["session_key"] for s in payload["sessions"]] == [SESSION_KEY]
    key, _ = inc._latest_session_report(store, root, other, "data_consolidation")
    assert key.endswith(f"year={YEAR}/session={OTHER_SESSION}/consolidation_report_{OTHER_SESSION}.json")
    # No partition of its own: neither the other sessions' reports nor the stage-root leftover count.
    with pytest.raises(inc.SessionStateError):
        inc._latest_session_report(store, root, unprocessed, "data_consolidation")


def test_two_sessions_on_one_lake_keep_separate_reports(build_raw, raw_root, tmp_path):
    config = lake_with_raw(build_raw, raw_root, tmp_path)
    assert run_chain(config) == [0, 0, 0, 0]
    session_b_scope = ["--year", str(YEAR), "--meeting-key", str(MEETING_KEY), "--session-key", str(OTHER_SESSION)]

    # M4 for a session that has no raw data fails; it must not touch or borrow session A's reports.
    code = cli.main(["--stage", "m4", *session_b_scope, "--config", str(config), "--overwrite-derived"])

    assert code != 0
    assert len(list(partition(tmp_path, "consolidation").glob(f"{M4_REPORT}*.json"))) == 1
    assert inc.session_reuse_decision(*context(config))[0] is True
    assert inc.session_reuse_decision(*context(config, OTHER_SESSION))[0] is False


# -- compatibility ------------------------------------------------------------------------


def test_broader_than_session_runs_keep_their_reports_at_the_stage_root(build_raw, raw_root, tmp_path):
    config = lake_with_raw(build_raw, raw_root, tmp_path)
    assert run_chain(config) == [0, 0, 0, 0]
    settings = load_settings(str(config))
    store = DataStore.from_settings(settings)
    year_scope = SessionScope(year=YEAR)  # a whole-season run has no single session to file the report under

    ConsolidationRunner(settings, store=store, scope=year_scope).run()
    FeatureRunner(settings, store=store, scope=year_scope).run()

    assert len(root_reports(tmp_path, "consolidation", M4_REPORT)) == 1
    assert len(root_reports(tmp_path, "features", M5_REPORT)) == 1
    # ...and they are not mistaken for the session's own provenance reports.
    assert len(list(partition(tmp_path, "consolidation").glob(f"{M4_REPORT}*.json"))) == 1
    assert inc.session_reuse_decision(*context(config))[0] is True


def test_the_legacy_layout_still_writes_reports_at_the_configured_root(build_raw, raw_root, tmp_path):
    build_raw(car_data=None, pit=[])
    config = write_config(tmp_path, raw_root, DEFAULT_ENDPOINTS)

    assert run_stages(config, fake_client()) == [0, 0, 0, 0]

    assert len(list((tmp_path / "consolidation").glob(f"{M4_REPORT}*.json"))) == 1
    assert len(list((tmp_path / "feature_reports").glob(f"{M5_REPORT}*.json"))) == 1
    assert not [p for p in (tmp_path / "consolidation").iterdir() if p.is_dir()]
    assert not [p for p in (tmp_path / "feature_reports").iterdir() if p.is_dir()]


# -- recovering a session processed before the fix ---------------------------------------------------


def M2_M3_FILES(lake: Path):
    """Everything M2 and M3 produce: raw data, extraction manifests and validation reports."""
    return [*lake.joinpath("raw").rglob("*.json"), *lake.joinpath("manifests", "extraction").rglob("*.json"),
            *lake.joinpath("validated").rglob("*.json")]


def snapshot(paths):
    return {str(p): Path(p).read_bytes() for p in paths}


def test_rerunning_only_m4_and_m5_recovers_a_session_without_repeating_m2_or_m3(build_raw, raw_root, tmp_path):
    """The Abu Dhabi 2025 state after the first AWS run: raw, M2 and M3 fine, M4/M5 reports at the root."""
    config = lake_with_raw(build_raw, raw_root, tmp_path)
    assert run_chain(config) == [0, 0, 0, 0]

    # Rebuild the pre-fix state: M4/M5 reports at the stage root, no M4/M5 provenance, no completion.
    for stage_dir, prefix in (("consolidation", M4_REPORT), ("features", M5_REPORT)):
        report = next(partition(tmp_path, stage_dir).glob(f"{prefix}*.json"))
        shutil.move(str(report), str(report.parents[2] / report.name))
        shutil.rmtree(partition(tmp_path, stage_dir))
    (stages_file,) = incremental(tmp_path, "stages.json")
    stages_doc = json.loads(stages_file.read_text(encoding="utf-8"))
    stages_doc["stages"] = {"m3": stages_doc["stages"]["m3"]}
    stages_file.write_text(json.dumps(stages_doc), encoding="utf-8")
    for state in incremental(tmp_path, "state.json"):
        state.unlink()
    assert inc.session_reuse_decision(*context(config))[0] is False  # the planner would redo everything

    lake = tmp_path / "lake"
    untouched = snapshot(M2_M3_FILES(lake))
    stale = snapshot(root_reports(tmp_path, "consolidation", M4_REPORT) + root_reports(tmp_path, "features", M5_REPORT))

    assert run_later_stages(config, "m4", "m5") == [0, 0]

    assert stage_names(tmp_path) == {"m3", "m4", "m5"}
    assert len(incremental(tmp_path, "state.json")) == 1
    assert inc.session_reuse_decision(*context(config)) == (True, "all M2-M5 artifacts and provenance match")
    # M2 inputs and the M3 report are byte-for-byte what they were: nothing was extracted or validated again.
    assert snapshot(untouched) == untouched
    assert set(map(str, M2_M3_FILES(lake))) == set(untouched)  # and no new raw file, manifest or report appeared
    # The misplaced reports are left alone (no deletion), and the new ones sit in the partition.
    assert snapshot(stale) == stale
    assert len(list(partition(tmp_path, "consolidation").glob(f"{M4_REPORT}*.json"))) == 1
    assert len(list(partition(tmp_path, "features").glob(f"{M5_REPORT}*.json"))) == 1
    # Last, because planning writes its own (entry-less) manifest: the session is now omitted.
    assert plan(config)[0] == []
