"""Exclusive session planning and the M3 gate (T1-T8 of the orchestration audit).

T1 planner scope           T5 PASS / WARNING continue
T2 CLI scope contract      T6 direct M4/M5 cannot bypass M3
T3 (state machine, see tests/infra/test_state_machine.py)
T4 (state machine, see tests/infra/test_state_machine.py)
T7 direct M6 cannot bypass M3
T8 multi-session and multi-season runs keep working
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.backends import S3StorageBackend
from f1_race_intelligence.storage.backends.local import LocalStorageBackend
from f1_race_intelligence.storage.layout import Layer, LakeLayout
from f1_race_intelligence.storage.raw_storage import RawDataStorage
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.consolidation.runner import ConsolidationRunner
from f1_race_intelligence.features.runner import FeatureRunner
from f1_race_intelligence.modeling.runner import ModelingRunner
from f1_race_intelligence.pipelines.incremental import _config_fingerprint
from f1_race_intelligence.validation.gate import (
    ValidationGateBlocked,
    check_validation_gate,
    enforce_validation_gate,
    modeling_gate_years,
)
from s3_fakes import FakeS3Client
from test_incremental_e2e import write_config

YEAR, MEETING, SESSION = 2025, 1276, 9839
OTHER_SESSION, OTHER_MEETING, OTHER_MEETING_SESSION = 9840, 1277, 9900


def catalog(store: DataStore, raw_root: Path, year=YEAR, meetings=None) -> None:
    raw = RawDataStorage(str(raw_root), store=store)
    meetings = meetings or {MEETING: [SESSION, OTHER_SESSION], OTHER_MEETING: [OTHER_MEETING_SESSION]}
    raw.save("meetings", {"year": year}, [{"meeting_key": key, "year": year} for key in meetings], file_name="meetings.json")
    for key, sessions in meetings.items():
        raw.save(
            "sessions",
            {"year": year, "meeting_key": key},
            [{"session_key": item, "session_name": "Race", "session_type": "Race"} for item in sessions],
            file_name="sessions.json",
        )


@pytest.fixture
def planned(tmp_path):
    raw_root = tmp_path / "raw"
    config = write_config(tmp_path, raw_root, ["laps"])
    settings = load_settings(str(config))
    settings.historical_extraction.years = [2024, YEAR]
    store = DataStore.from_settings(settings)
    catalog(store, raw_root)
    catalog(store, raw_root, year=2024, meetings={1200: [9000, 9001]})
    return config, settings, store


# -- T1 ------------------------------------------------------------------------------------


def test_t1_planner_selects_exactly_the_requested_session(planned) -> None:
    _, settings, store = planned
    client = MagicMock()
    with HistoricalExtractionPipeline(settings, client=client, store=store) as pipeline:
        units = pipeline.plan([YEAR], meeting_key=MEETING, session_key=SESSION)

        assert [(u["year"], u["meeting_key"], u["session_key"]) for u in units] == [(YEAR, MEETING, SESSION)]
        assert [d["session_key"] for d in pipeline.plan_decisions] == [SESSION]

        meeting_only = pipeline.plan([YEAR], meeting_key=MEETING)
        assert [u["session_key"] for u in meeting_only] == [SESSION, OTHER_SESSION]

        with pytest.raises(ValueError, match="exactly one year"):
            pipeline.plan([2024, YEAR], meeting_key=MEETING, session_key=SESSION)
        with pytest.raises(ValueError, match="needs meeting_key"):
            pipeline.plan([YEAR], session_key=SESSION)


# -- T2 ------------------------------------------------------------------------------------


def run_plan(config: Path, tmp_path: Path, *scope: str) -> tuple:
    output = tmp_path / "plan.json"
    code = cli.main(
        ["--stage", "plan", "--config", str(config), "--plan-output", str(output), *scope],
        client_factory=lambda settings: MagicMock(),
    )
    return code, (json.loads(output.read_text(encoding="utf-8")) if output.exists() else None)


def test_t2_cli_plan_scope_reaches_the_planner(planned, tmp_path) -> None:
    config, _, _ = planned
    code, plan = run_plan(config, tmp_path, "--year", str(YEAR), "--meeting-key", str(MEETING), "--session-key", str(SESSION))
    assert code == 0
    assert [(u["year"], u["meeting_key"], u["session_key"]) for u in plan["sessions"]] == [(YEAR, MEETING, SESSION)]


def test_t2_cli_rejects_partial_or_unmatched_plan_scope(planned, tmp_path) -> None:
    config, _, _ = planned
    assert run_plan(config, tmp_path, "--session-key", str(SESSION))[0] == 2
    assert run_plan(config, tmp_path, "--year", str(YEAR), "--session-key", str(SESSION))[0] == 2
    code, plan = run_plan(config, tmp_path, "--year", str(YEAR), "--meeting-key", str(MEETING), "--session-key", "1")
    assert code == 1 and plan is None  # nothing matched: fail, write no plan


# -- gate fixtures ---------------------------------------------------------------------------


@pytest.fixture
def lake(tmp_path):
    return DataStore(backend=LocalStorageBackend(str(tmp_path / "lake")), layout=LakeLayout())


def add_report(store: DataStore, stamp: str, status: str, scope: SessionScope = SessionScope()) -> None:
    root = store.root(Layer.VALIDATION_REPORTS, "validation")
    prefix = store.layout.records_prefix(root, year=scope.year, session_key=scope.session_key)
    store.backend.write_json(
        f"{prefix}/validation_report_{stamp}.json",
        {"global_status": status, "source": {"filter": None if scope.is_empty else scope.to_dict()}},
    )


SETTINGS = AppSettings(validation={"report_path": "validation"})
TARGET = SessionScope(year=YEAR, meeting_key=MEETING, session_key=SESSION)


# -- T5 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["PASS", "WARNING"])
def test_t5_pass_and_warning_let_the_session_continue(lake, status) -> None:
    add_report(lake, "20250101T000000Z", status, TARGET)
    assert check_validation_gate(lake, SETTINGS, TARGET) is None


# -- T6 ------------------------------------------------------------------------------------


def test_t6_fail_missing_or_foreign_verdicts_block_a_session(lake) -> None:
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, TARGET)

    add_report(lake, "20250101T000000Z", "PASS", SessionScope(year=YEAR, meeting_key=MEETING, session_key=OTHER_SESSION))
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, TARGET)  # another session's PASS

    add_report(lake, "20250102T000000Z", "FAIL", TARGET)
    assert "FAIL" in check_validation_gate(lake, SETTINGS, TARGET)

    add_report(lake, "20250103T000000Z", "PASS", TARGET)  # a newer verdict supersedes
    assert check_validation_gate(lake, SETTINGS, TARGET) is None


def test_t6_direct_m4_and_m5_write_nothing_without_a_passing_m3(planned, tmp_path) -> None:
    config, _, _ = planned
    scope = ["--year", str(YEAR), "--meeting-key", str(MEETING), "--session-key", str(SESSION), "--config", str(config)]
    for stage in ("m4", "m5"):
        assert cli.main(["--stage", stage, *scope]) == 1  # no M3 report at all
    assert not list((tmp_path / "processed").rglob("*")) and not list((tmp_path / "features").rglob("*"))
    assert not list((tmp_path / "consolidation").rglob("*")) and not list((tmp_path / "feature_reports").rglob("*"))


# -- T7 ------------------------------------------------------------------------------------


def test_t7_direct_m6_is_blocked_by_a_failed_validation(planned, tmp_path) -> None:
    config, settings, store = planned
    add_report_for_settings(store, settings, "20250101T000000Z", "FAIL", TARGET)
    assert cli.main(["--stage", "m6", "--config", str(config)]) == 1
    assert not (tmp_path / "modeling_reports").exists()


def add_report_for_settings(store, settings, stamp, status, scope):
    root = store.root(Layer.VALIDATION_REPORTS, settings.validation.report_path)
    prefix = store.layout.records_prefix(root, year=scope.year, session_key=scope.session_key)
    store.backend.write_json(
        f"{prefix}/validation_report_{stamp}.json", {"global_status": status, "source": {"filter": scope.to_dict()}}
    )


# -- T8 ------------------------------------------------------------------------------------


def test_t8_broad_runs_keep_working(planned, lake) -> None:
    _, settings, store = planned

    # Unscoped planning still covers every session of every configured season.
    with HistoricalExtractionPipeline(settings, client=MagicMock(), store=store) as pipeline:
        units = pipeline.plan()
    assert {u["session_key"] for u in units} == {9000, 9001, SESSION, OTHER_SESSION, OTHER_MEETING_SESSION}

    # No stored verdict: a season or whole-history run is not blocked (it never was).
    assert check_validation_gate(lake, SETTINGS, SessionScope(year=YEAR)) is None
    assert check_validation_gate(lake, SETTINGS, SessionScope()) is None

    # A FAIL blocks only the data it covers.
    add_report(lake, "20250101T000000Z", "FAIL", TARGET)
    assert check_validation_gate(lake, SETTINGS, SessionScope(year=YEAR)) is not None
    assert check_validation_gate(lake, SETTINGS, SessionScope(year=2024)) is None
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=[2024]) is None
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=[2024, YEAR]) is not None
    other = SessionScope(year=YEAR, meeting_key=MEETING, session_key=OTHER_SESSION)
    add_report(lake, "20250102T000000Z", "PASS", other)
    assert check_validation_gate(lake, SETTINGS, other) is None

# -- runners: direct invocation cannot bypass the gate (before any read or write) ----------------------------------


class Untouched:
    """A reader that fails the test if a runner touches it."""

    def __getattr__(self, name):
        raise AssertionError(f"a runner read {name!r} before the M3 gate decided")


def lake_settings() -> AppSettings:
    return AppSettings(validation={"report_path": "validation"}, modeling={"years": [YEAR]})


def test_runners_refuse_a_failed_m3_before_reading_or_writing_anything(lake, tmp_path) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", TARGET)
    settings = lake_settings()
    catalog, features_loader, modeling_loader = Untouched(), Untouched(), Untouched()

    with pytest.raises(ValidationGateBlocked, match="FAIL"):
        ConsolidationRunner(settings, catalog=catalog, store=lake, scope=TARGET).run()
    with pytest.raises(ValidationGateBlocked, match="FAIL"):
        FeatureRunner(settings, loader=features_loader, store=lake, scope=TARGET).run()
    modeling = ModelingRunner(settings, loader=modeling_loader, store=lake)
    with pytest.raises(ValidationGateBlocked, match="FAIL"):
        modeling.run()
    with pytest.raises(ValidationGateBlocked, match="FAIL"):
        modeling.reusable_report()  # the reuse path reads artifacts too

    written = {path.relative_to(tmp_path / "lake").parts[0] for path in (tmp_path / "lake").rglob("*") if path.is_file()}
    assert written == {"validated"}  # only the validation report the test itself wrote


def test_runners_refuse_a_session_with_no_m3_report(lake) -> None:
    catalog, loader = Untouched(), Untouched()
    with pytest.raises(ValidationGateBlocked, match="no M3 validation report"):
        ConsolidationRunner(lake_settings(), catalog=catalog, store=lake, scope=TARGET).run()
    with pytest.raises(ValidationGateBlocked, match="no M3 validation report"):
        FeatureRunner(lake_settings(), loader=loader, store=lake, scope=TARGET).run()


def test_the_gate_lets_pass_and_warning_through(lake) -> None:
    for stamp, status in (("20250101T000000Z", "PASS"), ("20250102T000000Z", "WARNING")):
        add_report(lake, stamp, status, TARGET)
        assert check_validation_gate(lake, lake_settings(), TARGET) is None
        enforce_validation_gate(lake, lake_settings(), TARGET)  # does not raise


# -- unreadable or invalid reports are not "absent" -------------------------------------------------------


def corrupt_report(tmp_path: Path, store: DataStore, stamp: str, content: bytes, scope: SessionScope = TARGET) -> None:
    root = store.root(Layer.VALIDATION_REPORTS, "validation")
    path = tmp_path / "lake" / store.layout.records_prefix(root, year=scope.year, session_key=scope.session_key)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"validation_report_{stamp}.json").write_bytes(content)


BAD_CONTENT = {
    "truncated json": b'{"global_status": "PA',
    "not utf-8": b"\xff\xfe\x00bad",
    "json but not an object": b"[1, 2, 3]",
    "unknown verdict": b'{"global_status": "MAYBE", "source": {}}',
    "missing verdict": b'{"source": {"filter": null}}',
    "filter not an object": b'{"global_status": "PASS", "source": {"filter": "x"}}',
    "source not an object": b'{"global_status": "PASS", "source": 7}',
}


@pytest.mark.parametrize("content", BAD_CONTENT.values(), ids=BAD_CONTENT.keys())
def test_a_newer_unusable_report_blocks_and_never_falls_back_to_an_older_pass(lake, tmp_path, content) -> None:
    add_report(lake, "20250101T000000Z", "PASS", TARGET)
    assert check_validation_gate(lake, SETTINGS, TARGET) is None
    corrupt_report(tmp_path, lake, "20250102T000000Z", content)

    reason = check_validation_gate(lake, SETTINGS, TARGET)

    assert reason is not None and "unreadable or invalid" in reason and "20250102T000000Z" in reason
    assert "no M3 validation report" not in reason  # present-but-broken is not the same as absent


def test_an_unusable_report_older_than_a_valid_verdict_is_superseded(lake, tmp_path) -> None:
    corrupt_report(tmp_path, lake, "20250101T000000Z", b"{broken")
    assert "unreadable or invalid" in check_validation_gate(lake, SETTINGS, TARGET)
    add_report(lake, "20250102T000000Z", "WARNING", TARGET)
    assert check_validation_gate(lake, SETTINGS, TARGET) is None


def test_an_unusable_report_blocks_wider_runs_but_not_other_sessions_or_seasons(lake, tmp_path) -> None:
    other = SessionScope(year=YEAR, meeting_key=MEETING, session_key=OTHER_SESSION)
    add_report(lake, "20250101T000000Z", "PASS", other)
    corrupt_report(tmp_path, lake, "20250102T000000Z", b"{broken", TARGET)

    assert "unreadable or invalid" in check_validation_gate(lake, SETTINGS, SessionScope(year=YEAR))
    assert "unreadable or invalid" in check_validation_gate(lake, SETTINGS, SessionScope(), years=[YEAR])
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=[2024]) is None  # other season
    assert check_validation_gate(lake, SETTINGS, other) is None  # other session's partition


def test_the_acceptable_verdicts_are_exactly_pass_and_warning(lake) -> None:
    for index, (status, accepted) in enumerate([("PASS", True), ("WARNING", True), ("FAIL", False)]):
        add_report(lake, f"2025010{index + 1}T000000Z", status, TARGET)
        assert (check_validation_gate(lake, SETTINGS, TARGET) is None) is accepted, status


# -- M6: the gate follows modeling.years only; deferral stays distinct from M3 FAIL -----------------------------


def test_the_m6_gate_years_are_exactly_modeling_years() -> None:
    both = AppSettings(historical_extraction={"years": [2024]}, modeling={"years": [YEAR]})
    assert modeling_gate_years(both) == [YEAR]
    assert modeling_gate_years(AppSettings(historical_extraction={"years": [2024]})) is None  # not 2024
    assert modeling_gate_years(AppSettings()) is None


def test_m6_ignores_a_fail_outside_modeling_years_and_not_one_inside(lake) -> None:
    settings = AppSettings(historical_extraction={"years": [2024]}, modeling={"years": [YEAR]})
    add_report(lake, "20250101T000000Z", "FAIL", SessionScope(year=2024, meeting_key=1200, session_key=9000))
    assert check_validation_gate(lake, settings, SessionScope(), years=modeling_gate_years(settings)) is None
    add_report(lake, "20250102T000000Z", "FAIL", TARGET)
    assert check_validation_gate(lake, settings, SessionScope(), years=modeling_gate_years(settings)) is not None

    unrestricted = AppSettings(historical_extraction={"years": [2024]})  # modeling.years empty: every season
    assert check_validation_gate(lake, unrestricted, SessionScope(), years=modeling_gate_years(unrestricted)) is not None


def with_years(config: Path, *, historical, modeling) -> None:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["historical_extraction"]["years"] = historical
    data["modeling"] = {**data.get("modeling", {}), "years": modeling}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_m6_cli_separates_a_failed_m3_from_an_incomplete_history(planned, capsys) -> None:
    config, _, store = planned
    with_years(config, historical=[YEAR], modeling=[YEAR])
    flags = ["--stage", "m6", "--config", str(config), "--require-complete-history"]

    # Incomplete history (catalogue present, sessions without provenance): deferred, not failed.
    assert cli.main(flags) == cli.EXIT_DEFERRED == 3
    assert "DEFERRED" in capsys.readouterr().err

    # A passing M3 changes nothing: still deferred.
    add_report_for_settings(store, load_settings(str(config)), "20250101T000000Z", "WARNING", TARGET)
    assert cli.main(flags) == cli.EXIT_DEFERRED

    # A failed M3 is a failure (1) that names the gate, whatever the history says.
    add_report_for_settings(store, load_settings(str(config)), "20250102T000000Z", "FAIL", TARGET)
    assert cli.main(flags) == cli.EXIT_FAILED == 1
    err = capsys.readouterr().err
    assert "M3 gate blocks m6" in err and "DEFERRED" not in err.split("M3 gate blocks m6")[-1]


def test_m6_cli_only_looks_at_modeling_years(planned) -> None:
    config, _, store = planned
    with_years(config, historical=[YEAR], modeling=[2024])
    add_report_for_settings(store, load_settings(str(config)), "20250101T000000Z", "FAIL", TARGET)  # a 2025 FAIL
    # modeling.years=[2024]: the 2025 FAIL is outside it, so the gate does not stop M6
    # (it then defers: no season is complete).
    assert cli.main(["--stage", "m6", "--config", str(config), "--require-complete-history"]) == cli.EXIT_DEFERRED


# -- the effective AWS configuration fingerprint ----------------------------------------------------------------

AWS_EFFECTIVE_FINGERPRINT = "b5265ebd02e74d407f7904dbda6d6b56d87961bfcec688e48d885db4d986d9de"


def test_the_effective_aws_configuration_keeps_its_fingerprint(monkeypatch) -> None:
    """What the pipeline task loads: configs/config.yaml (F1_CONFIG_PATH) with the stack's F1_* variables."""
    for name, value in {
        "F1_ENVIRONMENT": "aws",
        "F1_COMPONENT": "pipeline",
        "F1_STORAGE_BUCKET": "example-bucket",
        "F1_STORAGE_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("F1_EXTRACTION_YEARS", raising=False)
    settings = load_settings(str(Path(__file__).parents[2] / "configs" / "config.yaml"))

    assert (settings.environment, settings.storage.backend, settings.storage.layout) == ("aws", "s3", "lake")
    assert _config_fingerprint(settings, None) == AWS_EFFECTIVE_FINGERPRINT


# -- fail-closed filters: types and structure (finding 1) ------------------------------------------------------


def _report(status="PASS", **source):
    return json.dumps({"global_status": status, **source}).encode("utf-8")


def _filter(**values):
    return {"filter": {"year": None, "meeting_key": None, "session_key": None, **values}}


BAD_FILTERS = {
    "year as string": _report(source=_filter(year="2025")),
    "meeting_key as string": _report(source=_filter(year=YEAR, meeting_key="1276")),
    "session_key as string": _report(source=_filter(year=YEAR, session_key="9839")),
    "year as float": _report(source=_filter(year=2025.0)),
    "year as bool": _report(source=_filter(year=True)),
    "year as list": _report(source=_filter(year=[2025])),
    "filter without year": _report(source={"filter": {"meeting_key": None, "session_key": None}}),
    "filter without session_key": _report(source={"filter": {"year": YEAR, "meeting_key": None}}),
    "empty filter object": _report(source={"filter": {}}),
    "source without filter": _report(source={"raw_path": "x"}),
    "no source at all": _report(),
}


@pytest.mark.parametrize("content", BAD_FILTERS.values(), ids=BAD_FILTERS.keys())
def test_a_report_with_an_invalid_filter_blocks_m6_even_with_restricted_modeling_years(lake, tmp_path, content) -> None:
    # An older full-run PASS must not be what M6 falls back on.
    add_report(lake, "20250101T000000Z", "PASS")
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=[YEAR]) is None
    corrupt_report(tmp_path, lake, "20250102T000000Z", content, SessionScope())

    reason = check_validation_gate(lake, SETTINGS, SessionScope(), years=[YEAR])

    assert reason is not None and "unreadable or invalid" in reason


@pytest.mark.parametrize("content", BAD_FILTERS.values(), ids=BAD_FILTERS.keys())
def test_a_report_with_an_invalid_filter_blocks_a_session(lake, tmp_path, content) -> None:
    add_report(lake, "20250101T000000Z", "PASS", TARGET)
    corrupt_report(tmp_path, lake, "20250102T000000Z", content, TARGET)
    assert "unreadable or invalid" in check_validation_gate(lake, SETTINGS, TARGET)


def test_a_string_year_in_a_season_report_is_not_silently_skipped_for_m6(lake, tmp_path) -> None:
    corrupt_report(tmp_path, lake, "20250101T000000Z", _report("FAIL", source=_filter(year="2025")), SessionScope())
    assert "unreadable or invalid" in check_validation_gate(lake, SETTINGS, SessionScope(), years=[YEAR])


def test_well_formed_filters_are_still_accepted(lake) -> None:
    add_report(lake, "20250101T000000Z", "PASS")  # full run: filter null
    add_report(lake, "20250102T000000Z", "PASS", SessionScope(year=YEAR))
    add_report(lake, "20250103T000000Z", "WARNING", TARGET)
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=[YEAR]) is None
    assert check_validation_gate(lake, SETTINGS, TARGET) is None


# -- what covers a session: full, season, meeting and session reports (finding 4) --------------------------------


def test_season_and_meeting_reports_vouch_for_their_sessions(lake) -> None:
    add_report(lake, "20250101T000000Z", "PASS", SessionScope(year=YEAR))
    assert check_validation_gate(lake, SETTINGS, TARGET) is None  # a season report covers the session

    other_meeting = SessionScope(year=YEAR, meeting_key=OTHER_MEETING, session_key=OTHER_MEETING_SESSION)
    assert check_validation_gate(lake, SETTINGS, other_meeting) is None


def test_a_meeting_report_covers_only_its_own_meeting(lake) -> None:
    add_report(lake, "20250101T000000Z", "WARNING", SessionScope(year=YEAR, meeting_key=MEETING))
    assert check_validation_gate(lake, SETTINGS, TARGET) is None
    other_meeting = SessionScope(year=YEAR, meeting_key=OTHER_MEETING, session_key=OTHER_MEETING_SESSION)
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, other_meeting)


def test_a_report_of_another_season_does_not_cover_the_session(lake) -> None:
    add_report(lake, "20250101T000000Z", "PASS", SessionScope(year=2024))
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, TARGET)


def test_a_failed_season_report_blocks_its_sessions_until_a_newer_verdict(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", SessionScope(year=YEAR))
    assert "FAIL" in check_validation_gate(lake, SETTINGS, TARGET)
    add_report(lake, "20250102T000000Z", "PASS", TARGET)  # the session was re-validated
    assert check_validation_gate(lake, SETTINGS, TARGET) is None
    add_report(lake, "20250103T000000Z", "FAIL", SessionScope(year=YEAR))  # a newer failed season run
    assert "FAIL" in check_validation_gate(lake, SETTINGS, TARGET)


def test_a_report_that_names_what_the_scope_leaves_unknown_does_not_vouch(lake) -> None:
    add_report(lake, "20250101T000000Z", "PASS", SessionScope(year=YEAR, meeting_key=MEETING))
    unknown_meeting = SessionScope(year=YEAR, session_key=SESSION)
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, unknown_meeting)


def test_a_session_report_of_another_session_never_covers(lake) -> None:
    add_report(lake, "20250101T000000Z", "PASS", SessionScope(year=YEAR, meeting_key=MEETING, session_key=OTHER_SESSION))
    assert "no M3 validation report" in check_validation_gate(lake, SETTINGS, TARGET)


# -- a session gate lists only the prefixes that can matter (finding 3) --------------------------------------------


@pytest.fixture
def s3_lake():
    client = FakeS3Client(page_size=1000)
    store = DataStore(backend=S3StorageBackend("bucket", "dev", client=client), layout=LakeLayout())
    listings, reads = [], []
    original_list, original_get = client.list_objects_v2, client.get_object
    client.list_objects_v2 = lambda **kwargs: (listings.append((kwargs["Prefix"], kwargs.get("Delimiter"))), original_list(**kwargs))[1]
    client.get_object = lambda **kwargs: (reads.append(kwargs["Key"]), original_get(**kwargs))[1]
    return store, listings, reads


def test_a_session_gate_does_not_list_or_read_other_sessions_reports(s3_lake) -> None:
    store, listings, reads = s3_lake
    for session in range(1, 41):  # a season's worth of other sessions
        add_report(store, f"2025010{session % 9 + 1}T0000{session:02d}Z", "PASS", SessionScope(year=YEAR, meeting_key=MEETING, session_key=session))
    add_report(store, "20250201T000000Z", "PASS", TARGET)
    add_report(store, "20250101T000000Z", "FAIL", SessionScope(year=YEAR))  # a season report at the root
    add_report(store, "20250101T000001Z", "FAIL")  # a full-run report at the root
    listings.clear()
    reads.clear()

    assert check_validation_gate(store, SETTINGS, TARGET) is None  # the newest covering report is the session's PASS

    assert all(f"session={session}/" not in key for key in reads for session in range(1, 41) if session != SESSION), reads
    assert len(reads) == 3  # the root's two reports and the session's own
    assert ("dev/validated/", "/") in listings  # the root, not recursive
    assert all(delimiter == "/" or f"session={SESSION}" in prefix for prefix, delimiter in listings), listings


def test_a_session_gate_still_honours_root_reports_on_s3(s3_lake) -> None:
    store, _, _ = s3_lake
    add_report(store, "20250101T000000Z", "FAIL")  # full run, at the root
    assert "FAIL" in check_validation_gate(store, SETTINGS, TARGET)
    add_report(store, "20250102T000000Z", "PASS", SessionScope(year=YEAR))  # newer season report, also at the root
    assert check_validation_gate(store, SETTINGS, TARGET) is None


def test_wider_scopes_still_see_every_session_report(s3_lake) -> None:
    store, _, _ = s3_lake
    add_report(store, "20250101T000000Z", "FAIL", TARGET)
    assert "FAIL" in check_validation_gate(store, SETTINGS, SessionScope(year=YEAR))
    assert "FAIL" in check_validation_gate(store, SETTINGS, SessionScope(), years=[YEAR])


# -- wide runs group reports by the exact (year, meeting_key, session_key) triple --------------------------------

M6_YEARS = [YEAR]
MEETING_1276 = SessionScope(year=YEAR, meeting_key=MEETING)  # meeting reports carry no session_key
MEETING_1277 = SessionScope(year=YEAR, meeting_key=OTHER_MEETING)


def test_a_pass_for_another_meeting_never_replaces_a_fail_for_m6(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", MEETING_1276)
    add_report(lake, "20250102T000000Z", "PASS", MEETING_1277)  # newer, but another meeting
    reason = check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS)
    assert reason is not None and "FAIL" in reason
    # the same holds for a whole-season run and for the sessions of the failed meeting
    assert "FAIL" in check_validation_gate(lake, SETTINGS, SessionScope(year=YEAR))
    assert "FAIL" in check_validation_gate(lake, SETTINGS, TARGET)


def test_a_later_pass_of_the_same_meeting_lets_m6_continue(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", MEETING_1276)
    add_report(lake, "20250102T000000Z", "PASS", MEETING_1276)
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS) is None


def test_a_pass_for_a_sibling_session_never_replaces_a_fail_for_m6(lake) -> None:
    sibling = SessionScope(year=YEAR, meeting_key=MEETING, session_key=OTHER_SESSION)
    add_report(lake, "20250101T000000Z", "FAIL", TARGET)
    add_report(lake, "20250102T000000Z", "PASS", sibling)
    assert "FAIL" in check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS)


def test_a_later_pass_of_the_same_session_lets_m6_continue(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", TARGET)
    add_report(lake, "20250102T000000Z", "PASS", TARGET)
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS) is None


def test_a_season_pass_does_not_replace_a_meeting_fail_for_m6(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", MEETING_1276)
    add_report(lake, "20250102T000000Z", "PASS", SessionScope(year=YEAR))
    assert "FAIL" in check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS)


def test_a_meeting_pass_does_not_replace_a_season_fail(lake) -> None:
    add_report(lake, "20250101T000000Z", "FAIL", SessionScope(year=YEAR))
    add_report(lake, "20250102T000000Z", "PASS", MEETING_1276)
    assert "FAIL" in check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS)


@pytest.mark.parametrize("newer_pass", [SessionScope(year=YEAR), SessionScope()], ids=["season", "full-run"])
def test_an_unusable_meeting_report_blocks_even_with_a_newer_valid_pass(lake, tmp_path, newer_pass) -> None:
    # Meeting/season/full reports share the year-less partition, so the corrupt file cannot be
    # attributed to a scope; a full-run PASS used to share its group and hid it.
    corrupt_report(tmp_path, lake, "20250101T000000Z", b"{not json", MEETING_1276)
    add_report(lake, "20250102T000000Z", "PASS", newer_pass)
    reason = check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS)
    assert reason is not None and "unreadable or invalid" in reason


def test_an_unusable_session_report_is_settled_by_a_newer_pass_of_the_same_session(lake, tmp_path) -> None:
    corrupt_report(tmp_path, lake, "20250101T000000Z", b"{not json", TARGET)
    add_report(lake, "20250102T000000Z", "PASS", TARGET)
    assert check_validation_gate(lake, SETTINGS, SessionScope(), years=M6_YEARS) is None
