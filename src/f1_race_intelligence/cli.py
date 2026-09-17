"""Batch entrypoint: run any stage, over everything or over one session.

::

    python -m f1_race_intelligence --stage m2 --year 2024 --session-key 9472
    python -m f1_race_intelligence --stage m3,m4,m5 --year 2024 --session-key 9472
    python -m f1_race_intelligence --stage m6
    python -m f1_race_intelligence --stage plan --year 2024
    python -m f1_race_intelligence --stage m4 --session-key 9472 \\
        --storage-backend s3 --bucket my-bucket --prefix dev

Stages: ``plan`` (list the session work units of a season), ``m2`` extraction,
``m3`` validation, ``m4`` consolidation, ``m5`` features, ``m6`` modelling
datasets, and ``copy-raw`` (copy already-extracted raw data between stores).
Several stages run in order when given as a comma-separated list; the first
failure stops the chain.

Storage comes from configuration (``storage`` section, ``F1_STORAGE_*``
variables) and can be overridden per run. ``--input-storage`` and
``--output-storage`` take ``s3://bucket/prefix`` or a local directory, for a
stage that reads from one place and writes to another.

Exit codes: 0 success, 1 a stage failed (safe to retry: every stage is
idempotent), 2 invalid arguments or configuration.

This module only parses arguments and wires components together. What a
stage does is unchanged from running its own module with ``python -m``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.scope import SessionScope
from f1_race_intelligence.storage.store import DataStore, store_from_uri
from f1_race_intelligence.utils.logging import configure_logging, get_logger, log_context

logger = get_logger("f1_race_intelligence.cli")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

STAGES = ("plan", "m2", "m3", "m4", "m5", "m6", "copy-raw")


@dataclass
class StageOutcome:
    """What a stage reports back to the entrypoint."""

    ok: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


@dataclass
class RunContext:
    """Everything a stage needs, resolved once from arguments and settings."""

    settings: AppSettings
    scope: SessionScope
    input_store: DataStore
    output_store: DataStore
    args: argparse.Namespace
    client_factory: Callable[[AppSettings], Any]


# -- argument parsing -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m f1_race_intelligence",
        description="Run F1 Race Intelligence pipeline stages as batch jobs.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help=f"Stage or comma-separated stages to run, in order: {', '.join(STAGES)}",
    )

    scope = parser.add_argument_group("scope (omit to process everything configured)")
    scope.add_argument("--year", type=int)
    scope.add_argument("--meeting-key", type=int)
    scope.add_argument("--session-key", type=int)

    config = parser.add_argument_group("configuration")
    config.add_argument("--config", help="Config file (default: F1_CONFIG_PATH or configs/config.yaml)")
    config.add_argument("--environment", choices=["local", "aws"])
    config.add_argument("--overwrite", action="store_true", help="Rebuild outputs that already exist (m2, m4, m5, copy-raw)")
    config.add_argument("--strict", action="store_true", help="Exit non-zero when validation (m3) comes back FAIL")
    config.add_argument("--plan-output", help="Also write the plan's work units to this JSON file")

    storage = parser.add_argument_group("storage")
    storage.add_argument("--storage-backend", choices=["local", "s3"])
    storage.add_argument("--bucket")
    storage.add_argument("--prefix")
    storage.add_argument("--layout", choices=["auto", "legacy", "lake"])
    storage.add_argument("--local-root", help="Directory local storage keys resolve against")
    storage.add_argument("--input-storage", help="Read from s3://bucket/prefix or a local directory")
    storage.add_argument("--input-layout", choices=["auto", "legacy", "lake"], default=None)
    storage.add_argument("--output-storage", help="Write to s3://bucket/prefix or a local directory")
    storage.add_argument("--output-layout", choices=["auto", "legacy", "lake"], default=None)
    return parser


def parse_stages(value: str) -> List[str]:
    stages = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in stages if item not in STAGES]
    if not stages or unknown:
        raise ValueError(f"Unknown stage(s) {unknown or value!r}. Supported: {', '.join(STAGES)}")
    return stages


def settings_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Command-line flags as a nested configuration overlay."""
    overrides: Dict[str, Any] = {}
    storage = {
        key: value
        for key, value in (
            ("backend", args.storage_backend),
            ("bucket", args.bucket),
            ("prefix", args.prefix),
            ("layout", args.layout),
            ("local_root", args.local_root),
        )
        if value is not None
    }
    if storage:
        overrides["storage"] = storage
    if args.environment:
        overrides["environment"] = args.environment
    if args.overwrite:
        for section in ("historical_extraction", "consolidation", "features"):
            overrides[section] = {"overwrite": True}
    return overrides


def resolve_store(settings: AppSettings, uri: Optional[str], layout: Optional[str]) -> DataStore:
    if uri:
        return store_from_uri(uri, layout=layout or "auto", storage=settings.storage)
    store = DataStore.from_settings(settings)
    if layout and layout != "auto":
        from f1_race_intelligence.storage.layout import get_layout

        store = DataStore(backend=store.backend, layout=get_layout(layout))
    return store


# -- stages ---------------------------------------------------------------


def _default_client(settings: AppSettings) -> Any:
    from f1_race_intelligence.ingestion.openf1 import F1Client

    return F1Client(settings=settings)


def run_plan(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline

    years = [ctx.scope.year] if ctx.scope.year is not None else None
    client = ctx.client_factory(ctx.settings)
    with HistoricalExtractionPipeline(ctx.settings, client=client, store=ctx.output_store) as pipeline:
        units = pipeline.plan(years)

    if ctx.args.plan_output:
        Path(ctx.args.plan_output).parent.mkdir(parents=True, exist_ok=True)
        Path(ctx.args.plan_output).write_text(json.dumps(units, indent=2), encoding="utf-8")
    print(json.dumps(units))
    return StageOutcome(ok=True, summary={"work_units": len(units)})


def run_m2(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.pipelines.historical_extraction import HistoricalExtractionPipeline

    scope, settings = ctx.scope, ctx.settings
    client = ctx.client_factory(settings)
    with HistoricalExtractionPipeline(settings, client=client, store=ctx.output_store) as pipeline:
        if scope.is_session:
            manifest = pipeline.run_session(scope.year, scope.session_key, meeting_key=scope.meeting_key)
        else:
            if scope.year is not None:
                settings.historical_extraction.years = [scope.year]
            manifest = pipeline.run()

    summary = manifest.summary()
    # A 404 means the source has no such data (pit stops in 2023); that is
    # recorded, not retried. Anything else may succeed on a re-run.
    retryable = [entry for entry in manifest.errors if entry.status_code != 404]
    summary["retryable_failures"] = len(retryable)
    return StageOutcome(
        ok=not retryable,
        summary=summary,
        reason=f"{len(retryable)} extraction(s) failed and may succeed on retry" if retryable else None,
    )


def run_m3(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.validation.runner import ValidationRunner

    report = ValidationRunner(ctx.settings, store=ctx.input_store, scope=ctx.scope).run()
    status = report.global_status.value
    failed = ctx.args.strict and status == "FAIL"
    return StageOutcome(
        ok=not failed,
        summary={"global_status": status, **report.summary(), "files": report.scope.get("files_evaluated", 0)},
        reason="validation status is FAIL and --strict was given" if failed else None,
    )


def run_m4(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.consolidation.runner import ConsolidationRunner

    report = ConsolidationRunner(
        ctx.settings, store=ctx.input_store, output_store=ctx.output_store, scope=ctx.scope
    ).run()
    summary = report.summary()
    reason = None
    if report.aborted:
        reason = report.aborted
    elif summary["sessions_failed"]:
        reason = f"{summary['sessions_failed']} session(s) failed to consolidate"
    elif ctx.scope.is_session and not (summary["sessions_processed"] or summary["sessions_skipped"]):
        reason = f"no raw laps found for session {ctx.scope.session_key}"
    return StageOutcome(ok=reason is None, summary=summary, reason=reason)


def run_m5(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.features.runner import FeatureRunner

    report = FeatureRunner(ctx.settings, store=ctx.input_store, output_store=ctx.output_store, scope=ctx.scope).run()
    summary = report.summary()
    reason = None
    if summary["sessions_failed"]:
        # With fail_on_leakage off a guard failure is written and only
        # reported; it fails the job only when configuration refused to write.
        reason = f"anti-leakage checks failed for sessions {summary['sessions_with_blocking_failures']}"
    elif ctx.scope.is_session and not (summary["sessions_processed"] or summary["sessions_skipped"]):
        reason = f"no consolidated dataset found for session {ctx.scope.session_key}"
    return StageOutcome(ok=reason is None, summary=summary, reason=reason)


def run_m6(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.modeling.runner import ModelingRunner

    if not ctx.scope.is_empty:
        logger.warning("m6_ignores_scope", extra={"reason": "the split needs the whole history; use modeling.years"})
    report = ModelingRunner(ctx.settings, store=ctx.input_store, output_store=ctx.output_store).run()
    summary = report.summary()
    return StageOutcome(ok=report.aborted is None, summary=summary, reason=report.aborted)


def run_copy_raw(ctx: RunContext) -> StageOutcome:
    from f1_race_intelligence.storage.migration import copy_raw

    extraction = ctx.settings.historical_extraction
    summary = copy_raw(
        ctx.input_store,
        ctx.output_store,
        raw_path=extraction.output_path,
        manifest_path=extraction.manifest_path,
        scope=ctx.scope,
        overwrite=ctx.args.overwrite,
    )
    failed = len(summary.errors)
    return StageOutcome(
        ok=failed == 0,
        summary=summary.to_dict(),
        reason=f"{failed} file(s) could not be copied" if failed else None,
    )


STAGE_RUNNERS: Dict[str, Callable[[RunContext], StageOutcome]] = {
    "plan": run_plan,
    "m2": run_m2,
    "m3": run_m3,
    "m4": run_m4,
    "m5": run_m5,
    "m6": run_m6,
    "copy-raw": run_copy_raw,
}


# -- entrypoint -------------------------------------------------------------


def _peak_memory_mb() -> Optional[float]:
    """Peak resident memory of this process, where the platform reports it."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 1)


def run_stage(name: str, ctx: RunContext) -> StageOutcome:
    """Run one stage with its log context, timing and failure handling."""
    with log_context(stage=name, **ctx.scope.log_fields()):
        started = time.monotonic()
        logger.info(
            "stage_started",
            extra={
                "input_storage": ctx.input_store.describe(),
                "output_storage": ctx.output_store.describe(),
                "environment": ctx.settings.environment,
            },
        )
        try:
            outcome = STAGE_RUNNERS[name](ctx)
        except Exception as exc:  # the job must end with a status, not a bare traceback
            logger.error(
                "stage_failed",
                exc_info=True,
                extra={
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            return StageOutcome(ok=False, reason=f"{type(exc).__name__}: {exc}")

        logger.log(
            20 if outcome.ok else 40,
            "stage_finished",
            extra={
                "status": "succeeded" if outcome.ok else "failed",
                "reason": outcome.reason,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_memory_mb": _peak_memory_mb(),
                "summary": outcome.summary,
            },
        )
        return outcome


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    client_factory: Callable[[AppSettings], Any] = _default_client,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        stages = parse_stages(args.stage)
        if "m2" in stages and args.session_key is not None and args.year is None:
            raise ValueError("m2 for one session needs --year as well as --session-key")
        settings = load_settings(args.config, overrides=settings_overrides(args))
        input_store = resolve_store(settings, args.input_storage, args.input_layout)
        output_store = resolve_store(settings, args.output_storage, args.output_layout)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    configure_logging(
        level=settings.logging.level,
        json_format=settings.logging.json_format,
        file_path=settings.logging.file_path,
    )

    ctx = RunContext(
        settings=settings,
        scope=SessionScope(year=args.year, meeting_key=args.meeting_key, session_key=args.session_key),
        input_store=input_store,
        output_store=output_store,
        args=args,
        client_factory=client_factory,
    )

    for name in stages:
        outcome = run_stage(name, ctx)
        if not outcome.ok:
            print(f"{name}: FAILED - {outcome.reason}", file=sys.stderr)
            return EXIT_FAILED
        print(f"{name}: OK {json.dumps(outcome.summary, default=str)}", file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
