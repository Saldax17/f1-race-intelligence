"""Orchestration of a validation run.

Run it with::

    python -m f1_race_intelligence.validation.runner

The run streams the raw tree file by file: read, analyse once, apply the
file rules, fold the file's aggregates into a :class:`DatasetIndex`, and
move on. Nothing accumulates the records themselves, so validating a full
season costs the same memory as validating one file.

Raw data is never written to. A file that cannot be read is reported and
the pass continues, the same way the extraction pipeline steps over a
failing endpoint instead of abandoning the run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from f1_race_intelligence.config.settings import AppSettings, load_settings
from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFile, RawFileError
from f1_race_intelligence.utils.logging import configure_logging, get_logger
from f1_race_intelligence.validation import validators
from f1_race_intelligence.validation.manifest_index import ManifestIndex
from f1_race_intelligence.validation.models import (
    RuleStatus,
    Severity,
    ValidationReport,
    ValidationResult,
)
from f1_race_intelligence.validation.report import save_report
from f1_race_intelligence.validation.specs import get_spec
from f1_race_intelligence.validation.validators import (
    FILE_RULE_FUNCTIONS,
    FILE_RULES,
    RULE_COVERAGE,
    RULE_IDENTIFIERS,
    RULE_STRUCTURE,
    DatasetIndex,
    FileContext,
)

logger = get_logger(__name__)

PIPELINE_NAME = "data_validation"
PIPELINE_VERSION = "1.0.0"

SAMPLING_DISCLAIMER = (
    "Sampling was enabled: only the first max_records_per_file records of each file were "
    "inspected, so these results do NOT represent an exhaustive validation of the dataset. "
    "Cross-file rules (identifiers, coverage) can also report findings that a full run would "
    "not, because they only saw part of each file."
)


class ValidationRunner:
    """Validates the raw data produced by the extraction pipeline."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        *,
        catalog: Optional[RawDataCatalog] = None,
        manifest: Optional[ManifestIndex] = None,
    ) -> None:
        """Create a runner.

        Args:
            settings: Application settings; loaded from config when omitted.
            catalog: Reader over the raw tree; defaults to the configured
                ``raw_path``.
            manifest: Extraction manifest to explain missing data with. When
                omitted the most recent one is loaded, unless configuration
                turns manifest use off.

        Raises:
            ValueError: if configuration enables an unknown rule.
        """
        self._settings = settings or load_settings()
        self._config = self._settings.validation
        self._catalog = catalog or RawDataCatalog(base_path=self._config.raw_path)

        unknown = [rule for rule in self._config.enabled_rules if rule not in validators.ALL_RULES]
        if unknown:
            raise ValueError(
                f"Unknown rules in validation configuration: {sorted(unknown)}. "
                f"Supported rules: {sorted(validators.ALL_RULES)}"
            )

        self._manifest = manifest
        if manifest is None and self._config.use_manifest:
            self._manifest = ManifestIndex.load_latest(self._config.manifest_path)

    # -- orchestration ----------------------------------------------------

    def run(self) -> ValidationReport:
        """Validate everything under the raw path and return the report."""
        report = ValidationReport(
            pipeline_name=PIPELINE_NAME,
            pipeline_version=PIPELINE_VERSION,
            fail_on_error=self._config.fail_on_error,
            config=self._config_snapshot(),
            source={
                "raw_path": str(self._catalog.base_path),
                "manifest": self._manifest.describe() if self._manifest else None,
            },
        )

        files = self._catalog.discover()
        logger.info(
            "validation_start",
            extra={
                "pipeline": PIPELINE_NAME,
                "raw_path": str(self._catalog.base_path),
                "files": len(files),
                "rules": list(self._config.enabled_rules),
            },
        )

        index = DatasetIndex()
        executed: Set[Tuple[Optional[str], str]] = set()
        flagged: Set[Tuple[Optional[str], str]] = set()
        records_inspected = 0
        records_available = 0
        files_sampled = 0

        for raw_file in files:
            outcome = self._validate_file(raw_file, index, report, executed, flagged)
            records_inspected += outcome.inspected
            records_available += outcome.available
            files_sampled += 1 if outcome.sampled else 0

        self._run_dataset_rules(index, report, executed, flagged)
        self._add_clean_rule_results(report, index, executed, flagged)

        report.scope = {
            "files_evaluated": len(files),
            "endpoints": sorted(index.endpoints_present),
            "sessions": len(index.sessions_declared) or len(
                {key for keys in index.session_keys_by_endpoint.values() for key in keys}
            ),
            "records_inspected": records_inspected,
            "records_available": records_available,
        }
        self._add_sampling_disclosure(report, files_sampled, records_inspected, records_available)

        report.finished_at = datetime.now(timezone.utc)
        written = save_report(report, self._config.report_path, write_csv=self._config.write_csv)

        logger.info(
            "validation_summary",
            extra={
                "pipeline": PIPELINE_NAME,
                "global_status": report.global_status.value,
                **report.summary(),
                "duration_seconds": report.duration_seconds,
                "report_path": str(written[0]),
            },
        )
        return report

    # -- per-file ---------------------------------------------------------

    def _validate_file(
        self,
        raw_file: RawFile,
        index: DatasetIndex,
        report: ValidationReport,
        executed: Set[Tuple[Optional[str], str]],
        flagged: Set[Tuple[Optional[str], str]],
    ) -> "_FileOutcome":
        """Read and validate one file, never letting a bad one end the run."""
        try:
            envelope = self._catalog.load(raw_file)
        except RawFileError as exc:
            logger.error(
                "raw_file_unreadable",
                extra={"endpoint": raw_file.endpoint, "path": str(raw_file.path), "error": str(exc)},
            )
            executed.add((raw_file.endpoint, RULE_STRUCTURE))
            flagged.add((raw_file.endpoint, RULE_STRUCTURE))
            report.add(
                ValidationResult(
                    rule=RULE_STRUCTURE,
                    description="File could not be read as a raw data envelope",
                    severity=Severity.ERROR,
                    status=RuleStatus.FAIL,
                    endpoint=raw_file.endpoint,
                    file=str(raw_file.path),
                    details={"error": str(exc), **raw_file.partition},
                )
            )
            return _FileOutcome(inspected=0, available=0, sampled=False)

        available = len(envelope.records)
        limit = self._config.max_records_per_file
        sampled = limit is not None and available > limit
        records = envelope.records[:limit] if sampled else envelope.records

        context = FileContext(
            endpoint=raw_file.endpoint,
            path=raw_file.path,
            partition=raw_file.partition,
            spec=get_spec(raw_file.endpoint),
            records=records,
            total_records=available,
            sampled=sampled,
            missing_value_threshold=self._config.missing_value_threshold,
        )

        if context.spec is None:
            report.add(
                ValidationResult(
                    rule=RULE_STRUCTURE,
                    description="No validation spec is defined for this endpoint; content checks skipped",
                    severity=Severity.INFO,
                    status=RuleStatus.SKIPPED,
                    endpoint=raw_file.endpoint,
                    file=str(raw_file.path),
                )
            )

        analysis = validators.analyze_records(records, context.spec)

        for rule in FILE_RULES:
            if rule not in self._config.enabled_rules:
                continue
            executed.add((raw_file.endpoint, rule))
            findings = FILE_RULE_FUNCTIONS[rule](context, analysis)
            for finding in findings:
                if finding.status is RuleStatus.FAIL:
                    flagged.add((raw_file.endpoint, rule))
                report.add(finding)

        index.observe(context)
        logger.debug(
            "file_validated",
            extra={
                "endpoint": raw_file.endpoint,
                "path": str(raw_file.path),
                "records": len(records),
                **raw_file.partition,
            },
        )
        return _FileOutcome(inspected=len(records), available=available, sampled=sampled)

    # -- dataset-wide -----------------------------------------------------

    def _run_dataset_rules(
        self,
        index: DatasetIndex,
        report: ValidationReport,
        executed: Set[Tuple[Optional[str], str]],
        flagged: Set[Tuple[Optional[str], str]],
    ) -> None:
        if RULE_IDENTIFIERS in self._config.enabled_rules:
            executed.add((None, RULE_IDENTIFIERS))
            for finding in validators.validate_identifiers(index):
                if finding.status is RuleStatus.FAIL:
                    flagged.add((None, RULE_IDENTIFIERS))
                report.add(finding)

        if RULE_COVERAGE in self._config.enabled_rules:
            executed.add((None, RULE_COVERAGE))
            findings = validators.validate_coverage(
                index,
                self._expected_endpoints(index),
                self._manifest,
                expected_sessions=self._expected_sessions(),
            )
            for finding in findings:
                if finding.status is RuleStatus.FAIL:
                    flagged.add((None, RULE_COVERAGE))
                report.add(finding)

    def _expected_sessions(self) -> Optional[Sequence[int]]:
        """The sessions this extraction targeted, if the manifest says so.

        Falling back to ``None`` lets the coverage rule work from the files
        on disk, which is all we can know without a manifest.
        """
        if self._manifest is None:
            return None
        processed = self._manifest.sessions_processed
        return processed or None

    def _expected_endpoints(self, index: DatasetIndex) -> Sequence[str]:
        """Which endpoints should be present for every session.

        The extraction manifest is the best answer, because it records what
        was actually asked for. Without one, fall back to what the raw tree
        already holds, which can only detect gaps between sessions.
        """
        if self._manifest and self._manifest.expected_endpoints:
            endpoints = self._manifest.expected_endpoints
        else:
            endpoints = sorted(index.endpoints_present)
        return [endpoint for endpoint in endpoints if endpoint not in {"meetings", "sessions"}]

    def _add_clean_rule_results(
        self,
        report: ValidationReport,
        index: DatasetIndex,
        executed: Set[Tuple[Optional[str], str]],
        flagged: Set[Tuple[Optional[str], str]],
    ) -> None:
        """Record the rules that ran and found nothing.

        Without this the report would only list problems, and a reader
        could not tell a clean endpoint from one nobody checked. One line
        per endpoint and rule keeps that visible without a line per file.
        """
        for endpoint, rule in sorted(executed - flagged, key=lambda item: (item[0] or "", item[1])):
            report.add(
                ValidationResult(
                    rule=rule,
                    description="No issues found",
                    severity=Severity.INFO,
                    status=RuleStatus.PASS,
                    endpoint=endpoint,
                    details={"files_checked": index.files_by_endpoint.get(endpoint)} if endpoint else {},
                )
            )

    def _add_sampling_disclosure(
        self,
        report: ValidationReport,
        files_sampled: int,
        records_inspected: int,
        records_available: int,
    ) -> None:
        """State plainly whether this run was exhaustive.

        A sampled run is reported as a WARNING on purpose: it must never be
        mistaken for the dataset's official validation, and a run that only
        looked at part of the data should not be able to come back clean.
        """
        limit = self._config.max_records_per_file
        report.sampling = {
            "enabled": limit is not None,
            "max_records_per_file": limit,
            "files_sampled": files_sampled,
            "records_inspected": records_inspected,
            "records_available": records_available,
            "exhaustive": limit is None,
        }

        if limit is None:
            return

        report.sampling["disclaimer"] = SAMPLING_DISCLAIMER
        report.add(
            ValidationResult(
                rule="sampling",
                description=SAMPLING_DISCLAIMER,
                severity=Severity.WARNING,
                status=RuleStatus.FAIL,
                details={
                    "max_records_per_file": limit,
                    "files_sampled": files_sampled,
                    "records_inspected": records_inspected,
                    "records_available": records_available,
                },
            )
        )

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "raw_path": self._config.raw_path,
            "fail_on_error": self._config.fail_on_error,
            "missing_value_threshold": self._config.missing_value_threshold,
            "max_records_per_file": self._config.max_records_per_file,
            "use_manifest": self._config.use_manifest,
            "enabled_rules": list(self._config.enabled_rules),
        }


class _FileOutcome:
    """How many records one file contributed, and whether it was sampled."""

    __slots__ = ("inspected", "available", "sampled")

    def __init__(self, inspected: int, available: int, sampled: bool) -> None:
        self.inspected = inspected
        self.available = available
        self.sampled = sampled


def main() -> None:
    """Entry point for ``python -m f1_race_intelligence.validation.runner``."""
    settings = load_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)

    report = ValidationRunner(settings).run()
    summary = report.summary()

    print(
        f"Validation finished: {report.global_status.value} — "
        f"{summary['errors']} errors, {summary['warnings']} warnings, {summary['info']} info "
        f"across {report.scope.get('files_evaluated', 0)} files."
    )


if __name__ == "__main__":
    main()
