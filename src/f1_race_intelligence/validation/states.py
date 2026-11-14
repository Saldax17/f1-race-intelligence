"""Shared states for pipeline checks and run verdicts."""

from enum import Enum


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"


class RunVerdict(str, Enum):
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INCOMPLETE = "INCOMPLETE"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"
    ABORTED = "ABORTED"
