"""Lightweight request helpers shared by every OpenF1 endpoint method.

This project does not yet model OpenF1 response payloads (that belongs to
a future feature-engineering stage). What it does need now is a single
place to build query-parameter dicts, so that ``get_meetings``,
``get_laps``, etc. in :mod:`f1_race_intelligence.ingestion.openf1` don't
each repeat the same "drop unset filters" logic.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

QueryParams = Mapping[str, Any]


def build_params(**kwargs: Any) -> Dict[str, Any]:
    """Build an OpenF1 query-parameter dict, dropping keys whose value is ``None``."""
    return {key: value for key, value in kwargs.items() if value is not None}
