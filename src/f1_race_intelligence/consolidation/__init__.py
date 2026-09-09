"""Consolidation of validated raw data into the lap-grain analytical dataset.

Reads ``data/raw``, joins the endpoints onto one row per driver and lap,
and writes Parquet under ``data/processed``. Raw data is never modified.

Like ``pipelines`` and ``validation``, this package re-exports nothing: its
runner is executed with ``python -m``, and an eager import here would load
that module twice. Import from the modules directly::

    from f1_race_intelligence.consolidation.runner import ConsolidationRunner
"""
