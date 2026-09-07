"""Data validation over the extracted raw data.

Raw data is immutable here: this package reads ``data/raw``, judges it,
and writes a report. It never cleans, fills or rewrites anything — that
belongs to the stages that build a dataset from it.

Like ``pipelines``, this package re-exports nothing on purpose. Its runner
is meant to be executed with ``python -m``, and an eager import here would
load that module twice. Import from the modules directly::

    from f1_race_intelligence.validation.runner import ValidationRunner
"""
