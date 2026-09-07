"""Data pipelines: orchestration built on top of the ingestion layer.

Unlike the other packages, this one deliberately re-exports nothing.
``historical_extraction`` is meant to be run with ``python -m``, and an
eager import here would make Python load it twice (once as a package
attribute, once as ``__main__``), which emits a ``RuntimeWarning`` on
every run. Import from the module directly instead::

    from f1_race_intelligence.pipelines.historical_extraction import (
        HistoricalExtractionPipeline,
    )
"""
