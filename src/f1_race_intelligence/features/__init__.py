"""Feature engineering and target construction for lap time prediction.

Reads the consolidated lap dataset, builds ``next_lap_time`` and the
history features that describe a driver's race up to the current lap, and
writes a modelling dataset under ``data/features``.

Nothing here learns from the data: scaling, imputation and model-specific
encoding estimate parameters, and belong inside a training pipeline where
they can be fitted on the training split alone.

Like the other executable packages, this one re-exports nothing: its runner
is run with ``python -m``, and an eager import here would load that module
twice. Import from the modules directly::

    from f1_race_intelligence.features.runner import FeatureRunner
"""
