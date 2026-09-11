"""Modelling datasets: the reproducible layer between features and models.

Takes the M5 feature dataset, drops the rows that have no target, splits
the races chronologically, fits preprocessing on the training races alone,
and writes train/validation/test artifacts a model can consume.

No model is trained here. This stage exists so that whatever is trained
later is trained on data that was prepared without ever looking at the
races it will be judged on.

Like the other executable packages, this one re-exports nothing: its
runner is run with ``python -m``, and an eager import here would load that
module twice. Import from the modules directly::

    from f1_race_intelligence.modeling.runner import ModelingRunner
"""
