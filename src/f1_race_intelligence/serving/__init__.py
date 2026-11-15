"""Local inference prototype (M8): M6's fitted preprocessing plus a model registered by M7.

``contract``, ``model`` and ``service`` need only the base dependencies.
``app`` needs FastAPI (the ``serve`` extra) and is not imported here, so the
service can be used and tested without it.
"""
