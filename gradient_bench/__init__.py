"""gradient_bench - the science layer for HPLC method development.

Phase 0 of the Gradient Bench app: the 4-parameter Bayesian optimiser and the
peak picker, extracted from notebooks into an importable, tested library so an
API can sit on top of them.

    from gradient_bench.core import space, optimiser, drift, uncertainty, picker

Nothing here imports Flask or knows about HTTP. `scripts/replay_corpus.py`
exercises the whole thing headless.
"""
__version__ = "2.0.0"
APP_VERSION = "2.0.0"
VERSION_TAG = "version 2"
