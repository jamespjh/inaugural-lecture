"""
Backward-compatible shim for the array-abstraction layer.

The implementation now lives in :mod:`teachgrav.engines`.
This module re-exports the public symbols that the rest of the codebase
imports from here so that existing code continues to work unchanged.
"""

from .engines import create_engine          # noqa: F401
from .engines.base import to_numpy_host     # noqa: F401


def ArrayAbstraction(engine, seed=None):
    """Return an engine instance for *engine*.

    This is a factory function that provides backward compatibility with
    the previous ``ArrayAbstraction`` class.  It delegates to
    :func:`teachgrav.engines.create_engine`.
    """
    return create_engine(engine, seed=seed)
