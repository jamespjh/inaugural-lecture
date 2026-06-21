"""
Backward-compatible shim for the array-abstraction layer.

The implementation now lives in :mod:`teachgrav.engines`.
This module re-exports the public symbols that the rest of the codebase
imports from here so that existing code continues to work unchanged.
"""

from .engines import create_engine  # noqa: F401
from .engines.base import to_numpy_host  # noqa: F401


def move_to_device(value, target):
    """Best-effort move of backend arrays/tensors to ``target`` device.

    Args:
        value: backend-native array/tensor or python object.
        target: ``'cpu'`` or ``'gpu'``.

    Returns:
        Value moved when supported; otherwise returned unchanged.
    """
    if target not in {"cpu", "gpu"}:
        raise ValueError("target must be 'cpu' or 'gpu'")

    namespace_fn = getattr(value, "__array_namespace__", None)
    if namespace_fn is None:
        return value

    namespace = namespace_fn()
    ns_name = getattr(namespace, "__name__", "")

    if ns_name == "numpy":
        return value

    if ns_name == "torch":
        if target == "cpu" and hasattr(value, "cpu"):
            return value.cpu()
        if target == "gpu" and hasattr(value, "to"):
            import torch

            if torch.cuda.is_available():
                return value.to("cuda")
            has_mps = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if has_mps:
                return value.to("mps")
        return value

    if ns_name.startswith("jax") and hasattr(value, "to_device"):
        import jax

        devices = jax.devices(target)
        if devices:
            return value.to_device(devices[0])
        return value

    if ns_name.startswith("cupy"):
        try:
            import importlib

            cupy = importlib.import_module("cupy")
        except ImportError:
            return value
        if target == "cpu":
            return cupy.asnumpy(value)
        return cupy.asarray(value)

    return value


def ArrayAbstraction(engine, seed=None):
    """Return an engine instance for *engine*.

    This is a factory function that provides backward compatibility with
    the previous ``ArrayAbstraction`` class.  It delegates to
    :func:`teachgrav.engines.create_engine`.
    """
    return create_engine(engine, seed=seed)
