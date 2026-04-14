"""
Backward-compatible shim for the array-abstraction layer.

The implementation now lives in :mod:`teachgrav.engines`.
This module re-exports the public symbols that the rest of the codebase
imports from here so that existing code continues to work unchanged.
"""

from .engines import create_engine          # noqa: F401
from .engines.base import to_numpy_host     # noqa: F401


def infer_shape(data):
    """Return a tuple shape for array-like or nested-list data.

    Supports array-backed values (anything with a ``shape`` attribute)
    and python/numba-style nested lists.
    """
    if hasattr(data, 'shape'):
        return tuple(data.shape)

    shape = []
    current = data
    while hasattr(current, '__len__') and hasattr(current, '__getitem__'):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]
    return tuple(shape)


def infer_ndim(data):
    """Return the number of dimensions for arrays or nested lists."""
    if hasattr(data, 'ndim'):
        return int(data.ndim)
    return len(infer_shape(data))


def move_to_device(value, target):
    """Best-effort move of backend arrays/tensors to ``target`` device.

    Args:
        value: backend-native array/tensor or python object.
        target: ``'cpu'`` or ``'gpu'``.

    Returns:
        Value moved when supported; otherwise returned unchanged.
    """
    if target not in {'cpu', 'gpu'}:
        raise ValueError("target must be 'cpu' or 'gpu'")

    namespace_fn = getattr(value, '__array_namespace__', None)
    if namespace_fn is None:
        return value

    namespace = namespace_fn()
    ns_name = getattr(namespace, '__name__', '')

    if ns_name == 'numpy':
        return value

    if ns_name == 'torch':
        if target == 'cpu' and hasattr(value, 'cpu'):
            return value.cpu()
        if target == 'gpu' and hasattr(value, 'to'):
            import torch
            if torch.cuda.is_available():
                return value.to('cuda')
            has_mps = (
                hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()
            )
            if has_mps:
                return value.to('mps')
        return value

    if ns_name.startswith('jax') and hasattr(value, 'to_device'):
        import jax
        devices = jax.devices(target)
        if devices:
            return value.to_device(devices[0])
        return value

    if ns_name.startswith('cupy'):
        try:
            import importlib
            cupy = importlib.import_module('cupy')
        except ImportError:
            return value
        if target == 'cpu':
            return cupy.asnumpy(value)
        return cupy.asarray(value)

    return value


def flatten_array(data):
    """Return a flattened 1D representation of *data*.

    Uses backend-native ``flatten`` when available and falls back to
    pure-Python flattening for list-backed containers.
    """
    if hasattr(data, 'flatten'):
        return data.flatten()
    if hasattr(data, 'ravel'):
        return data.ravel()

    flat = []

    def _flatten_py(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                _flatten_py(item)
        else:
            flat.append(value)

    _flatten_py(data)
    return flat


def reshape_array(data, shape):
    """Reshape *data* to *shape* for both arrays and python lists."""
    if hasattr(data, 'reshape'):
        return data.reshape(shape)

    if not isinstance(shape, tuple):
        shape = tuple(shape)
    if any(dim == -1 for dim in shape):
        raise ValueError("reshape_array does not support -1 for list data")

    flat = flatten_array(data)

    expected = 1
    for dim in shape:
        expected *= dim
    if expected != len(flat):
        raise ValueError(
            f"Cannot reshape list of size {len(flat)} into shape {shape}")

    def _reshape_py(values, dims):
        if len(dims) == 1:
            return values[:dims[0]]
        step = 1
        for dim in dims[1:]:
            step *= dim
        return [
            _reshape_py(values[i * step:(i + 1) * step], dims[1:])
            for i in range(dims[0])
        ]

    return _reshape_py(flat, shape)


def ArrayAbstraction(engine, seed=None):
    """Return an engine instance for *engine*.

    This is a factory function that provides backward compatibility with
    the previous ``ArrayAbstraction`` class.  It delegates to
    :func:`teachgrav.engines.create_engine`.
    """
    return create_engine(engine, seed=seed)
