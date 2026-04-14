import random as _random
from operator import mul
from functools import reduce
from .base import BaseEngine, to_numpy_host


class PythonEngine(BaseEngine):
    """Pure-Python (list-based) engine."""

    np = None

    def seed_random(self, seed):
        self.random = _random
        if seed is not None:
            self.random.seed(seed)

    def array(self, data):
        host = to_numpy_host(data)
        if getattr(host, "ndim", 0) == 0:
            return host.item()
        return host.tolist()

    def random_array(self, shape, min=0.0, max=1.0):
        max_python_size = 1024
        if reduce(mul, shape) > max_python_size * max_python_size:
            raise ValueError(
                f"Size {shape} is too large for native-python. "
                f"Maximum total size is "
                f"{max_python_size * max_python_size} elements "
                f"({max_python_size}x{max_python_size}).")

        def build_nested(current_shape):
            if len(current_shape) == 0:
                return min + self.random.random() * (max - min)
            return [
                build_nested(current_shape[1:])
                for _ in range(current_shape[0])
            ]

        return build_nested(shape)

    def is_python_like_engine(self):
        return True


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
