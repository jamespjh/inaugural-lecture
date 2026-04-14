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
        if len(shape) != 2:
            raise ValueError("Python engine only supports 2D matrices.")
        return [[
            min + self.random.random() * (max - min)
            for _ in range(shape[1])]
            for _ in range(shape[0])]
