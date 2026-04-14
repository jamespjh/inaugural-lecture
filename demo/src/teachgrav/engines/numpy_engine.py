import numpy as np
from .base import BaseEngine


class NumpyEngine(BaseEngine):
    """NumPy CPU engine."""

    np = np

    def seed_random(self, seed):
        if seed is not None:
            self.random = np.random.default_rng(seed)
        else:
            self.random = np.random.default_rng()

    def array(self, data):
        return self.np.array(data)

    def random_array(self, shape, min=0.0, max=1.0):
        return self.random.uniform(min, max, size=shape)
