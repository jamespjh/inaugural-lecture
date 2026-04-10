from .base import BaseEngine


class CupyEngine(BaseEngine):
    """CuPy GPU engine."""

    def _setup(self):
        import cupy as cp
        self.np = cp

    def seed_random(self, seed):
        import cupy as cp
        if seed is not None:
            self.random = cp.random.RandomState(seed)
        else:
            self.random = cp.random

    def array(self, data):
        return self.np.array(data)

    def random_array(self, shape, min=0.0, max=1.0):
        return self.random.uniform(low=min, high=max, size=shape)
