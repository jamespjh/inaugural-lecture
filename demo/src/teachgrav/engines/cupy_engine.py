from .base import BaseEngine


def _ensure_cupy_array_api(cp):
    """Add minimal Array API hooks expected by this codebase."""
    if getattr(cp, "_teachgrav_array_api_patched", False):
        return

    if not hasattr(cp.ndarray, '__array_namespace__'):
        def __array_namespace__(self, api_version=None):
            return cp

        cp.ndarray.__array_namespace__ = __array_namespace__

    cp._teachgrav_array_api_patched = True


class CupyEngine(BaseEngine):
    """CuPy GPU engine."""

    def _setup(self):
        import cupy as cp
        _ensure_cupy_array_api(cp)
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
