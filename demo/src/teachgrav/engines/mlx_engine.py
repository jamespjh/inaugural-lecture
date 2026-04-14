import numpy as np
from .base import BaseEngine


class MlxBaseEngine(BaseEngine):
    """Shared base for all MLX engine variants."""

    def _setup(self):
        import mlx.core as mx
        self._configure_device(mx)
        self.np = mx

    def _configure_device(self, mx):
        raise NotImplementedError

    def seed_random(self, seed):
        # MLX PRNG is not yet reproducibly seedable; use NumPy RNG instead.
        if seed is not None:
            self.random = np.random.default_rng(seed)
        else:
            self.random = np.random.default_rng()

    def array(self, data):
        return self.np.array(data)

    def random_array(self, shape, min=0.0, max=1.0):
        res_np = self.random.uniform(min, max, size=shape)
        return self.np.array(res_np)


class MlxCpuEngine(MlxBaseEngine):
    """MLX on CPU."""

    def _configure_device(self, mx):
        mx.set_default_device(mx.cpu)


class MlxGpuEngine(MlxBaseEngine):
    """MLX on GPU (Apple Silicon)."""

    def _configure_device(self, mx):
        mx.set_default_device(mx.gpu)
