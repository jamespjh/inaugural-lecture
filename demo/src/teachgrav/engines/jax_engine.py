from .base import BaseEngine


class JaxBaseEngine(BaseEngine):
    """Shared base for all JAX engine variants."""

    def _setup(self):
        import jax.numpy as jnp
        import jax.random as jrandom

        self.np = jnp
        self.random = jrandom
        self.jax_device = self._pick_device()

    def _pick_device(self):
        raise NotImplementedError

    def seed_random(self, seed):
        self.key = self.random.key(seed if seed is not None else 0)

    def array(self, data):
        import jax

        res = self.np.array(data)
        return jax.device_put(res, self.jax_device)

    def random_array(self, shape, min=0.0, max=1.0):
        import jax

        self.key, subkey = self.random.split(self.key)
        res = self.random.uniform(subkey, shape, minval=min, maxval=max)
        return jax.device_put(res, self.jax_device)


class JaxCpuEngine(JaxBaseEngine):
    """JAX on CPU."""

    def _pick_device(self):
        import jax

        return jax.devices("cpu")[0]


class JaxGpuEngine(JaxBaseEngine):
    """JAX on GPU."""

    def _pick_device(self):
        import jax

        return jax.devices("gpu")[0]


class JaxMetalEngine(JaxBaseEngine):
    """JAX on Apple Metal GPU."""

    def _pick_device(self):
        import jax

        return jax.devices("METAL")[0]
