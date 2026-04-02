import numpy as np
import jax


class ArrayAbstraction:

    def __init__(self, engine):
        self.engine = engine
        if engine == 'numpy':
            self.np = np
        elif engine in ['jax-cpu', 'jax-gpu', 'jax-metal']:
            import jax.numpy as jnp
            import jax.random as jrandom
            self.np = jnp
            self.random = jrandom
            self.key = jrandom.key(0)
            if engine == 'jax-metal':
                self.jax_device = jax.devices("METAL")[0]
            elif engine == 'jax-gpu':
                self.jax_device = jax.devices("gpu")[0]
            else:
                self.jax_device = jax.devices("cpu")[0]
        elif engine == 'mlx-cpu':
            import mlx.core as mx
            mx.set_default_device(mx.cpu)  # # type: ignore
            self.np = mx
            self.random = mx.random
        elif engine == 'mlx-gpu':
            import mlx.core as mx
            mx.set_default_device(mx.gpu)  # type: ignore
            self.np = mx
            self.random = mx.random
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Valid engines "
                f"'numpy', 'jax-cpu', 'jax-gpu', 'jax-metal'.")

    def array(self, data):
        """Create an array in the appropriate engine."""
        res = self.np.array(data)
        if self.engine in ['jax-cpu', 'jax-gpu', 'jax-metal']:
            res = jax.device_put(res, self.jax_device)
        return res

    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'numpy':
            return np.random.uniform(min, max, size=shape)
        elif self.engine in ['jax-cpu', 'jax-gpu', 'jax-metal']:
            self.key, subkey = self.random.split(self.key)  # type: ignore
            res = self.random.uniform(subkey, shape,  # type: ignore
                                      minval=min,  # type: ignore
                                      maxval=max)  # type: ignore
            res = jax.device_put(res, self.jax_device)
            return res
        elif self.engine in ['mlx-cpu', 'mlx-gpu']:
            res = self.random.uniform(shape=shape)  # type: ignore
            return res
        else:
            raise ValueError(
                f"Unknown engine '{self.engine}'."
                f"Valid engines: 'numpy', 'jax-cpu', 'jax-gpu', 'jax-metal'.")
