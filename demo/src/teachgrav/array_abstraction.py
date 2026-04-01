import numpy as np


class ArrayAbstraction:
    """A simple class to demonstrate array abstraction."""

    def __init__(self, engine):
        self.engine = engine
        if engine == 'numpy':
            self.np = np
        elif engine == 'jax':
            import jax.numpy as jnp
            import jax.random as jrandom
            self.np = jnp
            self.random=jrandom
            self.key = jrandom.key(0)
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Valid engines: 'numpy', 'jax'.")

    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'numpy':
            return np.random.uniform(min, max, size=shape)
        elif self.engine == 'jax':
            self.key, subkey = self.random.split(self.key)
            return self.random.uniform(subkey, shape, minval=min, maxval=max)