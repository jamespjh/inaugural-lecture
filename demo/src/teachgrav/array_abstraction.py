import numpy as np
#import jax.numpy as jnp
#import jax.random as jrandom

#key = jrandom.key(0)

class ArrayAbstraction:
    """A simple class to demonstrate array abstraction."""
    def __init__(self, engine):
        self.engine = engine
        if engine == 'numpy':
            self.np = np
        elif engine == 'jax':
            self.np = jnp
        else:
            raise ValueError(f"Unknown engine '{engine}'. Valid engines: 'numpy', 'jax'.")
    
    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'numpy':
            return np.random.uniform(min, max, size=shape)
        elif self.engine == 'jax':
            key = jrandom.PRNGKey(0)
            return jrandom.uniform(key, shape, minval=min, maxval=max)