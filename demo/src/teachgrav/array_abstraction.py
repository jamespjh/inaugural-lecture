import numpy as np
import jax
from operator import mul
from functools import reduce
from .engine_support import jax_engines, mlx_engines
from .engine_support import torch_engines, valid_engines


def _ensure_torch_array_api(torch):
    """Add minimal Array API hooks expected by this codebase."""
    if getattr(torch, "_teachgrav_array_api_patched", False):
        return

    if not hasattr(torch, 'bool_'):
        torch.bool_ = torch.bool

    def _torch_array(data):
        if isinstance(
                data, (list, tuple)) and any(
                torch.is_tensor(x) for x in data):
            tensors = [
                x if torch.is_tensor(x) else torch.as_tensor(x)
                for x in data
            ]
            return torch.stack(tensors)
        return torch.as_tensor(data)

    torch.array = _torch_array

    if not hasattr(torch.Tensor, '__array_namespace__'):
        def __array_namespace__(self, api_version=None):
            return torch

        torch.Tensor.__array_namespace__ = __array_namespace__

    if not hasattr(torch.Tensor, 'astype'):
        def astype(self, dtype):
            return self.to(dtype=dtype)

        torch.Tensor.astype = astype

    torch._teachgrav_array_api_patched = True


class ArrayAbstraction:

    def __init__(self, engine, seed=None):
        self.engine = engine
        self.configure_engine(engine)
        self.seed_random(seed)

    def configure_engine(self, engine):
        if engine == 'python':
            self.np = None
            import random
            self.random = random
        elif engine == 'numba':
            self.np = None
        elif engine == 'numpy':
            self.np = np
            self.random = np.random
        elif engine == 'cupy':
            import cupy as cp
            self.np = cp
            self.random = cp.random
        elif engine in torch_engines:
            import torch
            _ensure_torch_array_api(torch)
            self.np = torch
            self.random = torch.rand
        elif engine in jax_engines:
            self.configure_jax()
        elif engine in mlx_engines:
            self.configure_mlx()
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Valid engines "
                f"are: {valid_engines}.")

    def configure_jax(self):
        import jax.numpy as jnp
        import jax.random as jrandom
        self.np = jnp
        self.random = jrandom
        if self.engine == 'jax-metal':
            self.jax_device = jax.devices("METAL")[0]
        elif self.engine == 'jax-gpu':
            self.jax_device = jax.devices("gpu")[0]
        else:
            self.jax_device = jax.devices("cpu")[0]

    def configure_mlx(self):
        import mlx.core as mx
        if self.engine == 'mlx-cpu':
            mx.set_default_device(mx.cpu)
        else:
            mx.set_default_device(mx.gpu)
        self.np = mx
        self.random = mx.random

    def seed_random(self, seed):
        """Seed the random number generator for reproducibility."""
        if self.engine in jax_engines:
            self.key = self.random.key(seed if seed is not None else 0)
            return
        if seed is None:
            return
        if self.engine in torch_engines:
            self.np.manual_seed(seed)
        elif self.engine == 'numpy':
            self.random = self.random.default_rng(seed)
        elif self.engine == 'numba':
            import numba
            numba.core.random.seed(seed)
        else:
            self.random.seed(seed)

    def array(self, data):
        """Create an array in the appropriate engine."""
        res = self.np.array(data)
        if self.engine in jax_engines:
            res = jax.device_put(res, self.jax_device)
        if self.engine == 'torch-gpu':
            res = res.to('cuda')
        return res

    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'python':
            max_python_size = 1024
            if reduce(mul, shape) > max_python_size * max_python_size:
                raise ValueError(
                    f"Size {shape} is too large for native-python." +
                    f"Maximum size is {max_python_size}.")
            return self.random_python_matrix(shape)
        elif self.engine == 'numba':
            from .array_numba import numba_python_matrix
            return numba_python_matrix(shape)
        elif self.engine == 'numpy':
            return self.random.uniform(min, max, size=shape)
        elif self.engine == 'cupy':
            return self.random.uniform(low=min, high=max, size=shape)
        elif self.engine in jax_engines:
            self.key, subkey = self.random.split(self.key)
            res = self.random.uniform(subkey, shape,
                                      minval=min,
                                      maxval=max)
            res = jax.device_put(res, self.jax_device)
            return res
        elif self.engine in mlx_engines:
            res = self.random.uniform(shape=shape, low=min, high=max)
            return res
        elif self.engine in torch_engines:
            res = self.random(size=shape) * (max - min) + min
            if self.engine == 'torch-gpu':
                res = res.to('cuda')
            return res
        else:
            raise ValueError(
                f"Unknown engine '{self.engine}'."
                f"Valid engines: {valid_engines}.")

    def random_python_matrix(self, size):
        if len(size) != 2:
            raise ValueError("Python engine only supports 2D matrices.")
        return [[
            self.random.random()
            for _ in range(size[0])]
            for _ in range(size[1])]
