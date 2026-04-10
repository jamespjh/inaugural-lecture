import numpy as np
from operator import mul
from functools import reduce
from .engine_support import jax_engines, mlx_engines
from .engine_support import torch_engines, valid_engines


def to_numpy_host(x):
    """Convert backend arrays/tensors to host NumPy arrays."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


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
        elif engine == 'numba':
            self.np = None
        elif engine == 'numpy':
            self.np = np
        elif engine == 'cupy':
            import cupy as cp
            self.np = cp
        elif engine in torch_engines:
            import torch
            _ensure_torch_array_api(torch)
            self.np = torch
        elif engine in jax_engines:
            self.configure_jax()
        elif engine in mlx_engines:
            self.configure_mlx()
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Valid engines "
                f"are: {valid_engines}.")

    def configure_jax(self):
        import jax
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
        if self.engine in torch_engines:
            import torch
            # Create a per-instance generator for reproducibility
            self.random = torch.Generator()
            if seed is not None:
                self.random.manual_seed(seed)
        elif self.engine == 'numpy':
            if seed is not None:
                self.random = np.random.default_rng(seed)
            else:
                self.random = np.random.default_rng()
        elif self.engine == 'cupy':
            import cupy as cp
            if seed is not None:
                self.random = cp.random.RandomState(seed)
            else:
                self.random = cp.random
        elif self.engine == 'numba':
            if seed is not None:
                from .array_numba import numba_seed
                numba_seed(seed)
        elif self.engine in mlx_engines:
            # Use NumPy RNG for MLX to ensure reproducibility
            if seed is not None:
                self.random = np.random.default_rng(seed)
            else:
                self.random = np.random.default_rng()
        elif self.engine == 'python':
            import random
            self.random = random
            if seed is not None:
                self.random.seed(seed)

    def array(self, data):
        """Create an array in the appropriate engine."""
        if self.engine == 'python':
            host = to_numpy_host(data)
            if getattr(host, "ndim", 0) == 0:
                return host.item()
            return host.tolist()
        if self.engine == 'numba':
            from .array_numba import to_numba_typed_list
            return to_numba_typed_list(to_numpy_host(data))

        res = self.np.array(data)
        if self.engine in jax_engines:
            import jax
            res = jax.device_put(res, self.jax_device)
        if self.engine == 'torch-gpu':
            res = res.to('cuda')
        if self.engine == 'torch-mps':
            if res.dtype == self.np.float64:
                res = res.to(dtype=self.np.float32)
            res = res.to('mps')
        return res

    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'python':
            max_python_size = 1024
            if reduce(mul, shape) > max_python_size * max_python_size:
                raise ValueError(
                    f"Size {shape} is too large for native-python. "
                    f"Maximum total size is "
                    f"{max_python_size * max_python_size} elements "
                    f"({max_python_size}x{max_python_size}).")
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
            import jax
            res = jax.device_put(res, self.jax_device)
            return res
        elif self.engine in mlx_engines:
            # Use NumPy RNG with per-instance seeding, then convert to MLX
            if self.random is not None:
                res_np = self.random.uniform(min, max, size=shape)
            else:
                raise ValueError(
                    "MLX engine requires seeding for RNG.")
            res = self.np.array(res_np)
            return res
        elif self.engine in torch_engines:
            # Use per-instance generator if available
            if self.random is not None:
                res = self.np.rand(size=shape,
                                   generator=self.random) * (max - min) + min
            else:
                res = self.np.rand(size=shape) * (max - min) + min
            if self.engine == 'torch-gpu':
                res = res.to('cuda')
            if self.engine == 'torch-mps':
                res = res.to('mps')
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
            for _ in range(size[1])]
            for _ in range(size[0])]
