import numpy as np
from abc import ABC, abstractmethod


def to_numpy_host(x):
    """Convert backend arrays/tensors to host NumPy arrays."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


class BaseEngine(ABC):
    """Abstract base class for all array-backend engines."""

    #: The numpy-like module for this engine (set by subclasses).
    np = None

    def __init__(self, seed=None):
        self._setup()
        self.seed_random(seed)

    def _setup(self):
        """Override to perform one-time engine initialisation."""
        pass

    @abstractmethod
    def seed_random(self, seed) -> None:
        """Seed (or initialise) the random number generator."""
        raise NotImplementedError

    @abstractmethod
    def array(self, data):
        """Return an engine-native array created from *data*."""
        raise NotImplementedError

    @abstractmethod
    def random_array(self, shape, min=0.0, max=1.0):
        """Return a uniformly-distributed random array of *shape*."""
        raise NotImplementedError

    def is_python_like_engine(self) -> bool:
        """Return True for list-backed engines (python/numba)."""
        return False
