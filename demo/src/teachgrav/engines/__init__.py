"""
Engine subpackage: one class per array-backend engine.

Use :func:`create_engine` to obtain an engine instance by name.
"""

from .base import BaseEngine, to_numpy_host  # noqa: F401
from .numpy_engine import NumpyEngine  # noqa: F401
from .python_engine import PythonEngine  # noqa: F401
from .numba_engine import NumbaEngine  # noqa: F401
from .cupy_engine import CupyEngine  # noqa: F401
from .jax_engine import (  # noqa: F401
    JaxCpuEngine,
    JaxGpuEngine,
    JaxMetalEngine,
)
from .mlx_engine import MlxCpuEngine, MlxGpuEngine  # noqa: F401
from .torch_engine import (  # noqa: F401
    TorchCpuEngine,
    TorchGpuEngine,
    TorchMpsEngine,
)

from ..engine_support import valid_engines


def engine_name_to_class_name(name):
    """Convert a dash-separated engine name to its engine class name.

    Examples::

        engine_name_to_class_name('numpy')     # -> 'NumpyEngine'
        engine_name_to_class_name('jax-cpu')   # -> 'JaxCpuEngine'
        engine_name_to_class_name('torch-mps') # -> 'TorchMpsEngine'
    """
    return "".join(part.capitalize() for part in name.split("-")) + "Engine"


def create_engine(name, seed=None):
    """Return an initialised engine instance for the given engine *name*.

    Parameters
    ----------
    name:
        One of the recognised engine names (e.g. ``'numpy'``, ``'jax-cpu'``).
    seed:
        Optional integer seed for the random-number generator.

    Raises
    ------
    ValueError
        If *name* is not a recognised engine name.
    """
    if name not in valid_engines:
        raise ValueError(
            f"Unknown engine '{name}'. Valid engines are: {valid_engines}."
        )
    class_name = engine_name_to_class_name(name)
    cls = globals()[class_name]
    return cls(seed=seed)
