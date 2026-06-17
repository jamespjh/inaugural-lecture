import logging
import numpy as np
from teachgrav.engines.numba_engine import NumbaEngine
from .pure_python_laws import _python_gravity_flat_law, _numba_gravity_flat_law
from .pl import PLModel
from ..array_abstraction import to_numpy_host

logger = logging.getLogger("Teachgrav")


class TrueLawModel(PLModel):
    def __init__(self, factory):
        super().__init__(factory=factory, G=1.0, power=2.0)

    def _uses_python_engine(self):
        """Return True if the factory uses a Python-like engine."""
        if self.factory is None:
            return False
        engine = self.factory.engine
        return engine.is_python_like_engine()

    def _uses_numba_engine(self):
        """Return True if the factory uses a Numba engine."""
        if self.factory is None:
            return False
        engine = self.factory.engine
        return isinstance(engine, NumbaEngine)

    def law(self, system):
        """Compute the derivatives of the state.

        For python/numba engines, system data is converted from any backend
        (numpy, JAX, etc.) to plain Python before computing with for loops.
        Returns a numpy array with the same shape as the input system data.
        """
        if self._uses_python_engine():
            numpy_data = to_numpy_host(system.data)
            data = numpy_data.flatten().tolist()
            masses = to_numpy_host(system.masses).tolist()
            immobile = to_numpy_host(system.immobile).tolist()
            if self._uses_numba_engine():
                result = _numba_gravity_flat_law(
                    self.G, data, masses, immobile)
            else:
                result = _python_gravity_flat_law(
                    self.G, data, masses, immobile)
            return np.array(result).reshape(numpy_data.shape)
        return super().law(system)

    @staticmethod
    def _to_py_list(array_like):
        """Convert any array-like (numpy, JAX, list, …) to a Python list."""
        return to_numpy_host(array_like).ravel().tolist()

    def flat_law(self, data, masses, immobile):
        """Compute gravitational derivatives from a flat state vector.

        For python and numba engines, inputs are converted from any backend
        format to plain Python lists before computing with for loops, and
        the result is returned as a numpy array.
        For all other engines we delegate to the vectorised PLModel
        implementation.
        """
        if self._uses_python_engine():
            py_data = self._to_py_list(data)
            py_masses = self._to_py_list(masses)
            py_immobile = [bool(x) for x in self._to_py_list(immobile)]
            if self._uses_numba_engine():
                result = _numba_gravity_flat_law(
                    self.G, py_data, py_masses, py_immobile)
            else:
                result = _python_gravity_flat_law(
                    self.G, py_data, py_masses, py_immobile)
            return np.array(result)
        return super().flat_law(data, masses, immobile)
