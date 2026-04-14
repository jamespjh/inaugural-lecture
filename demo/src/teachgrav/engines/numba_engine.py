from .base import BaseEngine, to_numpy_host


class NumbaEngine(BaseEngine):
    """Numba-typed-list engine."""

    np = None

    def seed_random(self, seed):
        self.random = None
        if seed is not None:
            from ..array_numba import numba_seed
            numba_seed(seed)

    def array(self, data):
        from ..array_numba import to_numba_typed_list
        return to_numba_typed_list(to_numpy_host(data))

    def random_array(self, shape, min=0.0, max=1.0):
        from ..array_numba import numba_python_matrix
        return numba_python_matrix(shape)

    def is_python_like_engine(self):
        return True
