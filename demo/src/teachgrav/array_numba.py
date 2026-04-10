import numba
import numpy as np


@numba.njit
def numba_seed(seed):
    np.random.seed(seed)


@numba.njit
def numba_python_matrix(size):
    return numba.typed.List([
        numba.typed.List([
            np.random.random()
            for _ in range(size[0])])
        for _ in range(size[1])])


def to_numba_typed_list(data):
    """Convert nested data to numba typed lists recursively."""
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        out = numba.typed.List()
        for item in data:
            out.append(to_numba_typed_list(item))
        return out
    if isinstance(data, np.generic):
        return data.item()
    return data
