import numba


@numba.njit
def numba_python_matrix(size):
    return numba.typed.List([
        numba.typed.List([
            numba.core.random.random()
            for _ in range(size[0])])
        for _ in range(size[1])])
