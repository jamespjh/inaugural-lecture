import importlib.util


jax_engines = ['jax-cpu', 'jax-gpu', 'jax-metal']
mlx_engines = ['mlx-cpu', 'mlx-gpu']
torch_engines = ['torch-cpu', 'torch-gpu', 'torch-mps']
valid_engines = [
    'python',
    'numba',
    'numpy',
    'cupy'] + jax_engines + mlx_engines + torch_engines


def detect_cuda():
    try:
        import cupy
    except ImportError:
        return False
    return cupy.cuda.is_available()


def detect_numba():
    return importlib.util.find_spec("numba") is not None


def detect_torch():
    return importlib.util.find_spec("torch") is not None


def detect_mlx():
    return importlib.util.find_spec("mlx") is not None


def detect_jax():
    return importlib.util.find_spec("jax") is not None


def detect_jax_gpu():
    if not detect_jax():
        return False
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


def detect_jax_metal():
    if not detect_jax():
        return False
    try:
        import jax
        return len(jax.devices("METAL")) > 0
    except RuntimeError:
        return False


def detect_torch_cuda():
    if not detect_torch():
        return False
    import torch
    return torch.cuda.is_available()


def detect_torch_mps():
    if not detect_torch():
        return False
    import torch
    return torch.backends.mps.is_available()


def get_available_engines():
    """Return available engines."""
    available = ['python', 'numpy']

    if detect_numba():
        available.append('numba')

    have_cuda = detect_cuda()
    if have_cuda:
        available.append('cupy')

    if detect_jax():
        available.append('jax-cpu')
        if detect_jax_gpu():
            available.append('jax-gpu')
        if detect_jax_metal():
            available.append('jax-metal')

    if detect_mlx():
        available.append('mlx-cpu')
        available.append('mlx-gpu')

    if detect_torch():
        available.append('torch-cpu')
        if detect_torch_cuda():
            available.append('torch-gpu')
        if detect_torch_mps():
            available.append('torch-mps')

    return available


def clear_gpu_cache():
    if detect_cuda():
        import cupy
        cupy.cuda.Device().synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    if detect_jax():
        import jax
        for buf in jax.live_arrays():
            buf.delete()
    if detect_torch_cuda():
        import torch
        torch.cuda.empty_cache()
