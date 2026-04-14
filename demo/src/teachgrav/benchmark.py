import time
from typing import Callable, Iterable, Optional
from .engine_support import jax_engines


class Timer:
    def __init__(self, warmup=10, repeat=100):
        self.warmup = warmup
        self.repeat = repeat

    def timeit(self, fn, *args):
        for _ in range(self.warmup):
            fn(*args)
        tic = time.perf_counter()
        for _ in range(self.repeat):
            fn(*args)
        toc = time.perf_counter()
        return (toc - tic) / self.repeat

    def timeit_cu(self, fn, *args):
        from cupyx.profiler import benchmark
        res = benchmark(
            fn, n_repeat=self.repeat, n_warmup=self.warmup, args=args)
        cpu = res.cpu_times.mean()
        gpu = res.gpu_times.mean()
        # Return the maximum of CPU and GPU time as the benchmark result
        return max(cpu, gpu)

    def timeit_jax(self, fn, *args):
        import jax

        def ffn(*args):
            res = fn(*args)
            jax.block_until_ready(res)
            return res
        return self.timeit(ffn, *args)

    def timeit_mps(self, fn, *args):
        import torch

        def ffn(*args):
            res = fn(*args)
            torch.mps.synchronize()
            return res
        return self.timeit(ffn, *args)

    def timeit_torch_cuda(self, fn, *args):
        import torch

        def ffn(*args):
            res = fn(*args)
            torch.cuda.synchronize()
            return res
        return self.timeit(ffn, *args)

    def timeit_engine(self, fn, engine=None, *args):
        if engine == 'cupy':
            return self.timeit_cu(fn, *args)
        if engine == 'torch-gpu':
            return self.timeit_torch_cuda(fn, *args)
        if engine in jax_engines:
            return self.timeit_jax(fn, *args)
        if engine == 'torch-mps':
            return self.timeit_mps(fn, *args)
        else:
            return self.timeit(fn, *args)


def benchmark(fn, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit(fn, *args)


def benchmark_engine(fn, engine, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit_engine(fn, engine, *args)


def benchmark_range(
    fn_at_size: Callable[[int], Callable[[], object]],
    sizes: Iterable[int],
    engine: Optional[str] = None,
) -> list[tuple[int, float]]:
    """Benchmark *fn_at_size* over an iterable of *sizes*.

    Args:
        fn_at_size: Callable that accepts a single ``size`` argument and
            returns a zero-argument callable suitable for timing.
        sizes: Iterable of size values to sweep over.
        engine: Engine name string used to select the correct timer variant.
            Pass ``None`` to use the plain CPU timer.

    Returns:
        List of ``(size, time)`` pairs where *time* is the mean wall-clock
        time in seconds.
    """
    results = []
    for size in sizes:
        fn = fn_at_size(size)
        t = benchmark_engine(fn, engine)
        results.append((size, t))
    return results
