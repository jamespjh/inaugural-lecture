import time
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

    def timeit_engine(self, fn, engine=None, *args):
        if engine in ['cupy', 'torch-gpu']:
            return self.timeit_cu(fn, *args)
        if engine in jax_engines:
            return self.timeit_jax(fn, *args)
        else:
            return self.timeit(fn, *args)


def benchmark(fn, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit(fn, *args)


def benchmark_engine(fn, engine, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit_engine(fn, engine, *args)
