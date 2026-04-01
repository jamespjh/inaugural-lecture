import jax
import time
import logging
logger = logging.getLogger("Teachgrav")


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


def benchmark(fn, *args):
    timer = Timer(warmup=5, repeat=5)

    def ffn(*args):
        result = fn(*args)
        jax.block_until_ready(result)
    fn(*args)  # Compile
    return timer.timeit(ffn, *args)
