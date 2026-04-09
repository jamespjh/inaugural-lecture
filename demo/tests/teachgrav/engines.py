import warnings


def get_available_engines():
    """Detect and return available compute engines for testing.

    Checks for JAX (with optional Nvidia GPU) and MLX (Apple Silicon).
    Issues a warning for each engine that cannot be tested on this machine.

    Returns:
        list[str]: Engine names available for testing.
    """
    available = ['numpy']

    jax_available = False
    try:
        import jax  # noqa: F401
        jax_available = True
        available.append('jax-cpu')
    except ImportError:
        warnings.warn(
            "JAX not available; skipping jax-cpu and jax-gpu tests.",
            UserWarning,
            stacklevel=1,
        )

    if jax_available:
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                available.append('jax-gpu')
            else:
                warnings.warn(
                    "No Nvidia GPU found; skipping jax-gpu tests.",
                    UserWarning,
                    stacklevel=1,
                )
        except RuntimeError:
            warnings.warn(
                "No Nvidia GPU found; skipping jax-gpu tests.",
                UserWarning,
                stacklevel=1,
            )

    try:
        import mlx.core  # noqa: F401
        available.append('mlx-cpu')
        available.append('mlx-gpu')
    except ImportError:
        warnings.warn(
            "MLX not available (requires Apple Silicon); "
            "skipping mlx-cpu and mlx-gpu tests.",
            UserWarning,
            stacklevel=1,
        )

    return available


ENGINES_TO_TEST = get_available_engines()
