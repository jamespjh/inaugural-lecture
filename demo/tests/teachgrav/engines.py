import warnings
from teachgrav.engine_support import (
    get_available_engines,
    valid_engines,
)
AVAILABLE_ENGINES = get_available_engines()

DONT_TEST_ENGINES = {'python', 'numba'}
ENGINES_TO_TEST = [
    engine for engine in AVAILABLE_ENGINES if engine not in DONT_TEST_ENGINES
]

_missing_engines = [
    engine for engine in valid_engines if engine not in ENGINES_TO_TEST
]
if _missing_engines:
    warnings.warn(
        "Skipping unavailable engines in tests: " +
        ", ".join(_missing_engines),
        UserWarning,
        stacklevel=2,
    )
