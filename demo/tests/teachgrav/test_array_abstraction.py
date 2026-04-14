"""Tests for the array abstraction layer (engines subpackage).

These tests are independent of scenarios and only exercise:
- engine_name_to_class_name metaprogramming
- create_engine factory
- BaseEngine.array() and BaseEngine.random_array()
- to_numpy_host helper
"""
import pytest
import numpy as np

from teachgrav.engines import (
    create_engine,
    engine_name_to_class_name,
    to_numpy_host,
    NumpyEngine,
    PythonEngine,
)
from engines import ENGINES_TO_TEST


# ---------------------------------------------------------------------------
# Metaprogramming: string → class name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ('numpy',      'NumpyEngine'),
    ('python',     'PythonEngine'),
    ('numba',      'NumbaEngine'),
    ('cupy',       'CupyEngine'),
    ('jax-cpu',    'JaxCpuEngine'),
    ('jax-gpu',    'JaxGpuEngine'),
    ('jax-metal',  'JaxMetalEngine'),
    ('mlx-cpu',    'MlxCpuEngine'),
    ('mlx-gpu',    'MlxGpuEngine'),
    ('torch-cpu',  'TorchCpuEngine'),
    ('torch-gpu',  'TorchGpuEngine'),
    ('torch-mps',  'TorchMpsEngine'),
])
def test_engine_name_to_class_name(name, expected):
    assert engine_name_to_class_name(name) == expected


# ---------------------------------------------------------------------------
# Factory: create_engine returns correct type
# ---------------------------------------------------------------------------

def test_create_engine_numpy():
    engine = create_engine('numpy')
    assert isinstance(engine, NumpyEngine)


def test_create_engine_python():
    engine = create_engine('python')
    assert isinstance(engine, PythonEngine)


def test_create_engine_unknown_raises():
    with pytest.raises(ValueError, match="Unknown engine"):
        create_engine('not-a-real-engine')


# ---------------------------------------------------------------------------
# to_numpy_host helper
# ---------------------------------------------------------------------------

def test_to_numpy_host_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_numpy_host(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


def test_to_numpy_host_python_list():
    result = to_numpy_host([1.0, 2.0])
    assert isinstance(result, np.ndarray)


def test_to_numpy_host_scalar():
    result = to_numpy_host(42.0)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# NumpyEngine — array() and random_array()
# ---------------------------------------------------------------------------

def test_numpy_engine_array_round_trip():
    engine = create_engine('numpy')
    data = [1.0, 2.0, 3.0]
    result = engine.array(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(data))


def test_numpy_engine_random_array_shape():
    engine = create_engine('numpy', seed=0)
    result = engine.random_array((3, 4))
    assert result.shape == (3, 4)


def test_numpy_engine_random_array_range():
    engine = create_engine('numpy', seed=0)
    result = engine.random_array((100,), min=-1.0, max=1.0)
    assert float(result.min()) >= -1.0
    assert float(result.max()) <= 1.0


def test_numpy_engine_seed_reproducibility():
    e1 = create_engine('numpy', seed=42)
    e2 = create_engine('numpy', seed=42)
    r1 = e1.random_array((10,))
    r2 = e2.random_array((10,))
    np.testing.assert_array_equal(r1, r2)


def test_numpy_engine_different_seeds():
    e1 = create_engine('numpy', seed=1)
    e2 = create_engine('numpy', seed=2)
    r1 = e1.random_array((10,))
    r2 = e2.random_array((10,))
    assert not np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# PythonEngine — array() and random_array()
# ---------------------------------------------------------------------------

def test_python_engine_array_round_trip():
    engine = create_engine('python')
    result = engine.array([1.0, 2.0])
    assert isinstance(result, list)
    assert result == [1.0, 2.0]


def test_python_engine_array_scalar():
    engine = create_engine('python')
    result = engine.array(np.float64(3.14))
    assert isinstance(result, float)


def test_python_engine_random_array_shape():
    engine = create_engine('python', seed=0)
    result = engine.random_array((3, 4))
    assert len(result) == 3
    assert all(len(row) == 4 for row in result)


def test_python_engine_random_array_range():
    engine = create_engine('python', seed=0)
    result = engine.random_array((10, 10), min=2.0, max=5.0)
    flat = [v for row in result for v in row]
    assert all(2.0 <= v <= 5.0 for v in flat)


def test_python_engine_random_array_too_large():
    engine = create_engine('python', seed=0)
    with pytest.raises(ValueError, match="too large"):
        engine.random_array((1025, 1025))


def test_python_engine_random_array_not_2d():
    engine = create_engine('python', seed=0)
    with pytest.raises(ValueError, match="2D"):
        engine.random_array((5,))


# ---------------------------------------------------------------------------
# Available engines — smoke tests via parametrize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_name", ENGINES_TO_TEST)
def test_create_engine_available(engine_name):
    """create_engine succeeds for every available engine."""
    engine = create_engine(engine_name, seed=7)
    assert engine is not None


@pytest.mark.parametrize("engine_name", ENGINES_TO_TEST)
def test_array_round_trip_available_engines(engine_name):
    """engine.array() produces something for each available engine."""
    engine = create_engine(engine_name, seed=7)
    result = engine.array([1.0, 2.0, 3.0])
    # Convert back to numpy for comparison
    host = to_numpy_host(result)
    np.testing.assert_allclose(host, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("engine_name", ENGINES_TO_TEST)
def test_random_array_shape_available_engines(engine_name):
    """engine.random_array() returns correct shape for each available engine."""
    engine = create_engine(engine_name, seed=7)
    result = engine.random_array((4, 3))
    host = to_numpy_host(result)
    assert host.shape == (4, 3)


@pytest.mark.parametrize("engine_name", ENGINES_TO_TEST)
def test_random_array_range_available_engines(engine_name):
    """engine.random_array() values are within [min, max]."""
    engine = create_engine(engine_name, seed=7)
    result = engine.random_array((50,), min=0.5, max=2.5)
    host = to_numpy_host(result).astype(float)
    assert float(host.min()) >= 0.5 - 1e-6
    assert float(host.max()) <= 2.5 + 1e-6
