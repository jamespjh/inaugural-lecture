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
from teachgrav.engines.python_engine import (
    infer_ndim,
    infer_shape,
    flatten_array,
    reshape_array,
)
from teachgrav.array_abstraction import move_to_device

from engines import ENGINES_TO_TEST


# ---------------------------------------------------------------------------
# Metaprogramming: string → class name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ('numpy', 'NumpyEngine'),
    ('python', 'PythonEngine'),
    ('numba', 'NumbaEngine'),
    ('cupy', 'CupyEngine'),
    ('jax-cpu', 'JaxCpuEngine'),
    ('jax-gpu', 'JaxGpuEngine'),
    ('jax-metal', 'JaxMetalEngine'),
    ('mlx-cpu', 'MlxCpuEngine'),
    ('mlx-gpu', 'MlxGpuEngine'),
    ('torch-cpu', 'TorchCpuEngine'),
    ('torch-gpu', 'TorchGpuEngine'),
    ('torch-mps', 'TorchMpsEngine'),
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


def test_infer_shape_numpy_array():
    arr = np.zeros((2, 3, 4))
    assert infer_shape(arr) == (2, 3, 4)


def test_infer_shape_nested_list():
    data = [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]
    assert infer_shape(data) == (2, 2, 2)


def test_infer_ndim_numpy_array():
    arr = np.zeros((2, 3, 4))
    assert infer_ndim(arr) == 3


def test_infer_ndim_nested_list():
    data = [[1.0, 2.0], [3.0, 4.0]]
    assert infer_ndim(data) == 2


def test_infer_ndim_scalar():
    assert infer_ndim(42.0) == 0


def test_engine_method_is_python_like_true_python():
    assert create_engine('python').is_python_like_engine() is True


def test_engine_method_is_python_like_true_numba():
    assert create_engine('numba').is_python_like_engine() is True


def test_engine_method_is_python_like_false_numpy():
    assert create_engine('numpy').is_python_like_engine() is False


def test_move_to_device_python_list_noop_cpu():
    data = [[1.0, 2.0], [3.0, 4.0]]
    moved = move_to_device(data, 'cpu')
    assert moved == data


def test_move_to_device_numpy_noop_gpu():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    moved = move_to_device(data, 'gpu')
    np.testing.assert_array_equal(moved, data)


def test_move_to_device_rejects_invalid_target():
    with pytest.raises(ValueError, match="target must be 'cpu' or 'gpu'"):
        move_to_device([1.0, 2.0], 'tpu')


def test_flatten_array_numpy():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    flat = flatten_array(data)
    np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 3.0, 4.0]))


def test_flatten_array_nested_list():
    data = [[1.0, 2.0], [3.0, 4.0]]
    flat = flatten_array(data)
    assert isinstance(flat, list)
    assert flat == [1.0, 2.0, 3.0, 4.0]


def test_reshape_array_numpy():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    reshaped = reshape_array(data, (2, 2))
    np.testing.assert_array_equal(reshaped, np.array([[1.0, 2.0],
                                                      [3.0, 4.0]]))


def test_reshape_array_list():
    data = [1.0, 2.0, 3.0, 4.0]
    reshaped = reshape_array(data, (2, 2))
    assert isinstance(reshaped, list)
    assert reshaped == [[1.0, 2.0], [3.0, 4.0]]


def test_reshape_array_list_invalid_size():
    with pytest.raises(ValueError, match="Cannot reshape list"):
        reshape_array([1.0, 2.0, 3.0], (2, 2))


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


def test_python_engine_random_array_shape_1d():
    engine = create_engine('python', seed=0)
    result = engine.random_array((5,))
    assert len(result) == 5


def test_python_engine_random_array_shape_3d():
    engine = create_engine('python', seed=0)
    result = engine.random_array((2, 3, 4))
    assert len(result) == 2
    assert all(len(plane) == 3 for plane in result)
    assert all(len(row) == 4 for plane in result for row in plane)


def test_python_engine_random_array_range():
    engine = create_engine('python', seed=0)
    result = engine.random_array((10, 10), min=2.0, max=5.0)
    flat = [v for row in result for v in row]
    assert all(2.0 <= v <= 5.0 for v in flat)


def test_python_engine_random_array_too_large():
    engine = create_engine('python', seed=0)
    with pytest.raises(ValueError, match="too large"):
        engine.random_array((1025, 1025))


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
    """engine.random_array() returns correct shape."""
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
