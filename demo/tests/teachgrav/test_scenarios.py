from teachgrav.scenarios import ScenarioFactory
from teachgrav.engines.base import to_numpy_host
import numpy as np
import pytest
from engines import ENGINES_TO_TEST

factory = ScenarioFactory()


def test_create_scenario_moon():
    system = factory.create_scenario("moon")
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2


def test_create_scenario_scatter():
    system = factory.create_scenario("scatter", n_bodies=10)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_create_scenario_scatter_3D():
    system = factory.create_scenario("scatter", n_bodies=10, dimensions=3)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_scenario_sun():
    system = factory.create_scenario("sun")
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2
    assert system.immobile[0]
    assert not system.immobile[1]


def test_create_scenario_single_preserves_initial_state():
    system = factory.create_scenario(
        "single",
        position=[2.0, -1.0],
        velocity=[0.5, -0.25],
        mass=3.0,
    )

    np = system.data.__array_namespace__()
    assert system.positions().shape == (1, 2)
    assert system.velocities().shape == (1, 2)
    assert system.masses.shape == (1,)
    assert np.allclose(system.positions()[0], np.array([2.0, -1.0]))
    assert np.allclose(system.velocities()[0], np.array([0.5, -0.25]))
    assert np.allclose(system.masses, np.array([3.0]))


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_same_seed_produces_same_scatter(engine):
    """Same seed should produce identical scatter scenarios across engines."""
    f1 = ScenarioFactory(engine=engine, seed=42)
    f2 = ScenarioFactory(engine=engine, seed=42)
    s1 = f1.create_scenario("scatter", n_bodies=5)
    s2 = f2.create_scenario("scatter", n_bodies=5)
    np = s1.positions().__array_namespace__()
    assert np.allclose(s1.positions(), s2.positions())
    assert np.allclose(s1.velocities(), s2.velocities())
    assert np.allclose(s1.masses, s2.masses)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_different_seeds_produce_different_scatter(engine):
    """Different seeds should produce different scatter."""
    f1 = ScenarioFactory(engine=engine, seed=1)
    f2 = ScenarioFactory(engine=engine, seed=2)
    s1 = f1.create_scenario("scatter", n_bodies=5)
    s2 = f2.create_scenario("scatter", n_bodies=5)
    np = s1.positions().__array_namespace__()
    assert not np.allclose(s1.positions(), s2.positions())
    assert not np.allclose(s1.velocities(), s2.velocities())
    assert not np.allclose(s1.masses, s2.masses)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_engine_consistent_seed_matches_numpy(engine):
    """With engine_consistent_seed=True, same seed yields identical scatter
    as the numpy engine regardless of which engine is requested."""
    f_numpy = ScenarioFactory(engine="numpy", seed=42)
    f_other = ScenarioFactory(engine=engine, seed=42, via_numpy=True)
    s_numpy = f_numpy.create_scenario("scatter", n_bodies=5)
    s_other = f_other.create_scenario("scatter", n_bodies=5)
    assert np.allclose(
        to_numpy_host(s_numpy.positions()),
        to_numpy_host(s_other.positions()),
    )
    assert np.allclose(
        to_numpy_host(s_numpy.velocities()),
        to_numpy_host(s_other.velocities()),
    )
    assert np.allclose(
        to_numpy_host(s_numpy.masses),
        to_numpy_host(s_other.masses),
    )
