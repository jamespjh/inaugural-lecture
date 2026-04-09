from teachgrav.scenarios import ScenarioFactory
import numpy as np

factory = ScenarioFactory()
jax_factory = ScenarioFactory(engine='jax-cpu')


def test_create_scenario_moon():
    system = factory.create_scenario('moon')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2


def test_create_scenario_scatter():
    system = factory.create_scenario('scatter', n_bodies=10)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_create_scenario_scatter_3D():
    system = factory.create_scenario('scatter', n_bodies=10, dimensions=3)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_scenario_sun():
    system = factory.create_scenario('sun')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2
    assert system.immobile[0]
    assert not system.immobile[1]


def test_create_scenario_single_preserves_initial_state():
    system = factory.create_scenario(
        'single',
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


def test_same_seed_produces_same_scatter():
    """Same seed should produce identical scatter scenarios."""
    f1 = ScenarioFactory(seed=42)
    f2 = ScenarioFactory(seed=42)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    assert np.allclose(
        np.array(s1.positions()), np.array(s2.positions()))
    assert np.allclose(
        np.array(s1.masses), np.array(s2.masses))


def test_different_seeds_produce_different_scatter():
    """Different seeds should produce different scatter scenarios."""
    f1 = ScenarioFactory(seed=1)
    f2 = ScenarioFactory(seed=2)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    assert not np.allclose(
        np.array(s1.positions()), np.array(s2.positions()))


def test_same_seed_jax_produces_same_scatter():
    """Same seed should produce identical scatter scenarios with JAX engine."""
    import jax.numpy as jnp
    f1 = ScenarioFactory(engine='jax-cpu', seed=99)
    f2 = ScenarioFactory(engine='jax-cpu', seed=99)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    assert jnp.allclose(s1.positions(), s2.positions())
    assert jnp.allclose(s1.masses, s2.masses)


def test_different_seeds_jax_produce_different_scatter():
    """Different seeds should produce different scatter scenarios with JAX."""
    import jax.numpy as jnp
    f1 = ScenarioFactory(engine='jax-cpu', seed=1)
    f2 = ScenarioFactory(engine='jax-cpu', seed=2)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    assert not jnp.allclose(s1.positions(), s2.positions())
