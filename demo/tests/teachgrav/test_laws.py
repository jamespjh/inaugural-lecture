import logging
import jax.numpy as np
from teachgrav.laws import law, flat_law
from teachgrav.scenarios import ScenarioFactory
logger = logging.getLogger(__name__)

factory = ScenarioFactory()
jax_factory = ScenarioFactory(engine='jax-cpu')


def test_law():
    system = factory.create_scenario('moon')
    derivatives = law(system)
    # 2 bodies, 4 derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt)
    assert derivatives.shape == (2, 2, 2)


def test_law_immobile():
    system = factory.create_scenario('sun')
    derivatives = law(system)
    logger.info(f"Derivatives:\n{derivatives}")
    # The Sun is immobile, so its derivatives should be zero
    assert derivatives[0][0][:].tolist() == [0.0, 0.0]
    assert derivatives[1][0][:].tolist() == [0.0, 0.0]
    # The Earth has an initial radial velocity of 1.0
    assert derivatives[0][1][:].tolist() == [0.0, 1.0]
    # The Earth should have an acceleration toward the origin of 1.0
    assert derivatives[1][1][:].tolist() == [-1.0, 0.0]


def test_law_scatter():
    system = factory.create_scenario('scatter', n_bodies=5)
    derivatives = law(system)
    assert derivatives.shape == (2, 5, 2)


def test_law_scatter_3D():
    system = factory.create_scenario('scatter', n_bodies=5, dimensions=3)
    derivatives = law(system)
    assert derivatives.shape == (2, 5, 3)


def test_law_vectorised():
    N_sys = 5
    N_bodies = 3
    # Test that the law can be called multiple times over an array of states
    systems = [
        factory.create_scenario(
            'scatter',
            n_bodies=N_bodies) for _ in range(N_sys)]
    # Our vectorisation assumes all systems have the same masses and
    # immobility, so we just take the first one
    for system in systems:
        system.masses = systems[0].masses
        system.immobile = systems[0].immobile
    simple_results = np.array([law(system) for system in systems])
    ICs = np.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    vector_results = flat_law(ICs.reshape(
        N_sys, -1), masses, immobile).reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6)
