import logging
from teachgrav.laws import law, flat_law
from teachgrav.scenarios import ScenarioFactory
logger = logging.getLogger(__name__)

factory = ScenarioFactory()

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


def test_law_scatter_3D_jax():
    jax_factory = ScenarioFactory(engine='jax-cpu')
    system = jax_factory.create_scenario('scatter', n_bodies=5, dimensions=3)
    derivatives = law(system)
    assert derivatives.shape == (2, 5, 3)


def test_law_scatter_3D_metal():
    metal_factory = ScenarioFactory(engine='mlx-gpu')
    system = metal_factory.create_scenario('scatter', n_bodies=5, dimensions=3)
    derivatives = law(system)
    assert derivatives.shape == (2, 5, 3)


def t_law_vectorised(factory):
    N_sys = 5
    N_bodies = 3
    # Test that the law can be called multiple times over an array of states
    systems = [
        factory.create_scenario(
            'scatter',
            n_bodies=N_bodies,
            fixed_masses=[1.0, 1.0, 1.0],
        )
        for _ in range(N_sys)
    ]
    simple_results = factory.engine.array([law(system) for system in systems])
    ICs = factory.engine.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    ICs_flat = ICs.reshape((N_sys, -1))
    results = flat_law(ICs_flat, masses, immobile)
    vector_results = results.reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6)


def test_law_vectorised():
    t_law_vectorised(factory)


def test_law_vectorised_jax():
    t_law_vectorised(ScenarioFactory(engine='jax-cpu'))


def test_law_vectorised_metal():
    t_law_vectorised(ScenarioFactory(engine='mlx-cpu'))
    assert False