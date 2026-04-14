import pytest
import logging
from teachgrav.laws.true_law import TrueLawModel, PYTHON_LIKE_ENGINES
from teachgrav.scenarios import ScenarioFactory
from teachgrav.engine_support import get_available_engines
from engines import ENGINES_TO_TEST
logger = logging.getLogger(__name__)

factory = ScenarioFactory()
model = TrueLawModel()
law = model.law
flat_law = model.flat_law

# Engines that support the pure-Python for-loop gravity law.
# Scenarios for these engines are created with numpy; TrueLawModel converts
# input data internally when called with a python/numba engine factory.
ENGINES_WITH_PYTHON_LAW = ENGINES_TO_TEST + [
    e for e in ['python', 'numba'] if e in get_available_engines()
]


def _make_law_and_system(engine, scenario, **kwargs):
    """Return (law_model, system) for the given engine and scenario.

    Scenarios for python/numba engines are created with numpy (since System
    requires Array API); TrueLawModel converts input data internally.
    """
    scenario_engine = 'numpy' if engine in PYTHON_LIKE_ENGINES else engine
    sys_factory = ScenarioFactory(engine=scenario_engine)
    law_factory = ScenarioFactory(engine=engine)
    system = sys_factory.create_scenario(scenario, **kwargs)
    law_model = TrueLawModel(factory=law_factory)
    return law_model, system


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law(engine):
    law_model, system = _make_law_and_system(engine, 'moon')
    derivatives = law_model.law(system)
    # 2 bodies, 4 derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt)
    assert derivatives.shape == (2, 2, 2)


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law_immobile(engine):
    law_model, system = _make_law_and_system(engine, 'sun')
    derivatives = law_model.law(system)
    logger.info(f"Derivatives:\n{derivatives}")
    # The Sun is immobile, so its derivatives should be zero
    assert derivatives[0][0][:].tolist() == [0.0, 0.0]
    assert derivatives[1][0][:].tolist() == [0.0, 0.0]
    # The Earth has an initial radial velocity of 1.0
    assert derivatives[0][1][:].tolist() == [0.0, 1.0]
    # The Earth should have an acceleration toward the origin of 1.0
    assert derivatives[1][1][:].tolist() == [-1.0, 0.0]


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law_scatter(engine):
    law_model, system = _make_law_and_system(engine, 'scatter', n_bodies=5)
    derivatives = law_model.law(system)
    assert derivatives.shape == (2, 5, 2)


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law_scatter_3D(engine):
    law_model, system = _make_law_and_system(
        engine, 'scatter', n_bodies=5, dimensions=3)
    derivatives = law_model.law(system)
    assert derivatives.shape == (2, 5, 3)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_law_vectorised(engine):
    factory = ScenarioFactory(engine=engine)
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
