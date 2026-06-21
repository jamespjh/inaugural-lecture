import pytest
import logging
from teachgrav.laws.true_law import TrueLawModel
from teachgrav.scenarios import ScenarioFactory
from teachgrav.engines.python_engine import infer_shape
from teachgrav.engine_support import get_available_engines
from engines import ENGINES_TO_TEST

logger = logging.getLogger(__name__)


def assert_shape(value, expected_shape):
    """Assert shape for both array-backed and list-backed outputs."""
    assert infer_shape(value) == expected_shape


# Engines that support the pure-Python for-loop gravity law.
# ScenarioFactory handles python/numba engines transparently by building the
# scenario with numpy internally; TrueLawModel converts data on the fly.
ENGINES_WITH_PYTHON_LAW = ENGINES_TO_TEST + [
    e for e in ["python", "numba"] if e in get_available_engines()
]


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law(engine):
    sys_factory = ScenarioFactory(engine=engine)
    system = sys_factory.create_scenario("moon")
    law_model = TrueLawModel(factory=sys_factory)
    derivatives = law_model.law(system)
    # 2 bodies, 4 derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt)
    assert_shape(derivatives, (2, 2, 2))


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law_immobile(engine):
    sys_factory = ScenarioFactory(engine=engine)
    system = sys_factory.create_scenario("sun")
    law_model = TrueLawModel(factory=sys_factory)
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
    sys_factory = ScenarioFactory(engine=engine)
    system = sys_factory.create_scenario("scatter", n_bodies=5)
    law_model = TrueLawModel(factory=sys_factory)
    derivatives = law_model.law(system)
    assert_shape(derivatives, (2, 5, 2))


@pytest.mark.parametrize("engine", ENGINES_WITH_PYTHON_LAW)
def test_law_scatter_3D(engine):
    sys_factory = ScenarioFactory(engine=engine)
    system = sys_factory.create_scenario("scatter", n_bodies=5, dimensions=3)
    law_model = TrueLawModel(factory=sys_factory)
    derivatives = law_model.law(system)
    assert_shape(derivatives, (2, 5, 3))


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_law_vectorised(engine):
    factory = ScenarioFactory(engine=engine)
    law_model = TrueLawModel(factory=factory)
    N_sys = 5
    N_bodies = 3
    # Test that the law can be called multiple times over an array of states
    systems = [
        factory.create_scenario(
            "scatter",
            n_bodies=N_bodies,
            fixed_masses=[1.0, 1.0, 1.0],
        )
        for _ in range(N_sys)
    ]
    simple_results = factory.engine.array(
        [law_model.law(system) for system in systems]
    )
    ICs = factory.engine.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    ICs_flat = ICs.reshape((N_sys, -1))
    results = law_model.flat_law(ICs_flat, masses, immobile)
    vector_results = results.reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6
    )
