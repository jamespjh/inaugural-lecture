import pytest
import numpy as np
import logging
from teachgrav.laws.true_law import TrueLawModel
from teachgrav.scenarios import ScenarioFactory
from engines import ENGINES_TO_TEST
logger = logging.getLogger(__name__)

factory = ScenarioFactory()
model = TrueLawModel()
law = model.law
flat_law = model.flat_law


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


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_law_scatter_3D(engine):
    eng_factory = ScenarioFactory(engine=engine)
    system = eng_factory.create_scenario('scatter', n_bodies=5, dimensions=3)
    derivatives = law(system)
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


# ------------------------------------------------------------------ #
# Tests for the pure-Python (numpy-free) nested-for-loop law
# ------------------------------------------------------------------ #

def _moon_flat_data():
    """Return flat state, masses and immobile flags for Earth-Moon in 2-D.

    Flat layout: [pos_earth_x, pos_earth_y, pos_moon_x, pos_moon_y,
                  vel_earth_x, vel_earth_y, vel_moon_x, vel_moon_y]
    """
    data = [0.0, 0.0, 1.0, 0.0,   # positions: earth at origin, moon at (1,0)
            0.0, 0.0, 0.0, 1.0]   # velocities: earth still, moon moving up
    masses = [1.0, 0.01]
    immobile = [False, False]
    return data, masses, immobile


def test_python_engine_flat_law_returns_correct_length():
    """Python-engine flat_law returns a list with the right length."""
    python_factory = ScenarioFactory(engine='python')
    python_model = TrueLawModel(factory=python_factory)
    data, masses, immobile = _moon_flat_data()

    result = python_model.flat_law(data, masses, immobile)

    # 2 bodies × 2 dimensions × 2 (positions + velocities) = 8 elements
    assert len(result) == 8


def test_python_engine_flat_law_matches_numpy():
    """Python for-loop implementation agrees with the numpy result."""
    python_factory = ScenarioFactory(engine='python')
    python_model = TrueLawModel(factory=python_factory)
    numpy_model = TrueLawModel()       # default factory → numpy path

    data, masses, immobile = _moon_flat_data()

    python_result = python_model.flat_law(data, masses, immobile)
    numpy_result = numpy_model.flat_law(
        np.array(data),
        np.array(masses),
        np.array(immobile, dtype=bool),
    )

    assert len(python_result) == len(numpy_result)
    for i, (p, n) in enumerate(zip(python_result, numpy_result)):
        assert abs(p - n) < 1e-10, (
            f"Mismatch at index {i}: python={p}, numpy={n}"
        )


def test_python_engine_flat_law_immobile_body():
    """Immobile bodies have zero derivatives in the python implementation."""
    python_factory = ScenarioFactory(engine='python')
    python_model = TrueLawModel(factory=python_factory)

    # Sun (body 0) fixed at origin; Earth (body 1) at (1, 0) moving up
    data = [0.0, 0.0, 1.0, 0.0,   # positions
            0.0, 0.0, 0.0, 1.0]   # velocities
    masses = [1.0, 0.01]
    immobile = [True, False]       # Sun is immobile

    result = python_model.flat_law(data, masses, immobile)

    # Sun derivatives (indices 0-1 for d_pos, 4-5 for d_vel) must be zero
    assert result[0] == 0.0, "Sun d_pos_x should be zero"
    assert result[1] == 0.0, "Sun d_pos_y should be zero"
    assert result[4] == 0.0, "Sun d_vel_x should be zero"
    assert result[5] == 0.0, "Sun d_vel_y should be zero"

    # Earth's position derivative equals its velocity
    assert abs(result[2] - 0.0) < 1e-10, "Earth d_pos_x = vel_x = 0"
    assert abs(result[3] - 1.0) < 1e-10, "Earth d_pos_y = vel_y = 1"

    # Earth should accelerate toward the Sun (negative x direction)
    assert result[6] < 0.0, "Earth should accelerate toward Sun (negative x)"
    assert abs(result[7]) < 1e-10, "Earth d_vel_y should be zero (no y force)"


def test_python_engine_flat_law_3d():
    """Python-engine flat_law works for 3-D systems."""
    python_factory = ScenarioFactory(engine='python')
    python_model = TrueLawModel(factory=python_factory)
    numpy_model = TrueLawModel()

    # Two bodies in 3-D: body 0 at origin, body 1 at (1,0,0)
    data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,   # positions (3-D)
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # velocities (3-D)
    masses = [1.0, 0.5]
    immobile = [False, False]

    python_result = python_model.flat_law(data, masses, immobile)
    numpy_result = numpy_model.flat_law(
        np.array(data),
        np.array(masses),
        np.array(immobile, dtype=bool),
    )

    assert len(python_result) == 12  # 2 bodies × 3 dims × 2 = 12
    for i, (p, n) in enumerate(zip(python_result, numpy_result)):
        assert abs(p - n) < 1e-10, (
            f"3-D mismatch at index {i}: python={p}, numpy={n}"
        )
