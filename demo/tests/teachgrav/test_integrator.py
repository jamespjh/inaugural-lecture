import pytest
from teachgrav.integrator import integrate_trajectory
from teachgrav.scenarios import ScenarioFactory
import logging
from engines import ENGINES_TO_TEST
logger = logging.getLogger("Teachgrav")

_ALL_TRAJECTORY_COMBOS = [
    ('euler', 'numpy'),
    ('RK45', 'numpy'),
    ('Tsit5', 'jax-cpu'),
]

_ALL_ORBIT_COMBOS = [
    ('LSODA', 'numpy'),
    ('Tsit5', 'jax-cpu'),
]


@pytest.mark.parametrize("method,engine", [
    (m, e) for m, e in _ALL_TRAJECTORY_COMBOS if e in ENGINES_TO_TEST
])
def test_integrate_trajectory(method, engine):
    factory = ScenarioFactory(engine=engine)
    system = factory.create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method=method, dt=0.01, until=1.0)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies


@pytest.mark.parametrize("method,engine", [
    (m, e) for m, e in _ALL_ORBIT_COMBOS if e in ENGINES_TO_TEST
])
def test_close_to_start_after_one_orbit(method, engine):
    factory = ScenarioFactory(engine=engine)
    system = factory.create_scenario('sun')
    from numpy import pi
    trajectory = integrate_trajectory(
        system, method=method, dt=0.01, until=2.0 * pi)
    # After one orbit, should be close to the starting position
    start_pos = trajectory.positions()[0]
    end_pos = trajectory.positions()[-1]
    logger.info(f"Start position:\n{start_pos}")
    logger.info(f"End position:\n{end_pos}")
    assert (start_pos.__array_namespace__().
            allclose(start_pos, end_pos, atol=0.02))


def test_integrate_trajectory_scatter_3D():
    factory = ScenarioFactory()
    system = factory.create_scenario('scatter', n_bodies=5, dimensions=3)
    trajectory = integrate_trajectory(
        system, method='LSODA', dt=0.01, until=1.0)
    # 101 time steps, 5 bodies, 3D positions
    assert trajectory.positions().shape == (101, 5, 3)
    # 101 time steps, 5 bodies, 3D velocities
    assert trajectory.velocities().shape == (101, 5, 3)
    assert trajectory.masses.shape == (5,)          # 5 bodies
