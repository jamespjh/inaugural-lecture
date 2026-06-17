import pytest
from teachgrav.integrator import integrate_trajectory
from teachgrav.scenarios import ScenarioFactory
import logging
from engines import ENGINES_TO_TEST, AVAILABLE_ENGINES
logger = logging.getLogger("Teachgrav")

_METHOD_ENGINE_COMBOS = [
    (m, e) for m, e in [
        ('RK45', 'numpy'),
        ('LSODA', 'numpy'),
    #    ('Tsit5', 'jax-cpu'), TODO: JAX integration needs factory propagated, needs tidy up
    ] if e in ENGINES_TO_TEST
]


def test_euler_trajectory():
    """Euler integration runs and output has correct shape."""
    factory = ScenarioFactory(engine='numpy')
    system = factory.create_scenario('moon')
    trajectory = integrate_trajectory(
        system, factory=factory, method='euler', dt=0.01, until=1.0)
    assert trajectory.positions().shape == (101, 2, 2)
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)


@pytest.mark.parametrize("method,engine", _METHOD_ENGINE_COMBOS)
def test_integrate_trajectory(method, engine):
    factory = ScenarioFactory(engine=engine)
    system = factory.create_scenario('moon')
    trajectory = integrate_trajectory(
        system, factory=factory, method=method, dt=0.01, until=1.0)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("method,engine", _METHOD_ENGINE_COMBOS)
def test_close_to_start_after_one_orbit(method, engine):
    factory = ScenarioFactory(engine=engine)
    system = factory.create_scenario('sun')
    from numpy import pi
    trajectory = integrate_trajectory(
        system, factory=factory, method=method, dt=0.01, until=2.0 * pi)
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
        system, factory=factory, method='LSODA', dt=0.01, until=1.0)
    # 101 time steps, 5 bodies, 3D positions
    assert trajectory.positions().shape == (101, 5, 3)
    # 101 time steps, 5 bodies, 3D velocities
    assert trajectory.velocities().shape == (101, 5, 3)
    assert trajectory.masses.shape == (5,)          # 5 bodies


_jax_single_body_cases = (
    [('jax-cpu', 'Tsit5', 1.0, 1e-6)] if 'jax-cpu' in AVAILABLE_ENGINES
    else []
)


@pytest.mark.parametrize("engine,method,until,atol", [
    ('numpy', 'euler', 2.0, 1e-12),
] + _jax_single_body_cases)
def test_constant_single_body_moves_at_expected_position(
        engine, method, until, atol):
    factory = ScenarioFactory(engine=engine)
    system = factory.create_scenario(
        'single',
        position=[2.0, -1.0],
        velocity=[0.5, -0.25],
        mass=1.0,
    )
    trajectory = integrate_trajectory(
        system,
        factory=factory,
        method=method,
        law='constant',
        dt=0.01,
        until=until,
    )

    np = trajectory.positions().__array_namespace__()
    start_pos = trajectory.positions()[0, 0]
    end_pos = trajectory.positions()[-1, 0]
    velocity = trajectory.velocities()[0, 0]
    expected_end = start_pos + velocity * until
    assert np.allclose(end_pos, expected_end, atol=atol)
