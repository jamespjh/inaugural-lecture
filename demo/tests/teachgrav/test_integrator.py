from teachgrav.integrator import integrate_trajectory
import numpy as np


def test_integrate_trajectory():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method='euler', dt=0.01, until=1.0)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies


def test_integrate_trajectory_rk4():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method='RK45', dt=0.01, until=1.0)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies


def test_close_to_start_after_one_orbit():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('sun')
    trajectory = integrate_trajectory(
        system, method='RK45', dt=0.01, until=2.0*np.pi)
    # After one orbit, should be close to the starting position
    start_pos = trajectory.positions()[0]
    end_pos = trajectory.positions()[-1]
    np.testing.assert_allclose(start_pos, end_pos, atol=0.01)


def test_integrate_trajectory_scatter_3D():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('scatter', n_bodies=5, seed=42, dimensions=3)
    trajectory = integrate_trajectory(
        system, method='RK45', dt=0.01, until=1.0)
    # 101 time steps, 5 bodies, 3D positions
    assert trajectory.positions().shape == (101, 5, 3)
    # 101 time steps, 5 bodies, 3D velocities
    assert trajectory.velocities().shape == (101, 5, 3)
    assert trajectory.masses.shape == (5,)          # 5 bodies
