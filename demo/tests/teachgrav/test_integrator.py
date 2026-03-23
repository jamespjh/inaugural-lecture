from teachgrav.integrator import integrate_trajectory


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
        system, method='rk4', dt=0.01, until=1.0)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies
