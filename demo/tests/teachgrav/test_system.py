from teachgrav.system import System, Trajectory

fixture_system = System(positions=[[0, 0], [1, 0]], velocities=[
                        [0, 0], [0, 1]], masses=[1, 1])


def test_system():
    system = fixture_system
    assert system.positions.shape == (2, 2)  # 2 bodies, 2D positions
    assert system.velocities.shape == (2, 2)  # 2 bodies, 2D velocities
    assert system.masses.shape == (2,)       # 2 bodies


def test_trajectory():
    system = fixture_system
    trajectory = Trajectory(system, steps=100)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions.shape == (101, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities.shape == (101, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies
