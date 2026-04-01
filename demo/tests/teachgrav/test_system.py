from teachgrav.system import System, Trajectory

from teachgrav.array_abstraction import ArrayAbstraction

ar = ArrayAbstraction('numpy')
np = ar.np

fixture_system = System(ar.np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]]),
                        masses=ar.np.array([1, 1]))


def test_system():
    system = fixture_system
    assert system.positions().shape == (2, 2)  # 2 bodies, 2D positions
    assert system.velocities().shape == (2, 2)  # 2 bodies, 2D velocities
    assert system.masses.shape == (2,)       # 2 bodies


def test_trajectory():
    system = fixture_system
    trajectory = Trajectory(system)
    # Append the initial state again for testing
    trajectory.append(system.data)
    # Append the initial state again for testing
    trajectory.append(system.data)
    # 101 time steps, 2 bodies, 2D positions
    assert trajectory.positions().shape == (3, 2, 2)
    # 101 time steps, 2 bodies, 2D velocities
    assert trajectory.velocities().shape == (3, 2, 2)
    assert trajectory.masses.shape == (2,)          # 2 bodies


def test_update():
    system = fixture_system
    new_positions = system.positions() + 1
    new_velocities = system.velocities() + 1
    new_system = system.update(np.array([new_positions, new_velocities]))
    assert np.array_equal(new_system.positions(), new_positions)
    assert np.array_equal(new_system.velocities(), new_velocities)
    assert np.array_equal(new_system.masses, system.masses)
    assert np.array_equal(new_system.immobile, system.immobile)


def test_trajectory_write_csv():
    system = fixture_system
    trajectory = Trajectory(system)
    for _ in range(10):
        trajectory.append(system.data)  # Append the initial state again
    import io
    stream = io.StringIO()
    trajectory.write(stream, format='csv')
    csv_output = stream.getvalue()
    # Check that the output has the expected
    # number of lines (header + 11 lines)
    assert len(csv_output.strip().split('\n')) == 11
    # Check that the first line corresponds
    # to the initial positions
    first_line = csv_output.strip().split('\n')[0]
    expected_first_line = ('0.00000,   0.00000,   1.00000,   0.00000,' +
                           '   0.00000,   0.00000,   0.00000,   1.00000')
    assert first_line == expected_first_line
