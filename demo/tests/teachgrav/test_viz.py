import tempfile
import os
from unittest.mock import patch

import numpy as np
import pytest

from teachgrav.visualisations.visualize import visualize
from teachgrav.visualisations.visualize import marker_sizes_from_masses
from teachgrav.visualisations.visualize import _equal_limits
from teachgrav.visualisations.visualize import _set_line_positions

from teachgrav.scenarios import ScenarioFactory
factory = ScenarioFactory()


def test_equal_limits_square_data():
    # When all ranges are equal, limits should be symmetric around midpoint
    mins = np.array([0.0, 0.0])
    maxs = np.array([2.0, 2.0])
    xlim, ylim = _equal_limits(mins, maxs, buffer=0.0)
    assert xlim == (0.0, 2.0)
    assert ylim == (0.0, 2.0)
    assert np.isclose(xlim[1] - xlim[0], ylim[1] - ylim[0])


def test_equal_limits_dominant_axis_2d():
    # When one axis spans more, all axes match the dominant span
    mins = np.array([0.0, 0.0])
    maxs = np.array([10.0, 2.0])
    xlim, ylim = _equal_limits(mins, maxs, buffer=0.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, 10.0)
    assert np.isclose(y_span, 10.0)
    # y should be centred on its midpoint
    assert np.isclose((ylim[0] + ylim[1]) / 2.0, 1.0)


def test_equal_limits_buffer():
    # Buffer is added outside the data range on each side
    mins = np.array([0.0, 0.0])
    maxs = np.array([4.0, 2.0])
    xlim, ylim = _equal_limits(mins, maxs, buffer=1.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, y_span)
    # Span should be max_range + 2*buffer = 4 + 2 = 6
    assert np.isclose(x_span, 6.0)


def test_equal_limits_3d_equal_range():
    """All three axes should span the same range."""
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([4.0, 4.0, 4.0])
    xlim, ylim, zlim = _equal_limits(mins, maxs, buffer=0.0)
    for lo, hi in (xlim, ylim, zlim):
        assert np.isclose(hi - lo, 4.0)


def test_equal_limits_3d_dominant_axis():
    """When one axis spans more than the others, all limits match it."""
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([10.0, 2.0, 2.0])
    xlim, ylim, zlim = _equal_limits(mins, maxs, buffer=0.0)
    for span in (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]):
        assert np.isclose(span, 10.0)


def test_equal_limits_3d_buffer():
    """Buffer is added outside the data range on each side."""
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([4.0, 4.0, 4.0])
    xlim, ylim, zlim = _equal_limits(mins, maxs, buffer=1.0)
    for lo, hi in (xlim, ylim, zlim):
        # 4 data span + 2*buffer = 6
        assert np.isclose(hi - lo, 6.0)


def test_axes_equal_spans():
    """axes() should produce equal x/y data spans regardless of figsize."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from teachgrav.visualisations.visualize import axes
    from teachgrav.system import System, Trajectory

    data = np.array([
        [[[0.0, 0.0], [10.0, 0.0]],    # positions at step 0
         [[0.0, 0.0], [0.0, 0.0]]],    # velocities at step 0
        [[[0.0, 1.0], [10.0, 2.0]],    # positions at step 1
         [[0.0, 0.0], [0.0, 0.0]]],    # velocities at step 1
    ])
    masses = np.array([1.0, 1.0])
    system = System(data[0], masses)
    traj = Trajectory(system)
    traj.data = data

    fig, ax, _ = axes(traj, options='dot', figsize=(6.4, 3.6))
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert np.isclose(x_span, y_span), (
        f"x_span ({x_span:.4f}) != y_span ({y_span:.4f}): "
        "equal data ranges expected")
    plt.close(fig)


def test_visualize():
    from teachgrav.integrator import integrate_trajectory
    system = factory.create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method='euler', dt=0.01, until=1.0)
    # Just test that it runs without error and creates a file

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = f"{tmpdir}/trajectory.mp4"
        visualize(trajectory, output=output_file)
        assert os.path.exists(output_file)


def test_marker_sizes_from_masses_log_linear_mapping():
    masses = np.array([1.0, 100.0, 10000.0])
    fig_width_points = 720.0

    marker_sizes = marker_sizes_from_masses(masses, fig_width_points)

    expected_min = fig_width_points / 500.0
    expected_max = fig_width_points / 50.0

    assert np.isclose(marker_sizes[0], expected_min)
    assert np.isclose(marker_sizes[-1], expected_max)
    assert np.isclose(marker_sizes[1], 0.5 * (expected_min + expected_max))


def test_marker_sizes_from_masses_rejects_non_positive():
    with pytest.raises(ValueError, match='strictly positive'):
        marker_sizes_from_masses(np.array([1.0, 0.0, 10.0]), 720.0)


def test_visualize_passes_fps_to_animate():
    from teachgrav.integrator import integrate_trajectory

    system = factory.create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method='euler', dt=0.01, until=1.0)

    with patch('teachgrav.visualisations.visualize.animate') as mock_animate:
        visualize(
            trajectory,
            output='out.mp4',
            mode='video',
            options='trail',
            duration=2,
            fps=11,
            figsize=(8.0, 4.0),
        )

    assert mock_animate.call_count == 1
    assert mock_animate.call_args.kwargs['fps'] == 11
    assert mock_animate.call_args.kwargs['figsize'] == (8.0, 4.0)


def test_axes_3d_returns_3d_axes():
    """axes() on a 3-D trajectory should return a 3-D Axes object."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from teachgrav.visualisations.visualize import axes
    from teachgrav.system import System, Trajectory

    data = np.array([
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    ])
    masses = np.array([1.0, 1.0])
    system = System(data[0], masses)
    traj = Trajectory(system)
    traj.data = data

    fig, ax, lines = axes(traj, options='trail', figsize=(6.4, 7.2))
    assert ax.name == '3d', f"Expected 3d axes, got {ax.name}"
    assert len(lines) == 2
    plt.close(fig)


def test_set_line_positions_2d():
    """_set_line_positions sets x/y data on a 2-D line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _set_line_positions(line, positions, is_3d=False)
    x_data, y_data = line.get_data()
    assert np.allclose(x_data, positions[:, 0])
    assert np.allclose(y_data, positions[:, 1])
    plt.close(fig)


def test_set_line_positions_3d():
    """_set_line_positions sets x/y/z data on a 3-D line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _set_line_positions(line, positions, is_3d=True)
    xyz = line.get_data_3d()
    assert np.allclose(xyz[0], positions[:, 0])
    assert np.allclose(xyz[1], positions[:, 1])
    assert np.allclose(xyz[2], positions[:, 2])
    plt.close(fig)
