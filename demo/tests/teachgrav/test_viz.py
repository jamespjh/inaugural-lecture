import tempfile
import os
from unittest.mock import patch

import numpy as np
import pytest

from teachgrav.visualisations.visualize import visualize
from teachgrav.visualisations.visualize import marker_sizes_from_masses
from teachgrav.visualisations.visualize import _equal_aspect_limits

from teachgrav.scenarios import ScenarioFactory
factory = ScenarioFactory()


def test_equal_aspect_limits_square_data():
    # When x and y ranges are equal, limits should be symmetric around midpoint
    mins = np.array([0.0, 0.0])
    maxs = np.array([2.0, 2.0])
    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer=0.0)
    assert xlim == (0.0, 2.0)
    assert ylim == (0.0, 2.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, y_span)


def test_equal_aspect_limits_wide_data():
    # When x range is wider, y should be expanded to match x range
    mins = np.array([0.0, 0.0])
    maxs = np.array([10.0, 2.0])
    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer=0.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, y_span)
    assert np.isclose(x_span, 10.0)
    # y should be centred on its midpoint
    assert np.isclose((ylim[0] + ylim[1]) / 2.0, 1.0)


def test_equal_aspect_limits_tall_data():
    # When y range is taller, x should be expanded to match y range
    mins = np.array([0.0, 0.0])
    maxs = np.array([2.0, 10.0])
    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer=0.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, y_span)
    assert np.isclose(y_span, 10.0)
    # x should be centred on its midpoint
    assert np.isclose((xlim[0] + xlim[1]) / 2.0, 1.0)


def test_equal_aspect_limits_buffer():
    # Buffer is added equally to both axes after equalising
    mins = np.array([0.0, 0.0])
    maxs = np.array([4.0, 2.0])
    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer=1.0)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    assert np.isclose(x_span, y_span)
    # Span should be max_range + 2*buffer = 4 + 2 = 6
    assert np.isclose(x_span, 6.0)


def test_axes_equal_aspect_ratio():
    """axes() should produce x and y spans of the same length."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from teachgrav.visualisations.visualize import axes
    from teachgrav.system import System, Trajectory

    # Wide x range, narrow y range: x spans 0..10, y spans 0..2
    # After equal-aspect, both spans should be 10 (+ 2*buffer)
    data = np.array([
        [[[0.0, 0.0], [10.0, 0.0]],    # positions at step 0
         [[0.0, 0.0], [0.0, 0.0]]],    # velocities at step 0
        [[[0.0, 1.0], [10.0, 2.0]],    # positions at step 1
         [[0.0, 0.0], [0.0, 0.0]]],    # velocities at step 1
    ])
    # data shape: (2, 2, 2, 2) → (steps, pv, bodies, coords)
    masses = np.array([1.0, 1.0])
    system = System(data[0], masses)
    traj = Trajectory(system)
    traj.data = data

    fig, ax, _ = axes(traj, options='dot')
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert np.isclose(x_span, y_span), (
        f"x span ({x_span}) != y span ({y_span}): axes are not 1:1")
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
        )

    assert mock_animate.call_count == 1
    assert mock_animate.call_args.kwargs['fps'] == 11
