import tempfile
import os
from unittest.mock import patch

import numpy as np
import pytest

from teachgrav.visualisations.visualize import visualize
from teachgrav.visualisations.visualize import marker_sizes_from_masses

from teachgrav.scenarios import ScenarioFactory
factory = ScenarioFactory()


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
