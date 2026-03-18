from teachgrav.viz import visualize
import tempfile
import os


def test_visualize():
    from teachgrav.scenarios import create_scenario
    from teachgrav.integrator import integrate_trajectory
    system = create_scenario('moon')
    trajectory = integrate_trajectory(
        system, method='euler', dt=0.01, steps=100)
    # Just test that it runs without error and creates a file

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = f"{tmpdir}/trajectory.mp4"
        visualize(trajectory, output=output_file)
        assert os.path.exists(output_file)
