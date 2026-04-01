from teachgrav.viz import visualize
import tempfile
import os

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
