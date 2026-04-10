import numpy as np
import pytest
from teachgrav.laws.boids_law import BoidsLawModel
from teachgrav.scenarios import ScenarioFactory

factory = ScenarioFactory(seed=42)


def test_boids_law_output_shape_single():
    """Law returns an array with the same shape as system.data."""
    system = factory.create_scenario('boids', n_boids=5)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == system.data.shape


def test_boids_law_output_shape_many():
    """Law handles a larger flock correctly."""
    system = factory.create_scenario('boids', n_boids=20)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == (2, 20, 2)


def test_boids_dposition_equals_velocity():
    """Derivative of position is velocity."""
    system = factory.create_scenario('boids', n_boids=5)
    model = BoidsLawModel()
    delta = model.law(system)
    # delta[0] is d(position)/dt, which should equal velocity
    arr = delta.__array_namespace__()
    assert arr.allclose(delta[0], system.velocities(), atol=1e-10)


def test_boids_immobile_mask_zeroes_derivatives():
    """Immobile boids have zero derivatives."""
    system = factory.create_scenario('boids', n_boids=4)
    arr = system.data.__array_namespace__()
    # Mark the first boid as immobile
    system.immobile = arr.array([True, False, False, False])
    model = BoidsLawModel()
    delta = model.law(system)
    assert arr.allclose(delta[:, 0, :], arr.zeros((2, 2)), atol=1e-10)


def test_boids_separation_repels_close_boids():
    """Boids within the avoidance radius are pushed apart."""
    # Place two boids very close together and far from the default
    # avoidance_radius, then observe that separation dominates.
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.zeros((2, 2))
    masses = np.array([1.0, 1.0])

    data = np.array([positions, velocities])

    from teachgrav.system import System
    system = System(data, masses=masses)

    # Use avoidance_radius > distance (1.0) so separation triggers.
    # flock_attraction set very small so cohesion does not dominate.
    model = BoidsLawModel(
        flock_attraction=0.0,
        avoidance_radius=2.0,
        formation_flying_radius=0.0,
        speed_matching_strength=0.0,
    )
    delta = model.law(system)
    # Body 0 should be accelerated in the -x direction (away from body 1)
    assert delta[1, 0, 0] < 0.0
    # Body 1 should be accelerated in the +x direction (away from body 0)
    assert delta[1, 1, 0] > 0.0


def test_boids_cohesion_attracts_distant_boids():
    """Boids are attracted toward each other via cohesion."""
    positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    velocities = np.zeros((2, 2))
    masses = np.array([1.0, 1.0])
    data = np.array([positions, velocities])

    from teachgrav.system import System
    system = System(data, masses=masses)

    # Use avoidance_radius = 0 so separation never triggers;
    # cohesion only.
    model = BoidsLawModel(
        flock_attraction=0.01,
        avoidance_radius=0.0,
        formation_flying_radius=0.0,
        speed_matching_strength=0.0,
    )
    delta = model.law(system)
    # Body 0 should be attracted toward body 1 (+x direction)
    assert delta[1, 0, 0] > 0.0
    # Body 1 should be attracted toward body 0 (-x direction)
    assert delta[1, 1, 0] < 0.0


def test_boids_alignment_matches_velocities():
    """Alignment brings boid velocities closer together."""
    positions = np.array([[0.0, 0.0], [0.1, 0.0]])
    velocities = np.array([[0.0, 0.0], [10.0, 0.0]])
    masses = np.array([1.0, 1.0])
    data = np.array([positions, velocities])

    from teachgrav.system import System
    system = System(data, masses=masses)

    # Only alignment, large formation_flying_radius, no other forces
    model = BoidsLawModel(
        flock_attraction=0.0,
        avoidance_radius=0.0,
        formation_flying_radius=100.0,
        speed_matching_strength=1.0,
    )
    delta = model.law(system)
    # Boid 0 (slower) should be accelerated toward boid 1's velocity (+x)
    assert delta[1, 0, 0] > 0.0
    # Boid 1 (faster) should be decelerated toward boid 0's velocity (-x)
    assert delta[1, 1, 0] < 0.0


@pytest.mark.parametrize("n_boids", [2, 10, 50])
def test_boids_various_flock_sizes(n_boids):
    """Boids law works for a range of flock sizes."""
    system = factory.create_scenario('boids', n_boids=n_boids)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == (2, n_boids, 2)
    assert not np.any(np.isnan(np.array(delta)))
