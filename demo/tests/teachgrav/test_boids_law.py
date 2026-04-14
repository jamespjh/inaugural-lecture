import numpy as np
import pytest
from teachgrav.laws.boids_law import BoidsLawModel
from teachgrav.scenarios import ScenarioFactory
from teachgrav.system import System
from engines import ENGINES_TO_TEST


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_law_output_shape_single(engine):
    """Law returns an array with the same shape as system.data."""
    eng_factory = ScenarioFactory(engine=engine, seed=42)
    system = eng_factory.create_scenario('boids', n_boids=5)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == system.data.shape


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_law_output_shape_many(engine):
    """Law handles a larger flock correctly."""
    eng_factory = ScenarioFactory(engine=engine, seed=42)
    system = eng_factory.create_scenario('boids', n_boids=20)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == (2, 20, 2)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_dposition_equals_velocity(engine):
    """Derivative of position is velocity."""
    eng_factory = ScenarioFactory(engine=engine, seed=42)
    system = eng_factory.create_scenario('boids', n_boids=5)
    model = BoidsLawModel()
    delta = model.law(system)
    # delta[0] is d(position)/dt, which should equal velocity
    arr = delta.__array_namespace__()
    assert arr.allclose(delta[0], system.velocities(), atol=1e-10)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_immobile_mask_zeroes_derivatives(engine):
    """Immobile boids have zero derivatives."""
    eng_factory = ScenarioFactory(engine=engine, seed=42)
    system = eng_factory.create_scenario('boids', n_boids=4)
    arr = system.data.__array_namespace__()
    # Mark the first boid as immobile
    system.immobile = arr.array([True, False, False, False])
    model = BoidsLawModel()
    delta = model.law(system)
    assert arr.allclose(delta[:, 0, :], arr.zeros((2, 2)), atol=1e-10)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_separation_repels_close_boids(engine):
    """Boids within the avoidance radius are pushed apart."""
    # Place two boids very close together and observe separation dominates.
    eng_factory = ScenarioFactory(engine=engine)
    arr = eng_factory.engine
    positions = arr.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = arr.array([[0.0, 0.0], [0.0, 0.0]])
    masses = arr.array([1.0, 1.0])
    data = arr.array([positions, velocities])
    system = System(data, masses=masses)

    # Use avoidance_radius > distance (1.0) so separation triggers.
    model = BoidsLawModel(
        flock_attraction=0.0,
        avoidance_radius=2.0,
        formation_flying_radius=0.0,
        speed_matching_strength=0.0,
    )
    delta = model.law(system)
    # Body 0 should be accelerated in the -x direction (away from body 1)
    assert float(delta[1, 0, 0]) < 0.0
    # Body 1 should be accelerated in the +x direction (away from body 0)
    assert float(delta[1, 1, 0]) > 0.0


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_cohesion_attracts_distant_boids(engine):
    """Boids are attracted toward each other via cohesion."""
    eng_factory = ScenarioFactory(engine=engine)
    arr = eng_factory.engine
    positions = arr.array([[0.0, 0.0], [10.0, 0.0]])
    velocities = arr.array([[0.0, 0.0], [0.0, 0.0]])
    masses = arr.array([1.0, 1.0])
    data = arr.array([positions, velocities])
    system = System(data, masses=masses)

    # Use avoidance_radius = 0 so separation never triggers; cohesion only.
    model = BoidsLawModel(
        flock_attraction=0.01,
        avoidance_radius=0.0,
        formation_flying_radius=0.0,
        speed_matching_strength=0.0,
    )
    delta = model.law(system)
    # Body 0 should be attracted toward body 1 (+x direction)
    assert float(delta[1, 0, 0]) > 0.0
    # Body 1 should be attracted toward body 0 (-x direction)
    assert float(delta[1, 1, 0]) < 0.0


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_boids_alignment_matches_velocities(engine):
    """Alignment brings boid velocities closer together."""
    eng_factory = ScenarioFactory(engine=engine)
    arr = eng_factory.engine
    positions = arr.array([[0.0, 0.0], [0.1, 0.0]])
    velocities = arr.array([[0.0, 0.0], [10.0, 0.0]])
    masses = arr.array([1.0, 1.0])
    data = arr.array([positions, velocities])
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
    assert float(delta[1, 0, 0]) > 0.0
    # Boid 1 (faster) should be decelerated toward boid 0's velocity (-x)
    assert float(delta[1, 1, 0]) < 0.0


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
@pytest.mark.parametrize("n_boids", [2, 10, 50])
def test_boids_various_flock_sizes(engine, n_boids):
    """Boids law works for a range of flock sizes."""
    eng_factory = ScenarioFactory(engine=engine, seed=42)
    system = eng_factory.create_scenario('boids', n_boids=n_boids)
    model = BoidsLawModel()
    delta = model.law(system)
    assert delta.shape == (2, n_boids, 2)
    assert not np.any(np.isnan(np.array(delta)))
