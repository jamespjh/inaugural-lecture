from teachgrav.scenarios import ScenarioFactory
from teachgrav.engines.base import to_numpy_host
import numpy as np
import pytest
from engines import ENGINES_TO_TEST

factory = ScenarioFactory()


def test_create_scenario_moon():
    system = factory.create_scenario('moon')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2


def test_create_scenario_scatter():
    system = factory.create_scenario('scatter', n_bodies=10)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_create_scenario_scatter_3D():
    system = factory.create_scenario('scatter', n_bodies=10, dimensions=3)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_scenario_sun():
    system = factory.create_scenario('sun')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2
    assert system.immobile[0]
    assert not system.immobile[1]


def test_create_scenario_single_preserves_initial_state():
    system = factory.create_scenario(
        'single',
        position=[2.0, -1.0],
        velocity=[0.5, -0.25],
        mass=3.0,
    )

    np = system.data.__array_namespace__()
    assert system.positions().shape == (1, 2)
    assert system.velocities().shape == (1, 2)
    assert system.masses.shape == (1,)
    assert np.allclose(system.positions()[0], np.array([2.0, -1.0]))
    assert np.allclose(system.velocities()[0], np.array([0.5, -0.25]))
    assert np.allclose(system.masses, np.array([3.0]))


def test_solar_system_default():
    """Default: 4 planets (0,1,4,1 moons), 1 sun, 6 total moons."""
    system = factory.create_scenario('solar_system')
    # 1 sun + 4 planets + (0+1+4+1) moons = 11 bodies
    assert len(system.positions()) == 11
    assert len(system.velocities()) == 11
    assert len(system.masses) == 11
    # No bodies should be immobile
    assert not any(system.immobile)


def test_solar_system_body_count():
    """Explicit moons_per_planet drives total body count."""
    system = factory.create_scenario('solar_system', moons_per_planet=[1, 2])
    # 1 sun + 2 planets + (1+2) moons = 6 bodies
    assert len(system.positions()) == 6
    assert len(system.masses) == 6


def test_solar_system_no_moons():
    """Planets with no moons."""
    system = factory.create_scenario(
        'solar_system', moons_per_planet=[0, 0, 0]
    )
    # 1 sun + 3 planets + 0 moons = 4 bodies
    assert len(system.positions()) == 4


def test_solar_system_masses():
    """Sun mass = M^2, planet mass = M, moon mass = 1."""
    M = 50.0
    system = factory.create_scenario('solar_system', moons_per_planet=[1], M=M)
    masses = to_numpy_host(system.masses)
    sun_mass, planet_mass, moon_mass = masses[0], masses[1], masses[2]
    assert np.isclose(sun_mass, M ** 2)
    assert np.isclose(planet_mass, M)
    assert np.isclose(moon_mass, 1.0)


def test_solar_system_planet_radii():
    """Planets are spaced by ratio k: planet i is at radius k**i from sun."""
    k = 1.5
    system = factory.create_scenario(
        'solar_system', moons_per_planet=[0, 0, 0], k=k
    )
    positions = to_numpy_host(system.positions())
    # Sun at index 0, planets at indices 1, 2, 3
    for i in range(3):
        planet_x = positions[1 + i][0]
        expected_r = k ** i
        assert np.isclose(planet_x, expected_r), (
            f"Planet {i}: expected x={expected_r}, got x={planet_x}"
        )


def test_solar_system_orbital_velocities():
    """Planets have circular-orbit speed around the sun (G=1)."""
    M = 100.0
    k = 1.5
    system = factory.create_scenario(
        'solar_system', moons_per_planet=[0, 0], M=M, k=k
    )
    velocities = to_numpy_host(system.velocities())
    sun_mass = M ** 2
    for i in range(2):
        r_planet = k ** i
        expected_v = np.sqrt(sun_mass / r_planet)
        planet_vy = velocities[1 + i][1]
        assert np.isclose(planet_vy, expected_v), (
            f"Planet {i}: expected vy={expected_v}, got vy={planet_vy}"
        )


def test_solar_system_moon_velocities():
    """Moon orbital speed is added to the parent planet's speed."""
    M = 100.0
    k = 1.2
    h = 0.1
    system = factory.create_scenario(
        'solar_system', moons_per_planet=[1], M=M, k=k, h=h
    )
    velocities = to_numpy_host(system.velocities())
    sun_mass = M ** 2
    planet_mass = M
    r_planet = k ** 0  # = 1.0
    v_planet = np.sqrt(sun_mass / r_planet)
    r_moon = h * (k ** 0)  # = h
    v_moon_orbital = np.sqrt(planet_mass / r_moon)
    expected_moon_vy = v_planet + v_moon_orbital
    # Indices: 0=sun, 1=planet, 2=moon
    moon_vy = velocities[2][1]
    assert np.isclose(moon_vy, expected_moon_vy)


def test_solar_system_invalid_M():
    """M <= 1 should raise ValueError."""
    with pytest.raises(ValueError, match="M must be > 1"):
        factory.create_scenario('solar_system', M=1.0)


def test_solar_system_invalid_k():
    """k <= 1 should raise ValueError."""
    with pytest.raises(ValueError, match="k must be > 1"):
        factory.create_scenario('solar_system', k=0.5)


def test_solar_system_invalid_h():
    """h <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="h must be > 0"):
        factory.create_scenario('solar_system', h=0.0)


def test_solar_system_negative_moon_count():
    """Negative moon counts should raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        factory.create_scenario(
            'solar_system', moons_per_planet=[1, -1]
        )


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_same_seed_produces_same_scatter(engine):
    """Same seed should produce identical scatter scenarios across engines."""
    f1 = ScenarioFactory(engine=engine, seed=42)
    f2 = ScenarioFactory(engine=engine, seed=42)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    np = s1.positions().__array_namespace__()
    assert np.allclose(s1.positions(), s2.positions())
    assert np.allclose(s1.velocities(), s2.velocities())
    assert np.allclose(s1.masses, s2.masses)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_different_seeds_produce_different_scatter(engine):
    """Different seeds should produce different scatter."""
    f1 = ScenarioFactory(engine=engine, seed=1)
    f2 = ScenarioFactory(engine=engine, seed=2)
    s1 = f1.create_scenario('scatter', n_bodies=5)
    s2 = f2.create_scenario('scatter', n_bodies=5)
    np = s1.positions().__array_namespace__()
    assert not np.allclose(s1.positions(), s2.positions())
    assert not np.allclose(s1.velocities(), s2.velocities())
    assert not np.allclose(s1.masses, s2.masses)


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_engine_consistent_seed_matches_numpy(engine):
    """With engine_consistent_seed=True, same seed yields identical scatter
    as the numpy engine regardless of which engine is requested."""
    f_numpy = ScenarioFactory(engine='numpy', seed=42)
    f_other = ScenarioFactory(engine=engine, seed=42,
                              via_numpy=True)
    s_numpy = f_numpy.create_scenario('scatter', n_bodies=5)
    s_other = f_other.create_scenario('scatter', n_bodies=5)
    assert np.allclose(
        to_numpy_host(s_numpy.positions()),
        to_numpy_host(s_other.positions()),
    )
    assert np.allclose(
        to_numpy_host(s_numpy.velocities()),
        to_numpy_host(s_other.velocities()),
    )
    assert np.allclose(
        to_numpy_host(s_numpy.masses),
        to_numpy_host(s_other.masses),
    )
