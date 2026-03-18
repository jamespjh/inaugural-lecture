# Scenario definitions for gravitational simulations

# Choices: - Moon orbiting Earth
#          - A random scatter of bodies with random initial velocities

from typing import Any
from .system import System
import numpy as np


def create_scenario(name: str, **kwargs: Any) -> System:
    """Return a scenario system by name."""
    dispatch = {
        "moon": init_moon_orbiting_earth,
        "scatter": init_random_scatter,
    }
    try:
        return dispatch[name](**kwargs)
    except KeyError as exc:
        valid = ", ".join(sorted(dispatch))
        raise ValueError(
            f"Unknown scenario '{name}'. Valid scenarios: {valid}") from exc


def init_moon_orbiting_earth() -> System:
    """A simple two-body system with the Moon orbiting the Earth."""
    earth_mass = 1.0
    moon_mass = 0.0123
    earth_position = [0.0, 0.0]
    moon_position = [1.0, 0.0]
    earth_velocity = [0.0, 0.0]
    # Placeholder velocity for circular orbit -
    # determine reduced mass, earth velocity, and moon velocity
    moon_velocity = [0.0, 1.0]

    return System(
        positions=[earth_position, moon_position],
        velocities=[earth_velocity, moon_velocity],
        masses=[earth_mass, moon_mass],
    )


def init_random_scatter(
    n_bodies: int = 20,
    seed: int | None = None,
    space_radius: float = 1.0,
    max_speed: float = 1.0,
    min_mass: float = 0.01,
    max_mass: float = 1.0,
) -> System:
    """Randomly scattered bodies with random velocities in 2D."""

    rng_np = np.random.default_rng(seed)
    masses = rng_np.uniform(min_mass, max_mass, n_bodies)
    positions = rng_np.uniform(-space_radius, space_radius, (n_bodies, 2))
    velocities = rng_np.uniform(-max_speed, max_speed, (n_bodies, 2))

    return System(
        positions=positions,
        velocities=velocities,
        masses=masses,
    )
