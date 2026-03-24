# Scenario definitions for gravitational simulations

# Choices: - Moon orbiting Earth
#          - A random scatter of bodies with random initial velocities

import logging
from typing import Any
from .system import System
import mlx.core as mx
logger = logging.getLogger("Teachgrav")


def create_scenario(name: str, **kwargs: Any) -> System:
    """Return a scenario system by name."""
    dispatch = {
        "moon": init_moon_orbiting_earth,
        "sun": init_earth_orbiting_sun,
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
    moon_mass = 0.01
    earth_position = [0.0, 0.0]
    moon_position = [1.0, 0.0]
    earth_velocity = [0.0, 0.0]
    moon_velocity = [0.0, 1.0]

    return System(
        [[earth_position, moon_position],
         [earth_velocity, moon_velocity]],
        masses=[earth_mass, moon_mass],
    )


def init_earth_orbiting_sun() -> System:
    """A simple two-body system with the Earth orbiting the Sun."""
    sun_mass = 1.0
    earth_mass = 0.01
    sun_position = [0.0, 0.0]
    earth_position = [1.0, 0.0]
    sun_velocity = [0.0, 0.0]
    # Initial velocity for a circular orbit at distance 1.0 with G=1.0 and
    # M=1.0
    earth_velocity = [0.0, 1.0]

    return System(
        [[sun_position, earth_position],
         [sun_velocity, earth_velocity]],
        masses=[sun_mass, earth_mass],
        immobile=[True, False],  # Sun is immobile, i.e. fixed at the origin
    )


def init_random_scatter(
    n_bodies: int = 4,
    randomise_count: int = False,
    seed: int | None = None,
    space_radius: float = 1.0,
    max_speed: float = 1.0,
    min_mass: float = 0.1,
    max_mass: float = 10.0,
    dimensions: int = 2,
) -> System:
    """Randomly scattered bodies with random velocities."""

    # rng_np = mx.random.default_rng(seed)
    if randomise_count:
        n_bodies = mx.random.randint(low=2, high=n_bodies+1)
    masses = mx.random.uniform(min_mass, max_mass, (n_bodies,))
    positions = mx.random.uniform(-space_radius, space_radius,
                                  (n_bodies, dimensions))
    velocities = mx.random.uniform(-max_speed, max_speed,
                                   (n_bodies, dimensions))
    # Reset the velocities so there is zero net momentum
    momenta = masses[:, mx.newaxis] * velocities
    total_momentum = momenta.sum(axis=0)
    velocities -= total_momentum / masses.sum()
    # Reset the positions so the center of mass is at the origin
    com = (masses[:, mx.newaxis] * positions).sum(axis=0)
    positions -= com / masses.sum()

    logger.info(f"Initialized random scatter scenario with {n_bodies} bodies, "
                f"masses [{masses}, "
                f"positions {positions}, "
                f"velocities {velocities}, "
                f"seed={seed}")
    return System(
        [positions, velocities],
        masses=masses,
    )
