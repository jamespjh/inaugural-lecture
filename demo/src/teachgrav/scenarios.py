# Scenario definitions for gravitational simulations

# Choices: - Moon orbiting Earth
#          - A random scatter of bodies with random initial velocities

import logging
from .system import System
from .array_abstraction import ArrayAbstraction
logger = logging.getLogger("Teachgrav")

class ScenarioFactory:
    def __init__(self, engine='numpy'):
        self.engine = ArrayAbstraction(engine)

    def create_scenario(self, name: str, **kwargs) -> System:
        """Return a scenario system by name."""
        dispatch = {
            "moon": self.init_moon_orbiting_earth,
            "sun": self.init_earth_orbiting_sun,
            "scatter": self.init_random_scatter,
        }
        try:
            return dispatch[name](**kwargs)
        except KeyError as exc:
            valid = ", ".join(sorted(dispatch))
            raise ValueError(
                f"Unknown scenario '{name}'. Valid scenarios: {valid}") from exc


    def init_moon_orbiting_earth(self) -> System:
        """A simple two-body system with the Moon orbiting the Earth."""
        earth_mass = 1.0
        moon_mass = 0.01
        earth_position = [0.0, 0.0]
        moon_position = [1.0, 0.0]
        earth_velocity = [0.0, 0.0]
        moon_velocity = [0.0, 1.0]

        return System(
            self.engine.np.array([[earth_position, moon_position],
                                  [earth_velocity, moon_velocity]]),
            masses=self.engine.np.array([earth_mass, moon_mass]),
        )


    def init_earth_orbiting_sun(self) -> System:
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
            self.engine.np.array([[sun_position, earth_position],
                                  [sun_velocity, earth_velocity]]),
            masses=self.engine.np.array([sun_mass, earth_mass]),
            immobile=self.engine.np.array([True, False]),  # Sun is immobile, i.e. fixed at the origin
        )


    def init_random_scatter(
        self,
        n_bodies: int = 4,
        randomise_count: int = False,
        space_radius: float = 1.0,
        max_speed: float = 1.0,
        min_mass: float = 0.1,
        max_mass: float = 10.0,
        dimensions: int = 2,
    ) -> System:
        """Randomly scattered bodies with random velocities."""

        if randomise_count:
            n_bodies = self.engine.random_array(1, 2, n_bodies + 1).item()
        masses = self.engine.random_array(n_bodies, min_mass, max_mass)
        positions = self.engine.random_array((n_bodies, dimensions), -space_radius,
                                    space_radius)
        velocities = self.engine.random_array((n_bodies, dimensions), -max_speed,
                                    max_speed)
        # Reset the velocities so there is zero net momentum
        momenta = masses[:, None] * velocities
        total_momentum = momenta.sum(axis=0)
        velocities -= total_momentum / masses.sum()
        # Reset the positions so the center of mass is at the origin
        com = (masses[:, None] * positions).sum(axis=0)
        positions -= com / masses.sum()

        logger.info(f"Initialized random scatter scenario with {n_bodies} bodies, "
                    f"masses [{masses}, "
                    f"positions {positions}, "
                    f"velocities {velocities}, ")
        return System(
            self.engine.np.array([positions, velocities]),
            masses=self.engine.np.array(masses),
        )
