from .base import Scenario
from ..system import System


class SunScenario(Scenario):
    """A simple two-body system with the Earth orbiting the Sun."""

    def create(self) -> System:
        sun_mass = 1.0
        earth_mass = 0.01
        sun_position = [0.0, 0.0]
        earth_position = [1.0, 0.0]
        sun_velocity = [0.0, 0.0]
        # Initial velocity for a circular orbit at distance 1.0 with G=1.0 and
        # M=1.0
        earth_velocity = [0.0, 1.0]

        return System(
            self.engine.array([[sun_position, earth_position],
                               [sun_velocity, earth_velocity]]),
            masses=self.engine.array([sun_mass, earth_mass]),
            # Sun is immobile, i.e. fixed at the origin
            immobile=self.engine.array([True, False]),
        )
