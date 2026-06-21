from .base import Scenario
from ..system import System


class MoonScenario(Scenario):
    """A simple two-body system with the Moon orbiting the Earth."""

    def create(self) -> System:
        earth_mass = 1.0
        moon_mass = 0.01
        earth_position = [0.0, 0.0]
        moon_position = [1.0, 0.0]
        earth_velocity = [0.0, 0.0]
        moon_velocity = [0.0, 1.0]

        return System(
            self.engine.array(
                [
                    [earth_position, moon_position],
                    [earth_velocity, moon_velocity],
                ]
            ),
            masses=self.engine.array([earth_mass, moon_mass]),
        )
