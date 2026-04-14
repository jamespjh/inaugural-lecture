from .base import Scenario
from ..system import System


class SingleScenario(Scenario):
    """
    A single body moving at constant velocity across the map.
    Unlike scatter scenarios, this does NOT normalize for centre of mass.
    This produces pure straight-line motion when used with constant law.
    """

    def create(
        self,
        position: list = None,
        velocity: list = None,
        mass: float = 1.0,
    ) -> System:
        """
        Args:
            position: initial position [x, y], default [0.0, 0.0]
            velocity: initial velocity [vx, vy], default [1.0, 1.0]
            mass: mass of the body, default 1.0
        """
        if position is None:
            position = [0.0, 0.0]
        if velocity is None:
            velocity = [1.0, 1.0]

        return System(
            self.engine.array([[position], [velocity]]),
            masses=self.engine.array([mass]),
        )
