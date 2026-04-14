from .base import Scenario
from ..system import System


class BoidsScenario(Scenario):
    """
    Flock of boids for use with BoidsLawModel.

    Starting positions and velocities match the original Reynolds-style
    boids initialisation from bad-boids/better_boids:
      x  in [-450, 50],  y  in [300, 600]
      xv in [0, 10],     yv in [-20, 20]
    """

    def create(self, n_boids: int = 50) -> System:
        """Initialise a flock of boids.

        Args:
            n_boids: number of boids (default 50).
        """
        positions = self.engine.array([
            self.engine.random_array((n_boids,), -450.0, 50.0),
            self.engine.random_array((n_boids,), 300.0, 600.0),
        ]).T  # shape (n_boids, 2)
        velocities = self.engine.array([
            self.engine.random_array((n_boids,), 0.0, 10.0),
            self.engine.random_array((n_boids,), -20.0, 20.0),
        ]).T  # shape (n_boids, 2)
        masses = self.engine.array([1.0] * n_boids)

        return System(
            self.engine.array([positions, velocities]),
            masses=masses,
        )
