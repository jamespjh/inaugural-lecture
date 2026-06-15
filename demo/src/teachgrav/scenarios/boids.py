from .base import Scenario
from ..system import System


class BoidsScenario(Scenario):
    """
    Flock of boids for use with BoidsLawModel.
    """

    def create(self, n_boids: int = 50, dimensions: int = 2) -> System:
        """Initialise a flock of boids.

        Args:
            n_boids: number of boids (default 50).
            dimensions: number of spatial dimensions, 2 or 3 (default 2).
        """
        if dimensions not in (2, 3):
            raise ValueError(
                f"dimensions must be 2 or 3, got {dimensions}")
        pos_components = [
            self.engine.random_array((n_boids,), -450.0, -350.0),
            self.engine.random_array((n_boids,), 300.0, 600.0),
        ]
        vel_components = [
            self.engine.random_array((n_boids,), 0.0, 50.0),
            self.engine.random_array((n_boids,), -5.0, 5.0),
        ]
        if dimensions == 3:
            pos_components.append(
                self.engine.random_array((n_boids,), -50.0, 50.0))
            vel_components.append(
                self.engine.random_array((n_boids,), -5.0, 5.0))
        positions = self.engine.array(pos_components).T  # (n_boids, D)
        velocities = self.engine.array(vel_components).T  # (n_boids, D)
        masses = self.engine.array([1.0] * n_boids)

        return System(
            self.engine.array([positions, velocities]),
            masses=masses,
        )
