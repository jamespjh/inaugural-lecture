from .base import Scenario
from ..system import System


class ScatterScenario(Scenario):
    """Randomly scattered bodies with random velocities."""

    def create(
        self,
        n_bodies: int = 4,
        randomise_count: bool = False,
        space_radius: float = 1.0,
        max_speed: float = 1.0,
        min_mass: float = 0.1,
        max_mass: float = 10.0,
        dimensions: int = 2,
        fixed_masses=None,
    ) -> System:
        if randomise_count:
            n_bodies = self.engine.random_array(
                (1,), 2, n_bodies + 1
            ).item()  # type: ignore
        if fixed_masses is not None:
            if len(fixed_masses) != n_bodies:
                raise ValueError(
                    f"Length of fixed_masses ({len(fixed_masses)}) must "
                    f"match n_bodies ({n_bodies})."
                )
            masses = self.engine.array(fixed_masses)
        else:
            masses = self.engine.random_array((n_bodies,), min_mass, max_mass)
        positions = self.engine.random_array(
            (n_bodies, dimensions), -space_radius, space_radius
        )
        velocities = self.engine.random_array(
            (n_bodies, dimensions), -max_speed, max_speed
        )
        # Reset the velocities so there is zero net momentum
        momenta = masses[:, None] * velocities
        total_momentum = momenta.sum(axis=0)
        velocities -= total_momentum / masses.sum()
        # Reset the positions so the center of mass is at the origin
        com = (masses[:, None] * positions).sum(axis=0)
        positions -= com / masses.sum()

        return System(
            self.engine.array([positions, velocities]),
            masses=self.engine.array(masses),
        )
