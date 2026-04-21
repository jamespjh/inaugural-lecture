import math
from typing import List, Optional

from .base import Scenario
from ..system import System


class SolarSystemScenario(Scenario):
    """A solar system: planets orbit a central sun, each with optional moons.

    Physical setup (G=1):
    - Sun mass: M**2
    - Planet mass: M
    - Moon mass: 1
    - Planet i is at radius k**i from the sun (i = 0, 1, 2, …).
    - Moon j of a planet is at radius h * k**j from the planet centre
      (j = 0, 1, 2, …).
    - Orbital speeds are derived from circular-orbit mechanics:
        v = sqrt(M_central / r)
    - No bodies are immobile; the sun starts at the origin with zero velocity.
    """

    def create(
        self,
        moons_per_planet: Optional[List[int]] = None,
        M: float = 1000.0,
        k: float = 1.2,
        h: float = 0.1,
    ) -> System:
        """Create the solar-system initial state.

        Args:
            moons_per_planet: List whose *length* gives the number of planets
                and whose *elements* give the number of moons for each planet.
                All elements must be non-negative integers.
                Defaults to ``[0, 1, 4, 1]``.
            M: Planet mass; must be > 1. Sun mass = M**2, moon mass = 1.
            k: Orbital radius spacing ratio; must be > 1. Planet i is at
                radius k**i.  Consecutive moon radii also grow by this factor.
            h: Innermost moon orbital radius (relative to its planet);
                must be > 0.

        Raises:
            ValueError: If M <= 1, k <= 1, h <= 0, or any element of
                moons_per_planet is negative.
        """
        if moons_per_planet is None:
            moons_per_planet = [0, 1, 4, 1]

        if M <= 1:
            raise ValueError(
                f"M must be > 1 for valid circular orbits, got M={M}."
            )
        if k <= 1:
            raise ValueError(
                f"k must be > 1 to space planet radii outward, got k={k}."
            )
        if h <= 0:
            raise ValueError(
                f"h must be > 0 (innermost moon radius), got h={h}."
            )
        negative = [n for n in moons_per_planet if n < 0]
        if negative:
            raise ValueError(
                "All elements of moons_per_planet must be non-negative "
                f"integers; got negative values: {negative}."
            )

        sun_mass = M ** 2
        planet_mass = float(M)
        moon_mass = 1.0

        positions = []
        velocities = []
        masses = []

        # Sun at the origin, initially at rest.
        positions.append([0.0, 0.0])
        velocities.append([0.0, 0.0])
        masses.append(sun_mass)

        for i, n_moons in enumerate(moons_per_planet):
            # Planet i orbits the sun at radius k**i.
            r_planet = k ** i
            # Circular-orbit speed around the sun: v = sqrt(M_sun / r)
            v_planet = math.sqrt(sun_mass / r_planet)

            # Place the planet on the positive x-axis.
            positions.append([r_planet, 0.0])
            # Velocity is perpendicular (positive y-direction).
            velocities.append([0.0, v_planet])
            masses.append(planet_mass)

            # Moons for this planet.
            for j in range(n_moons):
                # Moon j orbits the planet at radius h * k**j.
                r_moon = h * (k ** j)
                # Circular-orbit speed around the planet:
                # v = sqrt(M_planet/r)
                v_moon_orbital = math.sqrt(planet_mass / r_moon)

                # Place the moon along the x-axis beyond the planet.
                positions.append([r_planet + r_moon, 0.0])
                # Absolute velocity = planet velocity + moon orbital velocity
                # (same perpendicular direction).
                velocities.append([0.0, v_planet + v_moon_orbital])
                masses.append(moon_mass)

        return System(
            self.engine.array([positions, velocities]),
            masses=self.engine.array(masses),
        )
