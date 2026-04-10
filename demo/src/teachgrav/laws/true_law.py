import math
import logging
from .pl import PLModel

logger = logging.getLogger("Teachgrav")

# Engines that work with plain Python data structures (lists) rather than
# numpy arrays, and therefore need a numpy-free gravity implementation.
PYTHON_LIKE_ENGINES = {'python', 'numba'}


class TrueLawModel(PLModel):
    def __init__(self, factory=None):
        super().__init__(factory=factory, G=1.0, power=2.0)

    def _uses_python_engine(self):
        """Return True if the factory uses a Python-like engine."""
        if self.factory is None:
            return False
        return self.factory.engine.engine in PYTHON_LIKE_ENGINES

    def flat_law(self, data, masses, immobile):
        """Compute gravitational derivatives from a flat state vector.

        For python and numba engines the data lives in plain Python lists,
        so we use an explicit nested-for-loop implementation that requires no
        numpy array operations.  For all other engines we delegate to the
        vectorised PLModel implementation.
        """
        if self._uses_python_engine():
            return self._python_gravity_flat_law(data, masses, immobile)
        return super().flat_law(data, masses, immobile)

    def _python_gravity_flat_law(self, data, masses, immobile):
        """Pedagogically clear implementation of Newton's law of gravitation.

        Uses explicit nested for loops — no numpy array operations.

        Newton's law of gravitation:
          The gravitational acceleration on body i due to body j is

              a_ij = G * M_j / r_ij^2  *  (unit vector from i toward j)
                   = G * M_j * (pos_j - pos_i) / |pos_j - pos_i|^3

          The total acceleration on body i is the sum of a_ij over all j ≠ i.

        Args:
            data:     flat sequence of length 2*N*D — positions first, then
                      velocities, both in body-major, dimension-minor order.
            masses:   sequence of N body masses.
            immobile: sequence of N booleans; True means the body is fixed.

        Returns:
            Flat list of length 2*N*D with the time-derivatives of the state:
            [d_positions, d_velocities] in the same layout as *data*.
        """
        N = len(masses)            # number of bodies
        D = len(data) // (2 * N)  # number of spatial dimensions (e.g. 2 or 3)

        # ------------------------------------------------------------------ #
        # Parse positions and velocities from the flat data sequence.
        # Layout: [pos_body0_dim0, pos_body0_dim1, …, pos_bodyN_dimD-1,
        #          vel_body0_dim0, vel_body0_dim1, …, vel_bodyN_dimD-1]
        # ------------------------------------------------------------------ #
        positions = [
            [data[i * D + d] for d in range(D)]
            for i in range(N)
        ]
        velocities = [
            [data[N * D + i * D + d] for d in range(D)]
            for i in range(N)
        ]

        # The time-derivative of position is simply the velocity (kinematics).
        d_positions = [list(vel) for vel in velocities]

        # Initialise gravitational accelerations to zero for every body.
        d_velocities = [[0.0] * D for _ in range(N)]

        # ------------------------------------------------------------------ #
        # Compute the gravitational acceleration on each body.
        # ------------------------------------------------------------------ #
        for i in range(N):      # body i is the one being accelerated
            for j in range(N):  # body j exerts the gravitational force
                if i == j:
                    continue    # a body does not attract itself

                # Vector pointing from body i toward body j.
                displacement = [
                    positions[j][d] - positions[i][d]
                    for d in range(D)
                ]

                # Euclidean distance between the two bodies.
                distance = math.sqrt(
                    sum(displacement[d] ** 2 for d in range(D))
                )

                # Skip coincident bodies to avoid division by zero.
                # (Physically, two distinct bodies at the same position
                # produce infinite force; we treat the contribution as zero,
                # consistent with the numpy implementation.)
                if distance == 0.0:
                    continue

                # Add body j's contribution to the acceleration of body i:
                #   a += G * M_j * displacement / distance^3
                for d in range(D):
                    d_velocities[i][d] += (
                        self.G * masses[j] * displacement[d] / distance ** 3
                    )

        # ------------------------------------------------------------------ #
        # Zero out derivatives for immobile bodies (e.g. a fixed star).
        # ------------------------------------------------------------------ #
        for i in range(N):
            if immobile[i]:
                for d in range(D):
                    d_positions[i][d] = 0.0
                    d_velocities[i][d] = 0.0

        # ------------------------------------------------------------------ #
        # Pack the result back into flat format: [d_positions, d_velocities].
        # ------------------------------------------------------------------ #
        result = []
        for i in range(N):
            result.extend(d_positions[i])
        for i in range(N):
            result.extend(d_velocities[i])

        return result
