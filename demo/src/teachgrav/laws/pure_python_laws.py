import numba
import math


def _python_gravity_flat_law(G, data, masses, immobile):
    """Pedagogically clear implementation of Newton's law of gravitation.

    Uses explicit nested for loops — no numpy array operations.

    Newton's law of gravitation:
        The gravitational acceleration on body i due to body j is

            a_ij = G * M_j / r_ij^2  *  (unit vector from i toward j)
                = G * M_j * (pos_j - pos_i) / |pos_j - pos_i|^3

        The total acceleration on body i is the sum of a_ij over all j ≠ i.

    Args:
        data:     flat Python list of length 2*N*D — positions first,
                    then velocities, both in body-major, dimension-minor
                    order.
        masses:   Python list of N body masses.
        immobile: Python list of N booleans; True means the body is fixed.

    Returns:
        Flat Python list of length 2*N*D with the time-derivatives of
        the state: [d_positions, d_velocities] in the same layout as
        *data*.
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
            squares = [displacement[d] ** 2 for d in range(D)]
            distance = math.sqrt(sum(squares))

            # Skip coincident bodies to avoid division by zero.
            if distance == 0.0:
                continue

            # Add body j's contribution to the acceleration of body i:
            #   a += G * M_j * displacement / distance^3
            for d in range(D):
                d_velocities[i][d] += (
                    G * masses[j] * displacement[d] / distance ** 3
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


_numba_gravity_flat_law = numba.njit(_python_gravity_flat_law)