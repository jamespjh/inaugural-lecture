import logging
from .laws import Model
from ..system import to_shaped, restack_va

logger = logging.getLogger("Teachgrav")


class BoidsLawModel(Model):
    """
    Boids flocking model implementing three rules:

    1. Cohesion: fly towards the centre of the flock.
    2. Separation: avoid getting too close to nearby boids.
    3. Alignment: match velocity with nearby boids.

    Based on the algorithm by Craig Reynolds (1987), using the numpy
    implementation from https://github.com/jamespjh/bad-boids/tree/better_boids
    """

    def __init__(
        self,
        factory=None,
        flock_attraction=0.01,
        avoidance_radius=1000.0,
        avoidance_strength=5.0,
        formation_flying_radius=1000.0,
        speed_matching_strength=0.001,
        **kwargs
    ):
        self.factory = factory
        self.flock_attraction = flock_attraction
        self.avoidance_radius = avoidance_radius
        self.avoidance_strength = avoidance_strength
        self.formation_flying_radius = formation_flying_radius
        self.speed_matching_strength = speed_matching_strength

    def flat_law(self, data, masses, immobile):
        """
        Compute boids derivatives.

        Position derivative = velocity.
        Velocity derivative = sum of cohesion, separation, and alignment
        accelerations.

        Args:
            data: flat array of shape (C, 2*N*D) where C is batch count,
                  N is number of boids, D is number of dimensions.
            masses: per-body mass array of shape (N,) (not used by boids
                    rules but required by the Model interface).
            immobile: boolean mask of shape (N,) for fixed bodies.

        Returns:
            Flat array of derivatives with the same shape as data.
        """
        data_flat = self.add_vectorising_dimension_if_needed(data)
        num_vec = data_flat.shape[0]
        np = data_flat.__array_namespace__()

        data_shaped = to_shaped(data_flat, num_vec, len(masses))

        positions = data_shaped[:, 0, :, :]  # (C, N, D)
        velocities = data_shaped[:, 1, :, :]  # (C, N, D)

        # Derivative of position is velocity
        dpositions = velocities

        # Pairwise displacements from body i to body j: pos_j - pos_i
        # displacements[c, i, j, d] = positions[c, j, d] - positions[c, i, d]
        # Shape: (C, N, N, D)
        displacements = (
            positions[:, np.newaxis, :, :] - positions[:, :, np.newaxis, :]
        )

        # Squared distances for threshold comparisons: shape (C, N, N, 1)
        distances_sq = (displacements**2).sum(axis=-1, keepdims=True)

        # --- Cohesion: fly toward the centre of the flock ---
        # Each boid is attracted toward every other boid.
        # Sum of (pos_j - pos_i) over all j, scaled by flock_attraction.
        cohesion = displacements.sum(axis=2) * self.flock_attraction

        # --- Separation: avoid nearby boids ---
        # When boid j is within avoidance_radius of boid i, boid i is
        # repelled (subtract displacement, i.e. move away).
        avoidance_mask = distances_sq < (
            self.avoidance_radius**2
        )  # (C, N, N, 1)
        # Repulsion strength law: 1/distance, until avoidance_radius where it
        # becomes zero.
        repulsion_strengths = self.avoidance_strength / (
            np.sqrt(distances_sq) + 1e-10
        )
        separation = -(
            displacements * avoidance_mask * repulsion_strengths
        ).sum(axis=2)

        # --- Alignment: match velocity with nearby boids ---
        # Velocity differences: vel_j - vel_i, shape (C, N, N, D)
        vel_diffs = (
            velocities[:, np.newaxis, :, :] - velocities[:, :, np.newaxis, :]
        )
        formation_mask = distances_sq < (
            self.formation_flying_radius**2
        )  # (C, N, N, 1)
        alignment = (vel_diffs * formation_mask).sum(
            axis=2
        ) * self.speed_matching_strength

        dvelocities = cohesion + separation + alignment

        delta = restack_va(dpositions, dvelocities)  # (C, 2, N, D)

        # Zero out derivatives for immobile bodies
        mask = (~immobile).astype(delta.dtype)
        mask = mask[np.newaxis, np.newaxis, :, np.newaxis]
        delta = delta * mask

        return delta.flatten()
