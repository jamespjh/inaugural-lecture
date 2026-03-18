import logging

from .system import System
import numpy as np
logger = logging.getLogger("Teachgrav")


def law(system: System) -> np.ndarray:
    """Compute the derivatives of the state."""
    # Placeholder implementation, replace with actual physics
    d_positions = system.velocities

    # Each body experiences a gravitational force from
    # every other body, leading to acceleration
    # So we have an N*N*2 matrix of pairwise position differences
    # ... and thus an N*N*2 matrix of pairwise accelerations
    # Which we sum over the second axis to get the total acceleration on each
    # body
    G = 1.0  # Gravitational constant (arbitrary units)
    positions = system.positions
    masses = system.masses

    # Pairwise position differences: shape (N, N, 2)
    displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    logger.debug(f"Positions:\n{positions}")
    distances = np.linalg.norm(displacements, axis=2, keepdims=True)

    # Avoid division by zero, also avoids self-interaction
    distances[distances == 0] = np.inf

    # Pairwise accelerations due to gravity
    accelerations = -1.0 * G * \
        masses[np.newaxis, :, np.newaxis] * displacements / (distances ** 3)
    logger.debug(f"Accelerations:\n{accelerations}")
    # Sum accelerations from all other bodies
    d_velocities = np.sum(accelerations, axis=1)
    logger.debug(f"Total Accelerations:\n{d_velocities}")
    # Mask out the derivatives for immobile bodies
    d_positions[system.immobile] = 0
    d_velocities[system.immobile] = 0

    logger.debug(f"Total Accelerations after masking:\n{d_velocities}")
    logger.debug(f"Velocities:\n{d_positions}")
    # Shape (2 (pos, vel), N, 2 (x and y), )
    return np.stack([d_positions, d_velocities])
