import logging
from jax import jit
import jax.numpy as np
from functools import partial
logger = logging.getLogger("Teachgrav")



def law(system) -> np.ndarray:
    """Compute the derivatives of the state."""
    return flat_law(system.data.flatten(), system.masses,
                    system.immobile).reshape(system.data.shape)

def flat_law(data_flat, masses, immobile) -> np.ndarray:
    """Compute the derivatives of the state."""
    # Placeholder implementation, replace with actual physics
    data = data_flat.reshape((2, len(masses), -1))  # shape (2, N, D)
    dpositions = data[1, :, :]  # Derivative of position is velocity

    # Each body experiences a gravitational force from
    # every other body, leading to acceleration
    # So we have an N*N*2 matrix of pairwise position differences
    # ... and thus an N*N*2 matrix of pairwise accelerations
    # Which we sum over the second axis to get the total acceleration on each
    # body
    G = 1.0  # Gravitational constant (arbitrary units)
    positions = data[0, :, :]  # shape (N, D)

    # Pairwise position differences: shape (N, N, D)
    displacements = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    #logger.debug("Positions:\n%s", positions)
    distances = np.linalg.norm(displacements, axis=2, keepdims=True)

    # Avoid division by zero, also avoids self-interaction
    # Jax requires immutable arrays, so we use .at[].set()
    distances = np.where(distances == 0, np.inf, distances)

    # Pairwise accelerations due to gravity
    accelerations = -1.0 * G * \
        masses[np.newaxis, :, np.newaxis] * displacements / (distances ** 3)
    # Sum accelerations from all other bodies
    dvelocities = np.sum(accelerations, axis=1)
    #logger.debug("Total Accelerations:\n%s", dvelocities)
    delta = np.stack([dpositions, dvelocities])  # shape (2, N, D)
    # Mask out the derivatives for immobile bodies
    mask = (~immobile).astype(delta.dtype)   # 1 where mobile, 0 where immobile
    mask = mask[None, :, None]
    delta = delta * mask  # Zero out derivatives for immobile bodies

    # Shape (2 (pos, vel), N, D (x y z), )
    return delta.flatten()
