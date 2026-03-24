import logging

import mlx.core as mx
logger = logging.getLogger("Teachgrav")


def law(system) -> mx.array:
    """Compute the derivatives of the state."""
    return flat_law(system.data.flatten(), system.masses,
                    system.immobile).reshape(system.data.shape)


def flat_law(data_flat, masses, immobile) -> mx.array:
    """Compute the derivatives of the state."""
    data_flat = mx.asarray(data_flat)
    # Placeholder implementation, replace with actual physics
    data = data_flat.reshape((2, len(masses), -1))  # shape (2, N, D)
    delta = mx.zeros_like(data)  # 2, N, D
    delta[0, :, :] = data[1, :, :]  # Derivative of position is velocity

    # Each body experiences a gravitational force from
    # every other body, leading to acceleration
    # So we have an N*N*2 matrix of pairwise position differences
    # ... and thus an N*N*2 matrix of pairwise accelerations
    # Which we sum over the second axis to get the total acceleration on each
    # body
    G = 1.0  # Gravitational constant (arbitrary units)
    positions = data[0, :, :]  # shape (N, D)

    # Pairwise position differences: shape (N, N, D)
    displacements = positions[:, mx.newaxis, :] - positions[mx.newaxis, :, :]

    logger.debug("Positions:\n%s", positions)
    distances = mx.linalg.norm(displacements, axis=2, keepdims=True)

    # Avoid division by zero, also avoids self-interaction
    distances[distances == 0] = mx.inf

    # Pairwise accelerations due to gravity
    accelerations = -1.0 * G * \
        masses[mx.newaxis, :, mx.newaxis] * displacements / (distances ** 3)
    # Sum accelerations from all other bodies
    delta[1, :, :] = mx.sum(accelerations, axis=1)
    logger.debug("Total Accelerations:\n%s", delta[1, :, :])
    # Mask out the derivatives for immobile bodies
    mask = (~immobile).astype(delta.dtype)   # 1 where mobile, 0 where immobile
    mask = mask[None, :, None]
    delta = delta * mask  # Zero out derivatives for immobile bodies

    # Shape (2 (pos, vel), N, D (x y z), )
    return delta.flatten()
