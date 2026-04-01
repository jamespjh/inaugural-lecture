import logging
logger = logging.getLogger("Teachgrav")

def law(system):
    """Compute the derivatives of the state."""
    return flat_law(system.data.flatten(), system.masses,
                    system.immobile).reshape(system.data.shape)

# Vectorised over C multiple initial conditions in a batch
def flat_law(data_flat, masses, immobile):
    """Compute the derivatives of the state."""
    # Incoming data shape 2D, size (C, 2 N D)
    if data_flat.ndim > 1:
        num_vec = data_flat.shape[0]
    else:
        num_vec = 1
    data = data_flat.reshape((num_vec, 2, len(masses), -1))  # shape (C, 2, N, D)
    dpositions = data[:, 1, :, :]  # Derivative of position is velocity
    np = data.__array_namespace__()  # Get the array namespace (e.g., numpy or jax.numpy)
    # Each body experiences a gravitational force from
    # every other body, leading to acceleration
    # So we have an N*N*2 matrix of pairwise position differences
    # ... and thus an N*N*2 matrix of pairwise accelerations
    # Which we sum over the second axis to get the total acceleration on each
    # body
    G = 1.0  # Gravitational constant (arbitrary units)
    positions = data[:, 0, :, :]  # shape (C, N, D)
    # Pairwise position differences: shape (C, N, N, D)
    displacements = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    distances = np.linalg.norm(displacements, axis=-1, keepdims=True)

    # Avoid division by zero, also avoids self-interaction
    # Jax requires immutable arrays, so we use .at[].set()
    distances = np.where(distances == 0, np.inf, distances)
    # Pairwise accelerations due to gravity
    accelerations = -1.0 * G * \
        masses[np.newaxis, np.newaxis, :, np.newaxis] * displacements / (distances ** 3)
    # Sum accelerations from all other bodies
    dvelocities = np.sum(accelerations, axis=2)
    #logger.debug("Total Accelerations:\n%s", dvelocities)
    delta = np.stack([dpositions, dvelocities], axis=1)  # shape (C, 2, N, D)
    # Mask out the derivatives for immobile bodies
    mask = (~immobile).astype(delta.dtype)   # 1 where mobile, 0 where immobile
    mask = mask[np.newaxis, np.newaxis, :, np.newaxis]
    delta = delta * mask  # Zero out derivatives for immobile bodies

    # Shape (2 (pos, vel), N, D (x y z), )
    # Output shape: (C 2 N D)
    return delta.flatten()
