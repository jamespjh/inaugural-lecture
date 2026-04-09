import logging
from .laws import Model

logger = logging.getLogger("Teachgrav")


class ConstantLawModel(Model):
    """
    Model where bodies move at constant velocity.
    Acceleration (velocity derivative) is always zero.
    """
    def __init__(self, factory=None, **kwargs):
        self.factory = factory

    def flat_law(self, data, masses, immobile):
        """
        Compute derivatives for constant velocity motion.
        Position derivative = velocity (unchanged).
        Velocity derivative = 0 (no acceleration).
        """
        # data shape: (C, 2*N*D) where C is batch count (1 for single IC)
        # When reshaped: (C, 2, N, D) where 2 represents [position, velocity]
        data_flat = self.add_vectorising_dimension_if_needed(data)
        num_vec = data_flat.shape[0]
        
        # Get array namespace (numpy or jax.numpy)
        np = data_flat.__array_namespace__()
        
        # Reshape to (C, 2, N, D)
        from ..system import to_shaped
        data_shaped = to_shaped(data_flat, num_vec, len(masses))
        
        # Extract velocity component
        velocities = data_shaped[:, 1, :, :]  # (C, N, D)
        
        # Derivatives: dpos/dt = velocity, dvel/dt = 0
        dpositions = velocities
        dvelocities = np.zeros_like(velocities)
        
        # Restack into (C, 2, N, D) format
        from ..system import restack_va
        delta = restack_va(dpositions, dvelocities)
        
        # Apply immobile mask (immobile bodies have zero derivatives)
        mask = (~immobile).astype(delta.dtype)
        mask = mask[np.newaxis, np.newaxis, :, np.newaxis]
        delta = delta * mask
        
        return delta.flatten()
