from .system import System
import numpy as np


def law(system: System) -> np.ndarray:
    """Compute the derivatives of the state."""
    # Placeholder implementation, replace with actual physics
    d_positions = system.velocities
    # No acceleration in this placeholder
    d_velocities = np.zeros_like(system.velocities)
    return np.concatenate([d_positions, d_velocities], axis=1)
