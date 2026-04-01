from .system import System
import jax.numpy as np


def gp_law(system: System, pars) -> np.ndarray:
    """Compute the derivatives of the state using a learned GP model."""
    # Placeholder implementation, replace with actual GP prediction
    # For now, just return the same as the physics-based law
    raise NotImplementedError("GP law is not implemented yet")
