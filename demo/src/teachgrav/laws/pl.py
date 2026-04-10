from scipy.optimize import minimize
from .laws import Model
from ..system import to_shaped, restack_va
from ..array_abstraction import to_numpy_host

import logging
logger = logging.getLogger("Teachgrav")

# Fit a power-law model to the data
# We will use the same training data as the GP model, but instead of fitting
# a GP, we will fit a power-law model of the form:
# a = k * r^n
# where a is the acceleration, r is the distance, and k and n are the
# parameters to be learned.


class PLModel(Model):
    def __init__(self, factory, G=0.5, power=3.0, **kwargs):
        self.factory = factory
        self.G = G
        self.power = power

    def train(self, N_sys, **kwargs):
        """Train a model on random scatters for a given set of args."""
        # Placeholder implementation, replace with actual GP training code
        ICs, accelerations, masses, immobile = \
            self.factory.create_training_data(N_sys, **kwargs)
        target_acc = to_numpy_host(accelerations)

        logger.info("Training Power Law model...")

        def model(ICs, k, n):
            self.G = k
            self.power = n
            res = self.flat_law(ICs, masses=masses, immobile=immobile)
            vector_results = to_shaped(res, N_sys, len(masses))
            accelerations = vector_results[:, 1, :, :]
            flat_accelerations = accelerations.reshape((N_sys, -1))
            return flat_accelerations

        def objective(params):
            k, n = params
            pred = model(ICs, k, n)
            pred = to_numpy_host(pred)
            delta = pred - target_acc
            return float(to_numpy_host(delta ** 2).sum())

        res = minimize(
            objective, x0=[
                self.G, self.power], bounds=[
                (-5.0, 5.0), (-5.0, 5.0)])
        logger.info(f"Optimization result: {res}")
        logger.debug(f"Optimizer message: {res.message}")
        pars = res.x
        self.G = float(pars[0])
        self.power = float(pars[1])

        logger.info(f"Trained Power Law model with parameters:"
                    f"{self.G}, {self.power}")

    def flat_law(self, data, masses, immobile):
        """Compute the derivatives of the state."""
        # Incoming data shape 2D, size (C, 2 N D)
        data_flat = self.add_vectorising_dimension_if_needed(data)
        num_vec = data_flat.shape[0]
        data = to_shaped(data_flat, num_vec, num_bodies=len(immobile))
        G = data.__array_namespace__().array(self.G)
        power = data.__array_namespace__().array(self.power)
        dpositions = data[:, 1, :, :]  # Derivative of position is velocity
        # Get the array namespace (e.g., numpy or jax.numpy)
        np = data.__array_namespace__()
        # Each body experiences a gravitational force from
        # every other body, leading to acceleration
        # So we have an N*N*2 matrix of pairwise position differences
        # ... and thus an N*N*2 matrix of pairwise accelerations
        # Which we sum over the second axis to get the total acceleration
        # on each body
        positions = data[:, 0, :, :]  # shape (C, N, D)
        # Pairwise position differences: shape (C, N, N, D)
        displacements = positions[:, :, np.newaxis, :] - \
            positions[:, np.newaxis, :, :]
        distances = np.linalg.norm(displacements, axis=-1, keepdims=True)
        safe_distances = np.where(distances == 0, 1.0, distances)
        # Avoid division by zero, but will make no contribution
        # since we will zero out self-interactions next
        # Pairwise accelerations due to gravity
        accelerations = -1.0 * G * \
            masses[np.newaxis, np.newaxis, :, np.newaxis] * \
            displacements / (safe_distances ** (power + 1))
        # Clear self-interactions
        accelerations = np.where(distances > 0, accelerations, 0.0)
        # Sum accelerations from all other bodies
        dvelocities = np.sum(accelerations, axis=2)
        # logger.debug("Total Accelerations:\n%s", dvelocities)
        delta = restack_va(dpositions, dvelocities)  # shape (C, 2, N, D)
        # Mask out the derivatives for immobile bodies
        # 1 where mobile, 0 where immobile
        mask = (~immobile).astype(delta.dtype)
        mask = mask[np.newaxis, np.newaxis, :, np.newaxis]
        delta = delta * mask  # Zero out derivatives for immobile bodies

        # Shape (2 (pos, vel), N, D (x y z), )
        # Output shape: (C 2 N D)
        return delta.flatten()
