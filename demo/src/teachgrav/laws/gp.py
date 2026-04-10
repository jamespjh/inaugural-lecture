from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .laws import Model
from ..system import restack_va, to_shaped
from ..array_abstraction import to_numpy_host

import logging
logger = logging.getLogger("Teachgrav")


class GPModel(Model):
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.kernel = 1 * Matern(length_scale=1.0,
                                 length_scale_bounds=(1e-2, 1e2))
        self.gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=9)

    def normaliseX(self, X):
        """Normalise the data to have zero mean and unit variance."""
        self.X_mean = X.mean(axis=0)
        # Add small value to avoid division by zero
        self.X_std = X.std(axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std

    def denormaliseX(self, normed):
        """Denormalise the data using the original mean and std."""
        return normed * self.X_std + self.X_mean

    def renormaliseX(self, X):
        """Renormalise the data using the original mean and std."""
        return (X - self.X_mean) / self.X_std

    def normaliseY(self, Y):
        """Normalise the data to have zero mean and unit variance."""
        self.Y_mean = Y.mean(axis=0)
        # Add small value to avoid division by zero
        self.Y_std = Y.std(axis=0) + 1e-8
        return (Y - self.Y_mean) / self.Y_std

    def denormaliseY(self, normed):
        """Denormalise the data using the original mean and std."""
        return normed * self.Y_std + self.Y_mean

    def train(self, N_sys, **kwargs):
        """Train a GP model on random scatters for a given set of args."""
        # Placeholder implementation, replace with actual GP training code
        ICs, accelerations, masses, immobile = \
            self.factory.create_training_data(N_sys, **kwargs)
        norm_y = self.normaliseY(accelerations)
        norm_ICs = self.normaliseX(ICs)
        norm_y = to_numpy_host(norm_y)
        norm_ICs = to_numpy_host(norm_ICs)

        logger.info("Training GP model...")
        self.gaussian_process.fit(norm_ICs, norm_y)
        logger.info(f"Trained GP model with kernel: "
                    f"{self.gaussian_process.kernel_}")

    def flat_law(self, data, masses, immobile):
        """Compute the derivatives of the state using a learned GP model."""
        # Might be given multiple ICs in a batch, shape (C, 2 N D)
        ICs = self.add_vectorising_dimension_if_needed(data)
        pred_in = to_numpy_host(self.renormaliseX(ICs))
        means = self.gaussian_process.predict(pred_in)
        acc = self.factory.engine.array(means)  # Accelerations
        acc = self.denormaliseY(acc)
        velocities = to_shaped(
            ICs, ICs.shape[0], len(masses))[
            :, 1, :, :].reshape(
            acc.shape)  # Velocities
        # Shape N_sys, 2, N_bodies * D
        derivatives = restack_va(velocities, acc)
        return derivatives.flatten()
