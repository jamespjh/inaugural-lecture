from .system import System
from .laws import flat_law
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import logging
logger = logging.getLogger("Teachgrav")


class GPModel:
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))    
        self.gaussian_process = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=9)
     
    def normaliseX(self, X):
        """Normalise the data to have zero mean and unit variance."""
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8  # Add small value to avoid division by zero
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
        self.Y_std = Y.std(axis=0) + 1e-8  # Add small value to avoid division by zero
        return (Y - self.Y_mean) / self.Y_std

    def denormaliseY(self, normed):
        """Denormalise the data using the original mean and std."""
        return normed * self.Y_std + self.Y_mean

    def train(self, N_sys, **kwargs):
        """Train a GP model on random scatters for a given set of args."""
        # Placeholder implementation, replace with actual GP training code
        scenarios = [self.factory.create_scenario('scatter', **kwargs)
                     for _ in range(N_sys)]

        ICs = self.factory.engine.np.array([system.data.flatten()
                                           for system in scenarios])
        flatICs = ICs.reshape((N_sys, -1))
        masses = scenarios[0].masses
        immobile = scenarios[0].immobile
        # We will give it for free, that the derivatives of the position
        # are the velocities
        # We are only asking it to learn the accelerations
        # As a function of the positions and velocities
        # At fixed masses and immobilities
        results = flat_law(flatICs, masses, immobile)

        vector_results = results.reshape((N_sys, 2, len(masses), -1))
        accelerations = vector_results[:, 1, :, :]
        flat_accelerations = accelerations.reshape((N_sys, -1))
        norm_y = self.normaliseY(flat_accelerations)
        norm_ICs = self.normaliseX(flatICs)

        logger.info("Training GP model...")
        self.gaussian_process.fit(norm_ICs, norm_y)
        logger.info(f"Trained GP model with kernel: "
                    f"{self.gaussian_process.kernel_}")
        
    def gp_flat_law(self, flatICs, masses, immobile):
        """Compute the derivatives of the state using a learned GP model."""
        # Might be given multiple ICs in a batch, shape (C, 2 N D)
        if flatICs.ndim == 1:
            ICs = flatICs.reshape(1, -1)
        else:
            ICs = flatICs
        means = self.gaussian_process.predict(self.renormaliseX(ICs))
        acc = self.factory.engine.array(means)  # Accelerations
        acc = self.denormaliseY(acc)
        velocities = ICs.reshape(ICs.shape[0], 2, -1)[:, 1, :]
        derivatives = self.factory.engine.np.stack(
            [velocities, acc], axis=1)  # Shape N_sys, 2, N_bodies * D
        return derivatives

    def gp_law(self, system: System):
        """Compute the derivatives of the state using a learned GP model."""
        ICs = system.data.flatten().reshape(1, -1)
        means = self.gaussian_process.predict(self.renormaliseX(ICs))
        acc = self.factory.engine.array(means)[0]  # Accelerations
        acc = self.denormaliseY(acc)
        velocities = system.data[1].flatten()
        print("Scales: X mean:", self.X_mean, "X std:", self.X_std)
        print("Scales: Y mean:", self.Y_mean, "Y std:", self.Y_std)
        print("GP predicted accelerations:", acc)
        print("Current velocities:", velocities)
        derivatives = self.factory.engine.np.stack(
            [velocities, acc], axis=0)  # Shape 2, N_bodies * D
        return derivatives.reshape(system.data.shape)