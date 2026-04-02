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
        vector_results = (flat_law(flatICs, masses, immobile)
                          .reshape((N_sys, -1)))

        logger.info("Training GP model...")
        self.gaussian_process.fit(flatICs, vector_results)
        logger.info(f"Trained GP model with kernel: "
                    f"{self.gaussian_process.kernel_}")

    def gp_law(self, system: System):
        """Compute the derivatives of the state using a learned GP model."""
        means = self.gaussian_process.predict(system.data.reshape(1, -1))
        return self.factory.engine.array(means).reshape(system.data.shape)
