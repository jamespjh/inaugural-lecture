from .system import System
from teachgrav import laws
from scipy.optimize import minimize

import logging
logger = logging.getLogger("Teachgrav")

# Fit a power-law model to the data
# We will use the same training data as the GP model, but instead of fitting
# a GP, we will fit a power-law model of the form:
# a = k * r^n
# where a is the acceleration, r is the distance, and k and n are the
# parameters to be learned.
class PLModel:
    def __init__(self, factory, **kwargs):
        self.factory = factory

    def train(self, N_sys, **kwargs):
        """Train a model on random scatters for a given set of args."""
        # Placeholder implementation, replace with actual GP training code
        scenarios = [self.factory.create_scenario('scatter', **kwargs)
                     for _ in range(N_sys)]

        ICs = self.factory.engine.np.array([system.data.flatten()
                                           for system in scenarios])
        flatICs = ICs.reshape((N_sys, -1))
        masses = scenarios[0].masses
        immobile = scenarios[0].immobile
        # We will give it for free, everything except the power-law parameters
        results = laws.flat_law(flatICs, masses, immobile)

        vector_results = results.reshape((N_sys, 2, len(masses), -1))
        accelerations = vector_results[:, 1, :, :]
        flat_accelerations = accelerations.reshape((N_sys, -1))

        logger.info("Training Power Law model...")

        def model(ICs, k, n):
            res = laws.flat_law(ICs, G=k, power=n, masses=masses, immobile=immobile)
            vector_results = res.reshape((N_sys, 2, len(masses), -1))
            accelerations = vector_results[:, 1, :, :]
            flat_accelerations = accelerations.reshape((N_sys, -1))
            return flat_accelerations
        
        def objective(params):
            k, n = params
            pred = model(flatICs, k, n)
            return self.factory.engine.np.mean((pred - flat_accelerations) ** 2)

        pars = minimize(objective, x0=[0.5, 3.0], bounds=[(-5.0, 5.0), (-5.0, 5.0)]).x
        self.G = self.factory.engine.np.array(pars[0]).item()
        self.power = self.factory.engine.np.array(pars[1]).item()

        logger.info(f"Trained Power Law model with parameters:"
                    f"{self.G}, {self.power}")

    def flat_law(self, flatICs, masses, immobile):
        """Compute the derivatives of the state using a learned GP model."""
        # Might be given multiple ICs in a batch, shape (C, 2 N D)
        if flatICs.ndim == 1:
            ICs = flatICs.reshape(1, -1)
        else:
            ICs = flatICs
        prediction = laws.flat_law(ICs, G=self.G, power=self.power, masses=masses, immobile=immobile)
        return prediction

    def law(self, system: System):
        """Compute the derivatives of the state using a learned GP model."""
        return self.flat_law(system.data.flatten(), system.masses,
                        system.immobile).reshape(system.data.shape)