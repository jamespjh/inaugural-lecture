import yaml
from scipy.optimize import minimize
from .laws import Model
from ..system import to_shaped
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

    def _acquisition(self, N_sys, likelihoods, ICs, accelerations):
        """Choose an order for probabilistic training scenarios.

        Scaffold implementation: preserve the current sequential order until
        acquisition math is implemented.
        """
        return range(N_sys)

    def probabilistic_train(
        self,
        N_sys,
        G_values,
        power_values,
        obs_noise=200,
        acquisition=False,
        on_step=None,
        **kwargs,
    ):
        """Generate a grid of values for the parameters.
           For each observation, compute the likelihood of the
           data under the model for each point in the grid,
           and update the grid of posterior likelihoods.

           Call the callback on_step with the grid of likelihoods after
           updating from each observation.

        Args:
            N_sys: number of random scatter systems to train on.
            G_values: 1-D array of G grid values.
            power_values: 1-D array of power (n) grid values.
            obs_noise: standard deviation of the observation noise.
            acquisition: when true, choose scenario order via
                         ``_acquisition(...)``.
            on_step: optional callable invoked after each optimisation
                     iteration with the grid of likelihoods.
            **kwargs: extra keyword arguments forwarded to
                      ``factory.create_training_data``."""
        ICs, accelerations, masses, immobile = (
            self.factory.create_training_data(N_sys, **kwargs)
        )

        logger.info("Training Probabilistic Power Law model...")
        np = ICs.__array_namespace__()
        N_G = len(G_values)
        N_power = len(power_values)
        likelihoods = np.ones((N_G, N_power)) / (
            N_G * N_power
        )  # Uniform prior

        def model(ICs_batch, ks, ns):
            # Vectorised model that takes in a batch of parameter values
            # and returns the predicted accelerations for
            # each set of parameters
            # and each system in the batch.
            # ICs shape: (N_sys, 2, N_bodies, D)
            # ks shape: (N_G,)
            # ns shape: (N_power,)
            # Output shape: (N_G, N_power, N_sys, N_bodies, D)
            self.G = ks
            self.power = ns
            C = ICs_batch.shape[0]
            N_bodies = len(masses)

            flat = self.flat_law(
                ICs_batch.reshape((C, -1)),
                masses=masses,
                immobile=immobile,
            )
            vector_results = flat.reshape(
                (len(ks), len(ns), C, 2, N_bodies, -1)
            )
            return vector_results[:, :, :, 1, :, :]
        if acquisition:
            scenarios_order = self._acquisition(
                N_sys,
                likelihoods,
                ICs,
                accelerations,
            )
        else:
            scenarios_order = range(N_sys)
        for i in scenarios_order:
            accs = model(ICs[i: i + 1], G_values, power_values)
            observed = accelerations[i: i + 1].reshape((1, len(masses), -1))

            delta = accs - observed[np.newaxis, np.newaxis, :, :, :]
            sse = np.sum(delta**2, axis=(-1, -2, -3))  # (N_G, N_power)

            # Gaussian observation model up to proportionality constant.
            # Subtract max log-likelihood for numerical stability.
            log_step_likelihood = -0.5 * sse / (obs_noise**2)
            log_step_likelihood = log_step_likelihood - np.max(
                log_step_likelihood
            )
            step_likelihood = np.exp(log_step_likelihood)
            likelihoods = likelihoods * step_likelihood

            norm = np.sum(likelihoods)
            if norm > 0:
                likelihoods = likelihoods / norm
            else:
                logger.warning(
                    "Degenerate likelihoods encountered. "
                    "Resetting to uniform distribution.")
                # Degenerate case: keep a valid distribution.
                likelihoods = np.ones((N_G, N_power)) / (N_G * N_power)

            if on_step is not None:
                on_step(likelihoods)
            G_grid = G_values[:, np.newaxis]
            power_grid = power_values[np.newaxis, :]
            G_mean = float(np.sum(likelihoods * G_grid))
            power_mean = float(np.sum(likelihoods * power_grid))
            logger.debug(
                f"Step {i + 1}/{N_sys}: "
                f"Posterior mean parameters: G={G_mean}, "
                f"power={power_mean}")

        # Set model parameters to posterior means for downstream use.
        G_grid = G_values[:, np.newaxis]
        power_grid = power_values[np.newaxis, :]
        self.G = float(np.sum(likelihoods * G_grid))
        self.power = float(np.sum(likelihoods * power_grid))
        logger.info(
            f"Probabilistic training complete. Posterior mean parameters: G={
                self.G}, power={
                self.power}"
        )
        # Return the final posterior grid over parameter values.
        return likelihoods

    def train(self, N_sys, on_step=None, **kwargs):
        """Train a model on random scatters for a given set of args.

        Args:
            N_sys: number of random scatter systems to train on.
            on_step: optional callable invoked after each optimisation
                     iteration with a dict ``{'G': float, 'power': float}``.
            **kwargs: extra keyword arguments forwarded to
                      ``factory.create_training_data``.

        Returns:
            list of checkpoint dicts recorded at each optimisation step,
            e.g. ``[{'G': 0.5, 'power': 3.0}, ...]``.
        """
        ICs, accelerations, masses, immobile = (
            self.factory.create_training_data(N_sys, **kwargs)
        )
        target_acc = to_numpy_host(accelerations)

        logger.info("Training Power Law model...")

        checkpoints = []

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
            return float(to_numpy_host(delta**2).sum())

        def callback(xk):
            params = {"G": float(xk[0]), "power": float(xk[1])}
            checkpoints.append(params)
            if on_step is not None:
                on_step(params)

        res = minimize(
            objective,
            x0=[self.G, self.power],
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            callback=callback,
        )
        logger.info(f"Optimization result: {res}")
        logger.debug(f"Optimizer message: {res.message}")
        pars = res.x
        self.G = float(pars[0])
        self.power = float(pars[1])

        logger.info(
            f"Trained Power Law model with parameters:"
            f"{self.G}, {self.power}"
        )

        return checkpoints

    def save(self, path):
        """Save model parameters to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump({"G": float(self.G), "power": float(self.power)}, f)
        logger.info(f"Saved Power Law model parameters to {path}")

    @classmethod
    def load(cls, path, factory=None):
        """Load model parameters from a YAML file."""
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        model = cls(factory=factory, G=params["G"], power=params["power"])
        logger.info(
            f"Loaded Power Law model parameters from {path}: "
            f"G={model.G}, power={model.power}"
        )
        return model

    def flat_law(self, data, masses, immobile):
        """Compute the derivatives of the state."""
        # Incoming data shape 2D, size (C, 2 N D)
        data_flat = self.add_vectorising_dimension_if_needed(data)
        num_vec = data_flat.shape[0]
        data = to_shaped(data_flat, num_vec, num_bodies=len(immobile))
        G = self.add_vectorising_dimension_if_needed(
            self.factory.engine.array(self.G), target_ndim=1
        )  # shape N_G
        power = self.add_vectorising_dimension_if_needed(
            self.factory.engine.array(self.power), target_ndim=1
        )  # shape N_power
        np = data.__array_namespace__()
        dpositions = data[:, 1, :, :]  # Derivative of position is velocity
        # Each body experiences a gravitational force from
        # every other body, leading to acceleration
        # So we have an N*N*2 matrix of pairwise position differences
        # ... and thus an N*N*2 matrix of pairwise accelerations
        # Which we sum over the second axis to get the total acceleration
        # on each body
        positions = data[:, 0, :, :]  # shape (C, N, D)
        # Pairwise position differences: shape (C, N, N, D)
        displacements = (
            positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
        )
        distances = np.linalg.norm(displacements, axis=-1, keepdims=True)
        safe_distances = np.where(distances == 0, 1.0, distances)
        # Avoid division by zero, but will make no contribution
        # since we will zero out self-interactions next
        # Pairwise accelerations due to gravity
        accelerations = (
            -1.0
            * G[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            * masses[
                np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
            ]
            * displacements[np.newaxis, np.newaxis, :, :, :, :]
            / (
                safe_distances[np.newaxis, np.newaxis, :, :, :, :]
                ** (
                    power[
                        np.newaxis,
                        :,
                        np.newaxis,
                        np.newaxis,
                        np.newaxis,
                        np.newaxis,
                    ]
                    + 1
                )
            )
        )
        # Clear self-interactions
        accelerations = np.where(
            distances[np.newaxis, np.newaxis, :, :, :, :] > 0,
            accelerations,
            0.0,
        )
        # Sum accelerations from all other bodies
        # shape (N_G, N_power, C, N, D)
        dvelocities = np.sum(accelerations, axis=4)
        # logger.debug("Total Accelerations:\n%s", dvelocities)
        # Replicate dpositions identically for each paramter value, so we can
        # stack it with dvelocities
        dpositions = np.broadcast_to(
            dpositions, [len(G), len(power)] + list(dpositions.shape)
        )
        # shape N_G, N_p, C, 2, N, D)
        delta = np.stack([dpositions, dvelocities], axis=3)
        # Mask out the derivatives for immobile bodies
        # 1 where mobile, 0 where immobile
        mask = (~immobile).astype(delta.dtype)
        mask = mask[
            np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
        ]
        delta = delta * mask  # Zero out derivatives for immobile bodies

        # Shape (2 (pos, vel), N, D (x y z), )
        # Output shape: (N_G N_power C 2 N D)
        return delta.flatten()
