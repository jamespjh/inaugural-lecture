class Model:
    def __init__(self, factory, **kwargs):
        self.factory = factory

    def law(self, system):
        """Compute the derivatives of the state."""
        return self.flat_law(system.data.flatten(), system.masses,
                             system.immobile).reshape(system.data.shape)

    def flat_law(self, data, masses, immobile):
        """Compute the derivatives of the state using a learned model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def add_vectorising_dimension_if_needed(self, ICs):
        """Add a vectorising dimension to the ICs."""
        if ICs.ndim == 1:
            return ICs.reshape(1, -1)
        else:
            return ICs


def create_law(law_name: str, factory=None, model_data: str = None):
    """
    Factory function to create a law model by name.

    Args:
        law_name: one of 'gravity', 'constant', 'gaussian', 'power'
        factory: ScenarioFactory instance for fitted laws (gaussian, power)
        model_data: path to a saved model file (required for gaussian, power)

    Returns:
        Model instance for the selected law
    """
    import logging
    logger = logging.getLogger("Teachgrav")

    if law_name == 'gravity':
        from .true_law import TrueLawModel
        return TrueLawModel()
    elif law_name == 'constant':
        from .constant_law import ConstantLawModel
        return ConstantLawModel(factory=factory)
    elif law_name == 'gaussian':
        from .gp import GPModel
        if model_data is None:
            raise ValueError(
                "Law 'gaussian' requires a trained model file. "
                "Use --model-data to specify the path, or run "
                "'train-model --law gaussian' to train a model first."
            )
        if factory is None:
            raise ValueError(
                "Law 'gaussian' requires a ScenarioFactory. "
                "Pass factory=... when loading a Gaussian Process model."
            )
        logger.info(f"Loading Gaussian Process model from {model_data}")
        return GPModel.load(model_data, factory=factory)
    elif law_name == 'power':
        from .pl import PLModel
        if model_data is None:
            raise ValueError(
                "Law 'power' requires a trained model file. "
                "Use --model-data to specify the path, or run "
                "'train-model --law power' to train a model first."
            )
        logger.info(f"Loading Power Law model from {model_data}")
        return PLModel.load(model_data, factory=factory)
    else:
        raise ValueError(
            f"Unknown law '{law_name}'. "
            f"Valid choices: gravity, constant, gaussian, power"
        )
