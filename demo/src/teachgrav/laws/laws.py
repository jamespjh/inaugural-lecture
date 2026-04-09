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


def create_law(law_name: str, factory=None):
    """
    Factory function to create a law model by name.
    
    Args:
        law_name: one of 'gravity', 'constant', 'gaussian', 'power'
        factory: ScenarioFactory instance for fitted laws (gaussian, power)
    
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
        model = GPModel(factory=factory)
        if factory:
            logger.info("Training Gaussian Process model for fitted law use")
            model.train(N_sys=50)  # MVP: conservative default
        return model
    elif law_name == 'power':
        from .pl import PLModel
        model = PLModel(factory=factory)
        if factory:
            logger.info("Training Power Law model for fitted law use")
            model.train(N_sys=50)  # MVP: conservative default
        return model
    else:
        raise ValueError(
            f"Unknown law '{law_name}'. "
            f"Valid choices: gravity, constant, gaussian, power"
        )
