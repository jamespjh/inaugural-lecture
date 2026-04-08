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
