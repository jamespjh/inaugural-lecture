# Base class for scenario definitions

from ..system import System


class Scenario:
    """Base class for gravitational simulation scenarios.

    Subclasses must implement the ``create`` method which returns a ``System``
    instance.  The ``engine`` attribute provides array utilities from
    ``ArrayAbstraction``.
    """

    def __init__(self, engine):
        self.engine = engine

    def create(self, **kwargs) -> System:
        """Instantiate and return a System for this scenario.

        Args:
            **kwargs: scenario-specific parameters

        Returns:
            A configured System instance
        """
        raise NotImplementedError("Subclasses must implement create()")
