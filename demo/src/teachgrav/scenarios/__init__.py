# Scenario definitions for gravitational simulations
#
# Scenarios are defined as subclasses of Scenario (see base.py).
# The naming convention is: the string name "foo" maps to the class
# FooScenario in the module teachgrav.scenarios.foo.
# Adding a new scenario only requires creating a new module following
# that convention.

import importlib
import logging

from ..system import System, to_shaped
from ..array_abstraction import ArrayAbstraction
from .base import Scenario  # noqa: F401 – re-exported for convenience

logger = logging.getLogger("Teachgrav")

# Scenarios that generate random initial conditions and are
# therefore suitable for use as training data for fitted laws.
STOCHASTIC_SCENARIOS = ["scatter"]

# Known scenario names – used for validation error messages and argument
# parser choices.  Update this list when adding a new scenario module.
KNOWN_SCENARIOS = ["moon", "sun", "scatter", "single", "boids"]


def _load_scenario_class(name: str):
    """Return the Scenario subclass for *name* via metaprogramming.

    Convention: name ``"foo"`` -> module ``teachgrav.scenarios.foo``
                               -> class  ``FooScenario``
    """
    module_path = f"teachgrav.scenarios.{name}"
    class_name = (
        "".join(part.capitalize() for part in name.split("_")) + "Scenario"
    )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Valid scenarios: {', '.join(sorted(KNOWN_SCENARIOS))}"
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Scenario module '{module_path}' does not define '{class_name}'."
        ) from exc
    return cls


class ScenarioFactory:
    def __init__(self, engine="numpy", seed=None, via_numpy=True):
        self._engine_name = engine
        self._seed = seed
        self.via_numpy = via_numpy
        self.engine = ArrayAbstraction(engine, seed=seed)

    def create_scenario(self, name: str, **kwargs) -> System:
        """Return a scenario system by name.

        The string *name* is resolved to a ``Scenario`` subclass via the
        naming convention described in ``_load_scenario_class``.

        When *via_numpy* is ``True``
        the scenario is first generated with a numpy engine (using the same
        seed) and then the resulting arrays are converted to the target engine.
        This ensures identical initial conditions for the same seed regardless
        of which engine is used.
        And enables scenarios to be generated with engines that don't support
        all the array operations needed for scenario generation.
        """
        cls = _load_scenario_class(name)

        if self.via_numpy and self._engine_name != "numpy":
            numpy_engine = ArrayAbstraction("numpy", seed=self._seed)
            system = cls(numpy_engine).create(**kwargs)
            return system.to_engine(self.engine)

        return cls(self.engine).create(**kwargs)

    def create_training_data(self, N_sys, **kwargs):
        """Create training data for learned models."""
        scenarios = [
            self.create_scenario("scatter", **kwargs) for _ in range(N_sys)
        ]

        ICs = self.engine.array(
            [system.data.flatten() for system in scenarios]
        )
        flatICs = ICs.reshape((N_sys, -1))
        masses = scenarios[0].masses
        immobile = scenarios[0].immobile
        from teachgrav.laws.true_law import TrueLawModel

        results = TrueLawModel(self).flat_law(flatICs, masses, immobile)

        vector_results = to_shaped(results, N_sys, len(masses))
        accelerations = vector_results[:, 1, :, :]
        flat_accelerations = accelerations.reshape((N_sys, -1))
        return flatICs, flat_accelerations, masses, immobile
