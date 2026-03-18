import logging

from teachgrav.laws import law
logger = logging.getLogger(__name__)


def test_law():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('moon')
    derivatives = law(system)
    # 2 bodies, 4 derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt)
    assert derivatives.shape == (2, 2, 2)


def test_law_immobile():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('sun')
    derivatives = law(system)
    logger.info(f"Derivatives:\n{derivatives}")
    # The Sun is immobile, so its derivatives should be zero
    assert derivatives[0][0][:].tolist() == [0.0, 0.0]
    assert derivatives[1][0][:].tolist() == [0.0, 0.0]
    # The Earth has an initial radial velocity of 1.0
    assert derivatives[0][1][:].tolist() == [0.0, 1.0]
    # The Earth should have an acceleration toward the origin of 1.0
    assert derivatives[1][1][:].tolist() == [-1.0, 0.0]


def test_law_scatter():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('scatter', n_bodies=5, seed=42)
    derivatives = law(system)
    assert derivatives.shape == (2, 5, 2)
