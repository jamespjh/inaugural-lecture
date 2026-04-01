from teachgrav.scenarios import ScenarioFactory

factory = ScenarioFactory()
jax_factory = ScenarioFactory(engine='jax-cpu')


def test_create_scenario_moon():
    system = factory.create_scenario('moon')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2


def test_create_scenario_scatter():
    system = factory.create_scenario('scatter', n_bodies=10)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_create_scenario_scatter_3D():
    system = factory.create_scenario('scatter', n_bodies=10, dimensions=3)
    assert len(system.positions()) == 10
    assert len(system.velocities()) == 10
    assert len(system.masses) == 10


def test_scenario_sun():
    system = factory.create_scenario('sun')
    assert len(system.positions()) == 2
    assert len(system.velocities()) == 2
    assert len(system.masses) == 2
    assert system.immobile[0]
    assert not system.immobile[1]
