from teachgrav.scenarios import create_scenario


def test_create_scenario_moon():
    system = create_scenario('moon')
    assert len(system.positions) == 2
    assert len(system.velocities) == 2
    assert len(system.masses) == 2


def test_create_scenario_scatter():
    system = create_scenario('scatter', n_bodies=10, seed=42)
    assert len(system.positions) == 10
    assert len(system.velocities) == 10
    assert len(system.masses) == 10


def test_scenario_sun():
    system = create_scenario('sun')
    assert len(system.positions) == 2
    assert len(system.velocities) == 2
    assert len(system.masses) == 2
    assert system.immobile[0]
    assert not system.immobile[1]
