from teachgrav.laws import law

def test_law():
    from teachgrav.scenarios import create_scenario
    system = create_scenario('moon')
    derivatives = law(system)
    assert derivatives.shape == (2, 4)  # 2 bodies, 4 derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt)