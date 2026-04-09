from teachgrav.laws.constant_law import ConstantLawModel
from teachgrav.scenarios import ScenarioFactory


factory = ScenarioFactory()


def test_constant_law_derivative_single_body():
    system = factory.create_scenario(
        'single',
        position=[2.0, -1.0],
        velocity=[0.5, -0.25],
        mass=1.0,
    )
    model = ConstantLawModel()
    delta = model.law(system)

    np = delta.__array_namespace__()
    assert delta.shape == (2, 1, 2)
    assert np.allclose(delta[0, 0], system.velocities()[0])
    assert np.allclose(delta[1, 0], np.array([0.0, 0.0]))


def test_constant_law_respects_immobile_mask():
    system = factory.create_scenario('moon')
    np = system.data.__array_namespace__()
    system.immobile = np.array([True, False])

    model = ConstantLawModel()
    delta = model.law(system)

    assert delta.shape == (2, 2, 2)
    assert np.allclose(delta[:, 0, :], np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert np.allclose(delta[0, 1, :], system.velocities()[1])
    assert np.allclose(delta[1, 1, :], np.array([0.0, 0.0]))
