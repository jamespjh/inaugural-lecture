import pytest

from teachgrav.scenarios import ScenarioFactory
from teachgrav.laws.pl import PLModel
from teachgrav.laws.true_law import TrueLawModel
from engines import ENGINES_TO_TEST


def test_pl_train():
    factory = ScenarioFactory('numpy')
    model = PLModel(factory)
    model.train(256, n_bodies=3)


@pytest.mark.parametrize("n_bodies", [2, 3, 10])
def test_pl_predict(n_bodies):
    factory = ScenarioFactory('numpy')
    model = PLModel(factory)
    truth = TrueLawModel()
    masses = [1.0] * n_bodies
    model.train(256, n_bodies=n_bodies, fixed_masses=masses)

    scenario = factory.create_scenario('scatter', n_bodies=n_bodies,
                                       fixed_masses=masses)
    pl_res = model.law(scenario)
    res = truth.law(scenario)
    print("PL result:\n", pl_res)
    print("True result:\n", res)
    assert pl_res.shape == res.shape
    assert factory.engine.np.allclose(pl_res, res, atol=0.01)


@pytest.mark.parametrize("engine", [
    e for e in ['numpy', 'jax-cpu', 'mlx-cpu'] if e in ENGINES_TO_TEST
])
def test_pl_law_vectorised(engine):
    factory = ScenarioFactory(engine=engine)
    N_sys = 5
    N_bodies = 3
    masses = factory.engine.array([1.0, 1.0, 1.0])
    # Test that the law can be called multiple times over an array of states
    systems = [
        factory.create_scenario(
            'scatter',
            n_bodies=N_bodies,
            fixed_masses=masses,
        )
        for _ in range(N_sys)
    ]
    model = PLModel(factory)
    model.train(256, n_bodies=N_bodies, fixed_masses=masses)
    simple_results = factory.engine.array(
        [model.law(system) for system in systems])
    ICs = factory.engine.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    ICs_flat = ICs.reshape((N_sys, -1))
    results = model.flat_law(ICs_flat, masses, immobile)
    vector_results = results.reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6)
