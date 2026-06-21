import pytest
import numpy as np

from teachgrav.scenarios import ScenarioFactory
from teachgrav.laws.pl import PLModel
from teachgrav.laws.true_law import TrueLawModel
from engines import ENGINES_TO_TEST


@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_pl_train(engine):
    factory = ScenarioFactory(engine=engine)
    model = PLModel(factory)
    model.train(256, n_bodies=3)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("n_bodies", [2, 3, 10])
def test_pl_predict(n_bodies):
    factory = ScenarioFactory("numpy")
    model = PLModel(factory)
    truth = TrueLawModel(factory)
    masses = [1.0] * n_bodies
    model.train(256, n_bodies=n_bodies, fixed_masses=masses)

    scenario = factory.create_scenario(
        "scatter", n_bodies=n_bodies, fixed_masses=masses
    )
    pl_res = model.law(scenario)
    res = truth.law(scenario)
    print("PL result:\n", pl_res)
    print("True result:\n", res)
    assert pl_res.shape == res.shape
    assert factory.engine.np.allclose(pl_res, res, atol=0.01)


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("engine", ENGINES_TO_TEST)
def test_pl_law_vectorised(engine):
    factory = ScenarioFactory(engine=engine, via_numpy=False)
    N_sys = 5
    N_bodies = 3
    masses = factory.engine.array([1.0, 1.0, 1.0])
    # Test that the law can be called multiple times over an array of states
    systems = [
        factory.create_scenario(
            "scatter",
            n_bodies=N_bodies,
            fixed_masses=masses,
        )
        for _ in range(N_sys)
    ]
    model = PLModel(factory)
    model.train(256, n_bodies=N_bodies, fixed_masses=masses)
    simple_results = factory.engine.array(
        [model.law(system) for system in systems]
    )
    ICs = factory.engine.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    ICs_flat = ICs.reshape((N_sys, -1))
    results = model.flat_law(ICs_flat, masses, immobile)
    vector_results = results.reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6
    )


def test_pl_flat_law_vectorised_matches_scalar_pairs():
    """Diagonal entries of vectorised (G, power) match scalar pair calls."""
    factory = ScenarioFactory(engine="numpy", seed=7)
    masses = factory.engine.array([1.0, 1.5, 2.0])
    system = factory.create_scenario(
        "scatter", n_bodies=3, fixed_masses=masses
    )

    pair_G = np.array([0.2, 0.8, -1.1, 1.7])
    pair_power = np.array([1.8, 2.3, 3.0, 3.6])

    model = PLModel(factory)
    model.G = factory.engine.array(pair_G)
    model.power = factory.engine.array(pair_power)

    ICs_flat = factory.engine.array([system.data.flatten()])
    vector_flat = model.flat_law(ICs_flat, system.masses, system.immobile)
    vector = vector_flat.reshape(
        (len(pair_G), len(pair_power), 1) + system.data.shape
    )

    for i in range(len(pair_G)):
        scalar_model = PLModel(
            factory,
            G=float(pair_G[i]),
            power=float(pair_power[i]),
        )
        scalar_flat = scalar_model.flat_law(
            ICs_flat, system.masses, system.immobile
        )
        scalar = scalar_flat.reshape((1, 1, 1) + system.data.shape)
        assert np.allclose(vector[i, i, 0], scalar[0, 0, 0], atol=1e-12)


def test_pl_probabilistic_train_acquisition_pathway(monkeypatch):
    """Integer acquisition budget should route through _acquisition."""
    factory = ScenarioFactory(engine="numpy", seed=11)
    model = PLModel(factory)
    G_values = factory.engine.array([-1.0, 0.0, 1.0])
    power_values = factory.engine.array([2.0, 3.0, 4.0])
    N_sys = 5

    chosen_order = []
    original_acquisition = PLModel._acquisition

    def recording_acquisition(self, *args, **kwargs):
        for idx in original_acquisition(self, *args, **kwargs):
            chosen_order.append(int(idx))
            yield idx

    monkeypatch.setattr(PLModel, "_acquisition", recording_acquisition)

    posterior = model.probabilistic_train(
        N_sys=N_sys,
        G_values=G_values,
        power_values=power_values,
        acquisition=3,
        n_bodies=3,
    )
    np_mod = posterior.__array_namespace__()

    assert len(chosen_order) == N_sys
    assert sorted(chosen_order) == list(range(N_sys))
    assert posterior.shape == (len(G_values), len(power_values))
    assert np_mod.allclose(np_mod.sum(posterior), 1.0)
