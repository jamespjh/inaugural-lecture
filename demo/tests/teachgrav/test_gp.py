from teachgrav.laws import law
from teachgrav.scenarios import ScenarioFactory
from teachgrav.gp import GPModel

def test_gp_train():
    factory = ScenarioFactory('numpy')
    model = GPModel(factory)
    model.train(256, n_bodies=3)

def test_gp_predict():
    factory = ScenarioFactory('numpy')
    model = GPModel(factory)
    model.train(256, n_bodies=2, fixed_masses=[1.0, 1.0])

    scenario = factory.create_scenario('scatter', n_bodies=2,
                                       fixed_masses=[1.0, 1.0])
    gp_res = model.gp_law(scenario)
    res = law(scenario)
    print("GP result:\n", gp_res)
    print("True result:\n", res)
    assert gp_res.shape == res.shape
    assert factory.engine.np.allclose(gp_res, res, atol=0.2)

def t_law_vectorised(factory):
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
    model = GPModel(factory)
    model.train(256, n_bodies=N_bodies, fixed_masses=masses)
    simple_results = factory.engine.array([model.gp_law(system) for system in systems])
    ICs = factory.engine.array([system.data.flatten() for system in systems])
    masses = systems[0].masses
    immobile = systems[0].immobile
    ICs_flat = ICs.reshape((N_sys, -1))
    results = model.gp_flat_law(ICs_flat, masses, immobile)
    vector_results = results.reshape((N_sys, 2, N_bodies, -1))
    assert simple_results.shape == vector_results.shape
    assert vector_results.__array_namespace__().allclose(
        simple_results, vector_results, atol=1e-6)


def test_gp_law_vectorised():
    t_law_vectorised(ScenarioFactory(engine='numpy'))


def test_gp_law_vectorised_jax():
    t_law_vectorised(ScenarioFactory(engine='jax-cpu'))


def test_gp_law_vectorised_metal():
    t_law_vectorised(ScenarioFactory(engine='mlx-cpu'))


def test_normalise_denormalise():
    from teachgrav.gp import GPModel
    from teachgrav.scenarios import ScenarioFactory
    factory = ScenarioFactory('numpy')
    model = GPModel(factory)

    X = factory.engine.random_array((10, 5))
    normed = model.normaliseX(X)
    denormed = model.denormaliseX(normed)
    assert factory.engine.np.allclose(X, denormed, atol=1e-6)

    # Now make a single new data, and check that normalising and denormalising gives the same result
    new_X = factory.engine.random_array((5,))
    normed_new_X = model.renormaliseX(new_X)
    denormed_new_X = model.denormaliseX(normed_new_X)
    assert factory.engine.np.allclose(new_X, denormed_new_X, atol=1e-6)