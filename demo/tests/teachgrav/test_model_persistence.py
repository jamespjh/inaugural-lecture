"""Tests for model persistence (save/load) for PLModel and GPModel."""
import os
import tempfile
import pytest

from teachgrav.scenarios import ScenarioFactory
from teachgrav.laws.pl import PLModel
from teachgrav.laws.gp import GPModel
from teachgrav.laws.laws import create_law


class TestPLModelPersistence:
    def setup_method(self):
        self.factory = ScenarioFactory('numpy')

    def test_save_and_load_roundtrip(self):
        """PLModel save/load should preserve G and power parameters."""
        model = PLModel(factory=self.factory)
        model.train(20, n_bodies=3)
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = PLModel.load(path, factory=self.factory)
            assert abs(loaded.G - model.G) < 1e-10
            assert abs(loaded.power - model.power) < 1e-10
        finally:
            os.remove(path)

    def test_load_produces_same_predictions(self):
        """PLModel loaded from file should predict same as original."""
        model = PLModel(factory=self.factory)
        model.train(20, n_bodies=3)
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = PLModel.load(path, factory=self.factory)
            scenario = self.factory.create_scenario('scatter', n_bodies=3)
            orig = model.law(scenario)
            restored = loaded.law(scenario)
            assert self.factory.engine.np.allclose(
                orig, restored, atol=1e-10)
        finally:
            os.remove(path)


class TestGPModelPersistence:
    def setup_method(self):
        self.factory = ScenarioFactory('numpy')

    def test_save_and_load_roundtrip(self):
        """GPModel save/load should restore the sklearn model and norms."""
        model = GPModel(factory=self.factory)
        model.train(20, n_bodies=2, fixed_masses=[1.0, 1.0])
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = GPModel.load(path, factory=self.factory)
            scenario = self.factory.create_scenario(
                'scatter', n_bodies=2, fixed_masses=[1.0, 1.0])
            orig = model.law(scenario)
            restored = loaded.law(scenario)
            assert self.factory.engine.np.allclose(
                orig, restored, atol=1e-10)
        finally:
            os.remove(path)


class TestCreateLawModelData:
    def setup_method(self):
        self.factory = ScenarioFactory('numpy')
    def test_gaussian_requires_model_data(self):
        """create_law('gaussian') without model_data raises ValueError."""
        with pytest.raises(ValueError, match='--model-data'):
            create_law('gaussian', factory=self.factory)

    def test_power_requires_model_data(self):
        """create_law('power') without model_data raises ValueError."""
        with pytest.raises(ValueError, match='--model-data'):
            create_law('power', factory=self.factory)

    def test_gaussian_loads_from_model_data(self):
        """create_law('gaussian', model_data=path) loads and returns model."""
        factory = ScenarioFactory('numpy')
        model = GPModel(factory=factory)
        model.train(20, n_bodies=2, fixed_masses=[1.0, 1.0])
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = create_law('gaussian', factory=factory, model_data=path)
            assert isinstance(loaded, GPModel)
        finally:
            os.remove(path)

    def test_power_loads_from_model_data(self):
        """create_law('power', model_data=path) loads and returns model."""
        factory = ScenarioFactory('numpy')
        model = PLModel(factory=factory)
        model.train(20, n_bodies=3)
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = create_law('power', factory=factory, model_data=path)
            assert isinstance(loaded, PLModel)
        finally:
            os.remove(path)

    def test_gravity_ignores_model_data(self):
        """create_law('gravity') should work regardless of model_data."""
        from teachgrav.laws.true_law import TrueLawModel
        model = create_law('gravity', factory=self.factory, model_data='ignored_path')
        assert isinstance(model, TrueLawModel)
