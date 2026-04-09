"""Tests for the train-model CLI entry point."""
import os
import tempfile

from teachgrav.train import train_model


def test_train_power_law_saves_file():
    """train-model --law power should train and save a YAML file."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        output_path = f.name
    try:
        train_model(f'--law power --output {output_path} '
                    '--n-systems 20 --n-bodies 3')
        assert os.path.exists(output_path)
        import yaml
        with open(output_path) as fh:
            params = yaml.safe_load(fh)
        assert 'G' in params
        assert 'power' in params
        assert isinstance(params['G'], float)
        assert isinstance(params['power'], float)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_train_gaussian_saves_file():
    """train-model --law gaussian should train and save a joblib file."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        output_path = f.name
    try:
        train_model(f'--law gaussian --output {output_path} '
                    '--n-systems 20 --n-bodies 3')
        assert os.path.exists(output_path)
        import joblib
        state = joblib.load(output_path)
        assert 'gaussian_process' in state
        assert 'X_mean' in state
        assert 'Y_mean' in state
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_train_with_seed():
    """train-model --seed should accept and use a seed without error."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        output_path = f.name
    try:
        train_model(f'--law power --output {output_path} '
                    '--n-systems 10 --n-bodies 3 --seed 42')
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
