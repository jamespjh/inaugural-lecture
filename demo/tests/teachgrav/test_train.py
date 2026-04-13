"""Tests for the --train switch on the teachgrav CLI."""
import os
import tempfile
import pytest

from teachgrav.entry import parse_args, execute_scenario


# ---------------------------------------------------------------------------
# parse_args --train validation
# ---------------------------------------------------------------------------

def test_train_flag_parsed():
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml')
    assert args.train is True


def test_train_requires_fitted_law():
    with pytest.raises(ValueError, match='does not require training'):
        parse_args('--train --law gravity --scenario scatter '
                   '--model-data /tmp/model.yaml')


def test_train_requires_fitted_law_constant():
    with pytest.raises(ValueError, match='does not require training'):
        parse_args('--train --law constant --scenario scatter '
                   '--model-data /tmp/model.yaml')


def test_train_requires_stochastic_scenario():
    with pytest.raises(ValueError, match='not suitable for training'):
        parse_args('--train --law power --scenario moon '
                   '--model-data /tmp/model.yaml')


def test_train_requires_model_data():
    with pytest.raises(ValueError, match='--model-data'):
        parse_args('--train --law power --scenario scatter')


def test_train_rejects_video():
    with pytest.raises(ValueError, match='visualization options'):
        parse_args('--train --law power --scenario scatter '
                   '--model-data /tmp/model.yaml --video')


def test_train_rejects_outfile():
    with pytest.raises(ValueError, match='visualization options'):
        parse_args('--train --law power --scenario scatter '
                   '--model-data /tmp/model.yaml --outfile out.mp4')


# ---------------------------------------------------------------------------
# End-to-end training via execute_scenario
# ---------------------------------------------------------------------------

def test_train_power_via_execute_scenario():
    """--train --law power should train and save a YAML file."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        output_path = f.name
    try:
        args = parse_args(f'--train --law power --scenario scatter '
                          f'--model-data {output_path} --n-systems 20')
        execute_scenario(args)
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


def test_train_gaussian_via_execute_scenario():
    """--train --law gaussian should train and save a joblib file."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        output_path = f.name
    try:
        args = parse_args(f'--train --law gaussian --scenario scatter '
                          f'--model-data {output_path} --n-systems 20')
        execute_scenario(args)
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
    """--train --seed should accept a seed and succeed."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        output_path = f.name
    try:
        args = parse_args(f'--train --law power --scenario scatter '
                          f'--model-data {output_path} '
                          '--n-systems 10 --seed 42')
        execute_scenario(args)
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
