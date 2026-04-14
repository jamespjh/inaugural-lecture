"""Tests for the convergence video feature (Issue #8).

These tests cover:
  * PLModel.train() returning a list of checkpoints.
  * New CLI arguments: --convergence-video, --checkpoint-interval,
    --show-true-law.
  * Validation: --convergence-video only allowed for --law power.
  * _generate_convergence_video() running end-to-end without ffmpeg by
    using a mock/patch so the actual video-write step is skipped.
"""
import os
import tempfile
from unittest.mock import patch

import pytest

from teachgrav.entry import parse_args, execute_scenario
from teachgrav.laws.pl import PLModel
from teachgrav.scenarios import ScenarioFactory


# ---------------------------------------------------------------------------
# PLModel checkpoint collection
# ---------------------------------------------------------------------------

def test_train_returns_checkpoints():
    """PLModel.train() should return a non-empty list of checkpoint dicts."""
    factory = ScenarioFactory('numpy', seed=1)
    model = PLModel(factory=factory)
    checkpoints = model.train(N_sys=10)
    assert isinstance(checkpoints, list)
    assert len(checkpoints) > 0
    for ckpt in checkpoints:
        assert 'G' in ckpt
        assert 'power' in ckpt
        assert isinstance(ckpt['G'], float)
        assert isinstance(ckpt['power'], float)


def test_train_on_step_callback():
    """on_step is called once per optimisation iteration."""
    factory = ScenarioFactory('numpy', seed=2)
    model = PLModel(factory=factory)
    called_with = []

    def on_step(params):
        called_with.append(dict(params))

    checkpoints = model.train(N_sys=10, on_step=on_step)
    assert len(called_with) == len(checkpoints)
    for i, ckpt in enumerate(checkpoints):
        assert called_with[i]['G'] == ckpt['G']
        assert called_with[i]['power'] == ckpt['power']


def test_train_final_params_match_last_checkpoint():
    """After training, model.G and model.power equal the result of minimize."""
    factory = ScenarioFactory('numpy', seed=3)
    model = PLModel(factory=factory)
    checkpoints = model.train(N_sys=10)
    # The final stored parameters come from res.x, *not* the last callback.
    # They should still be floats and within the optimiser bounds.
    assert -5.0 <= model.G <= 5.0
    assert -5.0 <= model.power <= 5.0
    assert len(checkpoints) >= 1


# ---------------------------------------------------------------------------
# New parse_args arguments
# ---------------------------------------------------------------------------

def test_convergence_video_arg_parsed():
    """--convergence-video should be stored on the parsed namespace."""
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml '
                      '--convergence-video /tmp/convergence.mp4')
    assert args.convergence_video == '/tmp/convergence.mp4'


def test_checkpoint_interval_default():
    """--checkpoint-interval defaults to 1."""
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml')
    assert args.checkpoint_interval == 1


def test_checkpoint_interval_custom():
    """--checkpoint-interval can be set to any positive integer."""
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml '
                      '--checkpoint-interval 5')
    assert args.checkpoint_interval == 5


def test_show_true_law_flag():
    """--show-true-law flag should be parseable."""
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml '
                      '--convergence-video /tmp/conv.mp4 '
                      '--show-true-law')
    assert args.show_true_law is True


def test_show_true_law_default_false():
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml')
    assert args.show_true_law is False


def test_duration_allowed_with_convergence_video_training():
    args = parse_args('--train --law power --scenario scatter '
                      '--model-data /tmp/model.yaml '
                      '--convergence-video /tmp/conv.mp4 '
                      '--duration 15')
    assert args.duration == 15


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_convergence_video_rejected_for_gaussian():
    """--convergence-video should raise for --law gaussian."""
    with pytest.raises(ValueError, match='only supported for --law power'):
        parse_args('--train --law gaussian --scenario scatter '
                   '--model-data /tmp/model.joblib '
                   '--convergence-video /tmp/conv.mp4')


def test_checkpoint_interval_must_be_positive():
    with pytest.raises((ValueError, SystemExit)):
        parse_args('--train --law power --scenario scatter '
                   '--model-data /tmp/model.yaml '
                   '--checkpoint-interval 0')


# ---------------------------------------------------------------------------
# End-to-end: convergence video generation (ffmpeg mocked out)
# ---------------------------------------------------------------------------

def test_convergence_video_generated_via_execute_scenario():
    """--train with --convergence-video produces a convergence video."""
    with (
        tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as mf,
        tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf,
    ):
        model_path = mf.name
        video_path = vf.name

    try:
        args = parse_args(
            f'--train --law power --scenario scatter '
            f'--model-data {model_path} --n-systems 15 --seed 7 '
            f'--convergence-video {video_path}')

        # Patch FuncAnimation.save so the test does not need ffmpeg installed.
        with patch('matplotlib.animation.FuncAnimation.save'):
            execute_scenario(args)

    finally:
        for path in (model_path, video_path):
            if os.path.exists(path):
                os.remove(path)


def test_convergence_video_with_true_law_overlay():
    """--show-true-law should not raise errors end-to-end (ffmpeg mocked)."""
    with (
        tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as mf,
        tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as vf,
    ):
        model_path = mf.name
        video_path = vf.name

    try:
        args = parse_args(
            f'--train --law power --scenario scatter '
            f'--model-data {model_path} --n-systems 15 --seed 7 '
            f'--convergence-video {video_path} --show-true-law')

        with patch('matplotlib.animation.FuncAnimation.save'):
            execute_scenario(args)

    finally:
        for path in (model_path, video_path):
            if os.path.exists(path):
                os.remove(path)


def test_convergence_video_with_checkpoint_interval():
    """--checkpoint-interval reduces the number of frames used."""
    from teachgrav.entry import _generate_convergence_video

    checkpoints = [{'G': float(i) * 0.1, 'power': 2.0 + i * 0.1}
                   for i in range(10)]

    # Build minimal args
    class FakeArgs:
        scenario = 'scatter'
        n_bodies = None
        seed = 42
        checkpoint_interval = 3
        convergence_video = '/tmp/fake_convergence.mp4'
        show_true_law = False
        convergence_fps = 5

    generated_trajectories = []

    def capturing_convergence_video(trajectories, output, **kwargs):
        generated_trajectories.extend(trajectories)

    scenario_kwargs = {}
    with patch(
        'teachgrav.entry.convergence_video', capturing_convergence_video
    ):
        _generate_convergence_video(FakeArgs(), checkpoints, scenario_kwargs)

    # With 10 checkpoints and interval=3, we expect ceil(10/3)=4 frames
    expected_count = len(checkpoints[::3])
    assert len(generated_trajectories) == expected_count


def test_convergence_video_forwards_solver_flags_to_integrator():
    """Checkpoint and overlay integrations should use method/dt/until args."""
    from teachgrav.entry import _generate_convergence_video
    import numpy as np

    checkpoints = [{'G': 1.0, 'power': 2.0}, {'G': 1.1, 'power': 2.1}]

    class FakeArgs:
        scenario = 'scatter'
        n_bodies = None
        seed = 42
        checkpoint_interval = 1
        convergence_video = '/tmp/fake_convergence.mp4'
        show_true_law = True
        convergence_fps = 5
        method = 'RK45'
        dt = 0.123
        until = 4.5

    class FakeTraj:
        def __init__(self):
            self.data = np.zeros((2, 2, 2), dtype=float)

    calls = []

    def fake_integrate_trajectory(system, method, **kwargs):
        calls.append({'method': method, **kwargs})
        return FakeTraj()

    def fake_convergence_video(trajectories, output, **kwargs):
        return None

    with patch(
        'teachgrav.entry.integrate_trajectory',
        fake_integrate_trajectory,
    ), patch('teachgrav.entry.convergence_video', fake_convergence_video):
        _generate_convergence_video(FakeArgs(), checkpoints, {})

    # 2 checkpoint runs + 1 true-law overlay run
    assert len(calls) == 3
    for c in calls:
        assert c['method'] == 'RK45'
        assert c['dt'] == pytest.approx(0.123)
        assert c['until'] == pytest.approx(4.5)


def test_convergence_video_skips_failed_checkpoint_integrations():
    """Integration failures for some checkpoints should be skipped."""
    from teachgrav.entry import _generate_convergence_video
    import numpy as np

    checkpoints = [{'G': 1.0, 'power': 2.0}, {'G': 1.1, 'power': 2.1}]

    class FakeArgs:
        scenario = 'scatter'
        n_bodies = None
        seed = 42
        checkpoint_interval = 1
        convergence_video = '/tmp/fake_convergence.mp4'
        show_true_law = False
        convergence_fps = 5
        method = 'LSODA'
        dt = 0.01
        until = 2.0

    class FakeTraj:
        def __init__(self):
            self.data = np.zeros((2, 2, 2), dtype=float)

    call_count = {'n': 0}
    captured = {'n_frames': None}

    def fake_integrate_trajectory(system, method, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise ValueError("cannot reshape array")
        return FakeTraj()

    def fake_convergence_video(trajectories, output, **kwargs):
        captured['n_frames'] = len(trajectories)

    with patch(
        'teachgrav.entry.integrate_trajectory',
        fake_integrate_trajectory,
    ), patch('teachgrav.entry.convergence_video', fake_convergence_video):
        _generate_convergence_video(FakeArgs(), checkpoints, {})

    assert call_count['n'] == 2
    assert captured['n_frames'] == 1
