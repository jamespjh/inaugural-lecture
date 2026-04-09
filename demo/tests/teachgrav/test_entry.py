import pytest

from teachgrav.entry import parse_args, execute_scenario


def test_parse_model_data_arg():
    args = parse_args('--law power --model-data /tmp/model.yaml')
    assert args.model_data == '/tmp/model.yaml'
    assert args.law == 'power'


def test_default_model_data_is_none():
    args = parse_args(' ')
    assert args.model_data is None


def test_parse_args():
    args = parse_args(
        '--scenario scatter --method Tsit5 --outfile output.mp4 ' +
        '--visualise dot --video')
    assert args.scenario == 'scatter'
    assert args.method == 'Tsit5'
    assert args.outfile == 'output.mp4'
    assert args.visualise == 'dot'
    assert args.video
    assert args.format == 'mp4'


def test_default_args():
    args = parse_args(" ")
    assert args.scenario == 'moon'
    assert args.method == 'euler'
    assert args.outfile is None
    assert args.visualise is None
    assert not args.video
    assert args.duration == 30
    assert args.format == 'csv'


def test_duration_with_video_args():
    args = parse_args('--video --duration 45 --outfile output.mp4')
    assert args.video
    assert args.duration == 45


def test_duration_without_video_raises():
    with pytest.raises(
            ValueError,
            match='Option --duration can only be used with video output'):
        parse_args('--duration 45')


def test_parse_args_law_option():
    args = parse_args('--law constant')
    assert args.law == 'constant'


@pytest.mark.parametrize('law', ['gaussian', 'power'])
@pytest.mark.parametrize('method', ['RK45', 'Tsit5'])
def test_fitted_law_forces_euler_with_warning(caplog, law, method):
    with caplog.at_level('WARNING', logger='Teachgrav'):
        args = parse_args(f'--law {law} --method {method}')
    assert args.method == 'euler'
    assert f"Fitted law '{law}' is not compatible with solver" in caplog.text


def test_benchmark_mode_passes_law_and_timing(monkeypatch):
    class Args:
        log_level = 'WARNING'
        log_file = None
        engine = 'numpy'
        scenario = 'single'
        benchmark = True
        method = 'euler'
        law = 'constant'
        visualise = None
        outfile = None
        video = False
        duration = 30
        format = 'csv'

    captured = {}

    def fake_benchmark(fn, *args):
        captured['arg_count'] = len(args)
        fn()
        return 0.1

    monkeypatch.setattr('teachgrav.entry.benchmark', fake_benchmark)
    execute_scenario(Args())
    assert captured['arg_count'] == 0
