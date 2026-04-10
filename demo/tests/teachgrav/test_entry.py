import pytest

from teachgrav.entry import parse_args, execute_scenario


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


def test_n_bodies_scatter():
    args = parse_args('--scenario scatter --n-bodies 6')
    assert args.n_bodies == 6


def test_n_bodies_default_is_none():
    args = parse_args('--scenario scatter')
    assert args.n_bodies is None


@pytest.mark.parametrize('scenario', ['moon', 'sun', 'single'])
def test_n_bodies_non_scatter_raises(scenario):
    with pytest.raises(
            ValueError,
            match="Option --n-bodies can only be used with the scatter"):
        parse_args(f'--scenario {scenario} --n-bodies 5')


def test_parse_args_seed_option():
    args = parse_args('--seed 42')
    assert args.seed == 42


def test_parse_args_seed_default_is_none():
    args = parse_args(' ')
    assert args.seed is None


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
        n_bodies = None
        seed = None

    captured = {}

    def fake_benchmark(fn, *args):
        captured['arg_count'] = len(args)
        captured['args'] = args
        fn()
        return 0.1

    monkeypatch.setattr('teachgrav.entry.benchmark_engine', fake_benchmark)
    execute_scenario(Args())
    assert captured['arg_count'] == 1
    assert captured['args'][0] == 'numpy'


def test_benchmark_mode_runs_single_law_step_not_full_solve(monkeypatch):
    """Benchmark should call model.law() once, not the full solver."""
    class Args:
        log_level = 'WARNING'
        log_file = None
        engine = 'numpy'
        scenario = 'single'
        benchmark = True
        method = 'euler'
        law = 'gravity'
        visualise = None
        outfile = None
        video = False
        duration = 30
        format = 'csv'
        n_bodies = None
        seed = None

    law_call_count = {'count': 0}
    solve_call_count = {'count': 0}

    from teachgrav.laws import laws as laws_module
    original_create_law = laws_module.create_law

    def tracking_create_law(law_name, factory=None):
        model = original_create_law(law_name, factory=factory)
        original_law = model.law

        def tracking_law(system):
            law_call_count['count'] += 1
            return original_law(system)

        model.law = tracking_law
        return model

    monkeypatch.setattr('teachgrav.entry.create_law', tracking_create_law)

    def tracking_solve(*a, **kw):
        solve_call_count['count'] += 1

    monkeypatch.setattr('teachgrav.entry.solve', tracking_solve)

    def fake_benchmark(fn, *args):
        fn()
        return 0.1

    monkeypatch.setattr('teachgrav.entry.benchmark_engine', fake_benchmark)
    execute_scenario(Args())

    assert law_call_count['count'] == 1, \
        "Expected exactly one call to model.law() in benchmark mode"
    assert solve_call_count['count'] == 0, \
        "Expected no calls to solve() in benchmark mode"
