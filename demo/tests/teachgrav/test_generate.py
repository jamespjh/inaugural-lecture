from pathlib import Path
from unittest.mock import patch

import pytest

from teachgrav import generate


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_run_batch_calls_parse_args_for_each_config():
    configs = [
        {
            "scenario": "sun",
            "method": "LSODA",
            "outfile": "sun.mp4",
            "visualise": "dot",
            "duration": 5,
        },
        {
            "scenario": "moon",
            "method": "euler",
            "outfile": "moon.csv",
        },
    ]

    with patch("teachgrav.generate.entry.parse_args") as mock_parse_args:
        mock_parse_args.side_effect = ["args1", "args2"]
        parsed = generate.run_batch(configs)

    assert parsed == ["args1", "args2"]
    assert mock_parse_args.call_count == 2
    assert mock_parse_args.call_args_list[0].args[0] == (
        "--scenario sun --method LSODA --outfile sun.mp4 "
        "--visualise dot --duration 5"
    )
    assert mock_parse_args.call_args_list[1].args[0] == (
        "--scenario moon --method euler --outfile moon.csv"
    )


def test_generate_figures_uses_single_fixture_file():
    yaml_file = FIXTURES_DIR / "generate_single.yaml"

    with patch("teachgrav.generate.entry.parse_args") as mock_parse_args, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_execute:
        mock_parse_args.return_value = "parsed"
        generate.generate_figures(str(yaml_file))

    mock_parse_args.assert_called_once_with(
        "--scenario sun --method LSODA --outfile sun.mp4 --visualise dot "
        "--duration 5"
    )
    mock_execute.assert_called_once_with("parsed")


def test_generate_figures_uses_multiple_fixture_file():
    yaml_file = FIXTURES_DIR / "generate_multiple.yaml"

    with patch("teachgrav.generate.entry.parse_args") as mock_parse_args, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_execute:
        mock_parse_args.side_effect = ["first", "second"]
        generate.generate_figures(str(yaml_file))

    assert mock_parse_args.call_count == 2
    assert mock_execute.call_count == 2
    assert mock_execute.call_args_list[0].args[0] == "first"
    assert mock_execute.call_args_list[1].args[0] == "second"


def test_generate_figures_missing_yaml_file_raises():
    missing_file = FIXTURES_DIR / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        generate.generate_figures(str(missing_file))


def test_generate_figures_invalid_yaml_structure_raises():
    yaml_file = FIXTURES_DIR / "generate_invalid.yaml"

    with pytest.raises(
            ValueError,
            match="YAML root must be a list of invocation dictionaries"):
        generate.generate_figures(str(yaml_file))


def test_run_batch_rejects_non_dict_entries():
    with pytest.raises(
            ValueError,
            match="Each YAML entry must be a dictionary"):
        generate.run_batch(["not-a-dict"])


def test_run_batch_rejects_underscore_keys():
    with pytest.raises(
            ValueError,
            match="YAML keys must use '-' separators"):
        generate.run_batch([{"log_level": "INFO"}])


def test_generate_figures_train_scenario():
    """Test YAML-driven power law convergence training scenario."""
    yaml_file = FIXTURES_DIR / "generate_train.yaml"

    with patch("teachgrav.generate.entry.parse_args") as mock_parse_args, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_execute:
        mock_parse_args.return_value = "parsed_train_args"
        generate.generate_figures(str(yaml_file))

    mock_parse_args.assert_called_once()
    call_args = mock_parse_args.call_args[0][0]
    assert "--train" in call_args
    assert "--law power" in call_args
    assert "--scenario scatter" in call_args
    assert "--outfile power_law_convergence.mp4" in call_args
    assert "--show-true-law" in call_args
    assert "--log-level INFO" in call_args
    assert "--model-data" not in call_args
    mock_execute.assert_called_once_with("parsed_train_args")


# ---------------------------------------------------------------------------
# Tests for benchmark helper functions and run_benchmark
# ---------------------------------------------------------------------------

def test_parse_range_notation_integers():
    result = generate._parse_range_notation("[1:5:1]")
    assert result == [1, 2, 3, 4]


def test_parse_range_notation_floats():
    result = generate._parse_range_notation("[0.0:0.5:0.25]")
    assert len(result) == 2
    assert abs(result[0] - 0.0) < 1e-9
    assert abs(result[1] - 0.25) < 1e-9


def test_parse_range_notation_non_range_string():
    assert generate._parse_range_notation("euler") == "euler"


def test_parse_range_notation_non_string():
    assert generate._parse_range_notation(5) == 5
    assert generate._parse_range_notation([1, 2]) == [1, 2]


def test_expand_config_arrays_separates_scalars_and_lists():
    config = {
        "scenario": "moon",
        "method": ["euler", "RK45"],
        "engine": "numpy",
        "key": "my-label",
    }
    base, arrays = generate._expand_config_arrays(config)
    assert base == {"scenario": "moon", "engine": "numpy"}
    assert arrays == [("method", ["euler", "RK45"])]


def test_expand_config_arrays_expands_range_notation():
    config = {"scenario": "scatter", "n-bodies": "[2:6:2]"}
    base, arrays = generate._expand_config_arrays(config)
    assert base == {"scenario": "scatter"}
    assert arrays == [("n-bodies", [2, 4])]


def test_run_benchmark_warns_viz_options():
    configs = [{"scenario": "moon", "visualise": "dot"}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              return_value=0.1):
        mock_parse.return_value = _make_mock_args()
        with pytest.warns(UserWarning, match="visualise.*ignored"):
            generate.run_benchmark(configs)


def test_run_benchmark_single_no_sweep(capsys):
    configs = [{"scenario": "moon"}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              return_value=0.042):
        mock_parse.return_value = _make_mock_args()
        generate.run_benchmark(configs)

    out = capsys.readouterr().out
    assert "time" in out
    assert "0.042" in out


def test_run_benchmark_single_one_param(capsys):
    configs = [{"scenario": "moon", "method": ["euler", "RK45"]}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              side_effect=[0.001, 0.002]):
        mock_parse.return_value = _make_mock_args()
        generate.run_benchmark(configs)

    out = capsys.readouterr().out
    assert "method" in out
    assert "euler" in out
    assert "RK45" in out
    assert "0.001" in out
    assert "0.002" in out


def test_run_benchmark_single_two_params(capsys):
    configs = [{"scenario": "scatter", "method": ["euler", "RK45"],
                "n-bodies": [2, 4]}]
    times = [0.01, 0.02, 0.03, 0.04]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              side_effect=times):
        mock_parse.return_value = _make_mock_args(scenario='scatter')
        generate.run_benchmark(configs)

    out = capsys.readouterr().out
    # Header should contain row-param name and column values
    lines = [line for line in out.strip().split("\n") if line]
    header = lines[0]
    assert "n-bodies" in header
    assert "euler" in header
    assert "RK45" in header
    # Two data rows (n-bodies = 2 and 4)
    assert len(lines) == 3


def test_run_benchmark_warns_more_than_two_params(capsys):
    configs = [{"scenario": "scatter",
                "method": ["euler"],
                "n-bodies": [2],
                "engine": ["numpy"]}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              return_value=0.1):
        mock_parse.return_value = _make_mock_args(scenario='scatter')
        with pytest.warns(UserWarning, match="More than two array parameters"):
            generate.run_benchmark(configs)


def test_run_benchmark_multiple_scenarios(capsys):
    yaml_file = FIXTURES_DIR / "benchmark_multi.yaml"
    times = [0.1, 0.2, 0.3, 0.4]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
        patch("teachgrav.generate.entry.benchmark_scenario",
              side_effect=times):
        mock_parse.return_value = _make_mock_args()
        import yaml as _yaml
        with open(str(yaml_file)) as f:
            configs = _yaml.safe_load(f)
        generate.run_benchmark(configs)

    out = capsys.readouterr().out
    lines = [line for line in out.strip().split("\n") if line]
    # Header: method, moon, sun
    assert "moon" in lines[0]
    assert "sun" in lines[0]
    # Two data rows (one per method value)
    assert len(lines) == 3


def test_generate_figures_benchmark_mode_calls_run_benchmark():
    yaml_file = FIXTURES_DIR / "benchmark_single.yaml"
    with patch("teachgrav.generate.run_benchmark") as mock_bench:
        generate.generate_figures(str(yaml_file), benchmark=True)
    mock_bench.assert_called_once()
    configs_arg, output_arg = mock_bench.call_args.args
    assert isinstance(configs_arg, list)
    assert output_arg is None


def test_generate_figures_benchmark_output_file():
    yaml_file = FIXTURES_DIR / "benchmark_single.yaml"
    with patch("teachgrav.generate.run_benchmark") as mock_bench:
        generate.generate_figures(str(yaml_file), benchmark=True,
                                  output="/tmp/out.csv")
    _, output_arg = mock_bench.call_args.args
    assert output_arg == "/tmp/out.csv"


def test_generate_figures_warns_non_csv_output():
    yaml_file = FIXTURES_DIR / "benchmark_single.yaml"
    with patch("teachgrav.generate.run_benchmark"):
        with pytest.warns(UserWarning, match=".csv extension"):
            generate.generate_figures(str(yaml_file), benchmark=True,
                                      output="/tmp/out.txt")


def test_generate_figures_figure_output_no_warning():
    """--output pointing to a .png should not warn about .csv extension."""
    yaml_file = FIXTURES_DIR / "benchmark_single.yaml"
    import warnings as _warnings
    with patch("teachgrav.generate.run_benchmark"):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            generate.generate_figures(str(yaml_file), benchmark=True,
                                      output="/tmp/out.png")
    csv_warnings = [w for w in caught if '.csv extension' in str(w.message)]
    assert csv_warnings == []


# ---------------------------------------------------------------------------
# Tests for _is_figure_output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("out.png", True),
    ("out.svg", True),
    ("out.pdf", True),
    ("out.PNG", True),   # case-insensitive
    ("out.csv", False),
    ("", False),
    (None, False),
])
def test_is_figure_output(path, expected):
    assert generate._is_figure_output(path) is expected


# ---------------------------------------------------------------------------
# Tests for figure output routing in _write_benchmark_csv
# ---------------------------------------------------------------------------

def test_write_benchmark_csv_routes_to_figure(tmp_path):
    """_write_benchmark_csv calls _plot_benchmark_figure for .png output."""
    output = str(tmp_path / "out.png")
    headers = ["n-bodies", "numpy", "jax-cpu"]
    rows = [[2, 0.001, 0.002], [4, 0.002, 0.003]]
    with patch("teachgrav.generate._plot_benchmark_figure") as mock_plot:
        generate._write_benchmark_csv(headers, rows, output)
    mock_plot.assert_called_once_with(headers, rows, output)


def test_write_benchmark_csv_does_not_write_file_for_figure(tmp_path):
    """_write_benchmark_csv must not create a CSV file when output is .png."""
    output = str(tmp_path / "out.png")
    headers = ["n-bodies", "numpy"]
    rows = [[2, 0.001]]
    with patch("teachgrav.generate._plot_benchmark_figure"):
        generate._write_benchmark_csv(headers, rows, output)
    assert not (tmp_path / "out.png").exists()


# ---------------------------------------------------------------------------
# Tests for run_benchmark outfile auto-detection
# ---------------------------------------------------------------------------

def test_run_benchmark_uses_config_outfile_as_figure(tmp_path):
    """run_benchmark picks up outfile from config and routes to figure."""
    output_path = str(tmp_path / "bench.png")
    configs = [{"scenario": "moon", "outfile": output_path,
                "n-bodies": [2, 4]}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.benchmark_scenario",
                  return_value=0.1), \
            patch("teachgrav.generate._plot_benchmark_figure") as mock_plot:
        mock_parse.return_value = _make_mock_args()
        generate.run_benchmark(configs)
    mock_plot.assert_called_once()
    _, _, saved_path = mock_plot.call_args.args
    assert saved_path == output_path


def test_run_benchmark_explicit_output_overrides_config_outfile(tmp_path):
    """Explicit output arg takes precedence over config outfile."""
    config_outfile = str(tmp_path / "config.png")
    explicit_outfile = str(tmp_path / "explicit.png")
    configs = [{"scenario": "moon", "outfile": config_outfile}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.benchmark_scenario",
                  return_value=0.1), \
            patch("teachgrav.generate._plot_benchmark_figure") as mock_plot:
        mock_parse.return_value = _make_mock_args()
        generate.run_benchmark(configs, output=explicit_outfile)
    _, _, saved_path = mock_plot.call_args.args
    assert saved_path == explicit_outfile


def test_run_benchmark_outfile_not_warned_for_figure():
    """outfile in config should not trigger 'ignored' warning."""
    configs = [{"scenario": "moon", "outfile": "bench.png"}]
    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.benchmark_scenario",
                  return_value=0.1), \
            patch("teachgrav.generate._plot_benchmark_figure"):
        mock_parse.return_value = _make_mock_args()
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            generate.run_benchmark(configs)
    outfile_warnings = [w for w in caught
                        if 'outfile' in str(w.message).lower()
                        and 'ignored' in str(w.message).lower()]
    assert outfile_warnings == []


# ---------------------------------------------------------------------------
# Tests for generate_figures auto-routing benchmark: true configs
# ---------------------------------------------------------------------------

def test_generate_figures_auto_routes_benchmark_configs():
    """benchmark: true in YAML routes to run_benchmark."""
    yaml_file = FIXTURES_DIR / "generate_mixed_benchmark.yaml"
    with patch("teachgrav.generate.run_benchmark") as mock_bench, \
            patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_exec:
        mock_parse.return_value = "parsed"
        generate.generate_figures(str(yaml_file))
    # Only the sun scenario goes to simulation
    mock_parse.assert_called_once()
    mock_exec.assert_called_once_with("parsed")
    # benchmark config goes to run_benchmark
    mock_bench.assert_called_once()
    configs_arg = mock_bench.call_args.args[0]
    assert len(configs_arg) == 1
    assert configs_arg[0]['scenario'] == 'scatter'


def test_generate_figures_benchmark_only_yaml():
    """A YAML with only benchmark: true configs calls run_benchmark."""
    yaml_file = FIXTURES_DIR / "generate_benchmark_only.yaml"
    with patch("teachgrav.generate.run_benchmark") as mock_bench, \
            patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_exec:
        generate.generate_figures(str(yaml_file))
    mock_parse.assert_not_called()
    mock_exec.assert_not_called()
    mock_bench.assert_called_once()


def test_generate_figures_expands_non_benchmark_arrays_with_outfile_template(
):
    yaml_file = FIXTURES_DIR / "generate_templated.yaml"

    with patch("teachgrav.generate.entry.parse_args") as mock_parse, \
            patch("teachgrav.generate.entry.execute_scenario") as mock_exec:
        mock_parse.side_effect = ["args0", "args1"]
        generate.generate_figures(str(yaml_file))

    assert mock_parse.call_count == 2
    call0 = mock_parse.call_args_list[0].args[0]
    call1 = mock_parse.call_args_list[1].args[0]
    assert "--seed 0" in call0
    assert "--outfile scatter_0.mp4" in call0
    assert "--seed 1" in call1
    assert "--outfile scatter_1.mp4" in call1
    assert mock_exec.call_count == 2


def test_expand_non_benchmark_config_cartesian_product():
    config = {
        "scenario": "scatter",
        "method": ["euler", "RK45"],
        "seed": [0, 1],
        "outfile": "run_{method}_{seed}.mp4",
    }
    expanded = generate._expand_non_benchmark_config(config)
    outfiles = sorted(c["outfile"] for c in expanded)
    assert len(expanded) == 4
    assert outfiles == [
        "run_RK45_0.mp4",
        "run_RK45_1.mp4",
        "run_euler_0.mp4",
        "run_euler_1.mp4",
    ]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_mock_args(scenario='moon'):
    from unittest.mock import MagicMock
    args = MagicMock()
    args.scenario = scenario
    args.method = 'euler'
    args.engine = 'numpy'
    args.law = 'gravity'
    args.n_bodies = None
    args.seed = None
    args.benchmark = True
    return args
