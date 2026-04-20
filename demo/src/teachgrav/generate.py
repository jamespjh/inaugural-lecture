import argparse
import csv
import itertools
import math
import re
import sys
import warnings

import numpy as np
import yaml

from . import entry
from .engine_support import get_available_engines
from .visualisations.visualize import figsize_from_aspect

_VIZ_KEYS = {'visualise', 'video', 'outfile', 'format', 'duration'}
_FIGURE_EXTENSIONS = frozenset({'.png', '.svg', '.pdf'})

# Fixed colour palette keyed by engine name.
_ENGINE_COLORS = {
    'numpy': '#4878d0',
    'jax-cpu': '#ee854a',
    'jax-gpu': '#6acc65',
    'jax-metal': '#d65f5f',
    'mlx-cpu': '#956cb4',
    'mlx-gpu': '#8c613c',
    'cupy': '#dc7ec0',
    'torch-cpu': '#797979',
    'torch-gpu': '#d5bb67',
    'torch-mps': '#82c6e2',
}


def _is_figure_output(path):
    """Return True if *path* looks like a figure file path."""
    if not path:
        return False
    ext = '.' + path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    return ext in _FIGURE_EXTENSIONS


def _config_to_force_args(config):
    parts = []
    for key, value in config.items():
        if "_" in key:
            raise ValueError(
                "YAML keys must use '-' separators (for example 'log-level')")

        if value is None:
            continue

        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                parts.append(option)
            continue

        parts.append(option)
        parts.append(str(value))
    return " ".join(parts)


def run_batch(configs):
    if not isinstance(configs, list):
        raise ValueError("YAML root must be a list of invocation dictionaries")

    parsed_args = []
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("Each YAML entry must be a dictionary")
        force_args = _config_to_force_args(config)
        parsed_args.append(entry.parse_args(force_args))
    return parsed_args


def _parse_range_notation(value):
    """Parse [start:stop:step] string notation into a list of values.

    Returns a list if the value matches range notation, otherwise returns
    the original value unchanged.
    """
    if not isinstance(value, str):
        return value
    match = re.match(r'^\[([^:]+):([^:]+):([^:]+)\]$', value.strip())
    if not match:
        return value
    start = float(match.group(1))
    stop = float(match.group(2))
    step = float(match.group(3))
    values = list(np.arange(start, stop, step))
    if values and all(v == int(v) for v in values):
        values = [int(v) for v in values]
    return values


def _expand_config_arrays(config):
    """Separate array and scalar parameters in a config dict.

    Skips the special 'key' field used for scenario labeling.
    Expands [start:stop:step] notation in string values.

    Returns:
        base_config: dict of scalar (non-array) parameters
        array_params: ordered list of (key, values) for list parameters
    """
    base_config = {}
    array_params = []
    for key, value in config.items():
        if key == 'key':
            continue
        parsed = (
            _parse_range_notation(value)
            if isinstance(value, str) else value
        )
        if isinstance(parsed, list):
            array_params.append((key, parsed))
        else:
            base_config[key] = parsed
        if key == 'engine' and parsed == "ALL":
            array_params.append((key, get_available_engines()))
    return base_config, array_params


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def _expand_non_benchmark_config(config):
    """Expand one non-benchmark config into concrete invocations.

    Array-valued keys (including range notation) are expanded by Cartesian
    product. If ``outfile`` contains ``{key}`` placeholders, they are filled
    from the expanded parameter values.
    """
    base_config, array_params = _expand_config_arrays(config)
    if not array_params:
        return [base_config]

    keys = [k for k, _ in array_params]
    values_product = itertools.product(*(vals for _, vals in array_params))
    expanded = []
    for combo in values_product:
        params = dict(zip(keys, combo))
        config_for_run = dict(base_config)
        config_for_run.update(params)

        outfile = config_for_run.get('outfile')
        if isinstance(outfile, str):
            format_values = dict(config_for_run)
            config_for_run['outfile'] = outfile.format_map(
                _SafeFormatDict(format_values)
            )

        expanded.append(config_for_run)

    return expanded


def _build_benchmark_args(base_config, override_params):
    """Build parsed args for a single benchmark run.

    Strips visualization options, merges overrides, and adds benchmark=True.
    """
    config_for_run = {k: v for k, v in base_config.items()
                      if k not in _VIZ_KEYS}
    config_for_run.update(override_params)
    config_for_run['benchmark'] = True
    force_args = _config_to_force_args(config_for_run)
    return entry.parse_args(force_args)


def _plot_benchmark_figure(headers, rows, output_path, figsize):
    """Save a log-log benchmark figure to *output_path*.

    *headers* is ``[x_label, series1, series2, ...]``.
    *rows* is ``[[x_val, t1, t2, ...], ...]``.
    *figsize* is ``(width_inches, height_inches)`` (default: half a 16:9
    projector column).
    """
    import matplotlib.pyplot as plt

    x_label = headers[0]
    series_labels = headers[1:]

    x_vals = [float(row[0]) for row in rows]

    fig, ax = plt.subplots(figsize=figsize)
    for col_idx, label in enumerate(series_labels, start=1):
        times = []
        for row in rows:
            try:
                t = float(row[col_idx])
            except (ValueError, IndexError):
                t = float('nan')
            times.append(t)
        pairs = [(x, t) for x, t in zip(x_vals, times)
                 if not math.isnan(t) and t > 0]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        color = _ENGINE_COLORS.get(label)
        ax.loglog(xs, ys, 'o-', label=label, color=color,
                  linewidth=2, markersize=5)

    ax.set_xlabel(x_label.replace('-', ' ').title(), fontsize=13)
    ax.set_ylabel('Time per simulation step (s)', fontsize=13)
    ax.set_title('Gravity simulation performance by engine', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    if x_vals:
        ax.set_xlim(min(x_vals) * 0.8, max(x_vals) * 1.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')


def _write_benchmark_csv(headers, rows, output=None):
    """Write benchmark results as CSV to a file or stdout."""
    if output:
        with open(output, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def run_benchmark(configs, output=None):
    """Run benchmark sweeps defined in YAML configs and write results.

    For a single scenario config, array parameters define the sweep dimensions:
      - First array parameter  -> columns
      - Second array parameter -> rows

    For multiple scenario configs, scenarios form the column axis (labeled by
    the 'key' field), and the first array parameter of each scenario forms the
    row axis.

    If the config contains ``outfile`` pointing to a figure path (e.g.
    ``.png``), a log-log figure is generated instead of CSV output.

    Visualization options present in any config trigger a warning and are
    ignored during benchmarking (except ``outfile`` for figure output).

    Args:
        configs: list of invocation dicts (parsed from YAML)
        output:  path to output file (CSV or figure), or None for stdout
    """
    if not isinstance(configs, list):
        raise ValueError("YAML root must be a list of invocation dictionaries")
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("Each YAML entry must be a dictionary")

    # Extract figure output from config outfile when no explicit output given
    effective_output = output
    if effective_output is None:
        for config in configs:
            outfile = config.get('outfile', '')
            if _is_figure_output(outfile):
                effective_output = outfile
                break

    seen_viz_keys = set()
    for config in configs:
        for key in config:
            if key in _VIZ_KEYS and key not in seen_viz_keys:
                # outfile is not ignored when it's being used for figure output
                if key == 'outfile' and _is_figure_output(effective_output):
                    continue
                seen_viz_keys.add(key)
                warnings.warn(
                    f"Visualization option '{key}' is ignored "
                    "in benchmark mode.",
                    UserWarning,
                    stacklevel=2,
                )

    if len(configs) > 1:
        _run_multi_scenario_benchmark(configs, effective_output)
    else:
        _run_single_scenario_benchmark(configs[0], effective_output)


def _run_multi_scenario_benchmark(configs, output):
    """Handle benchmarking when multiple scenario configs are given.

    Scenarios become columns (labeled by 'key'); the first array parameter
    of each scenario becomes the row axis.
    """
    figsize = figsize_from_aspect(configs[0].get('aspect'))
    scenario_labels = []
    scenario_base_configs = []
    scenario_array_params = []

    for i, config in enumerate(configs):
        label = config.get('key', config.get('scenario', f'scenario_{i}'))
        scenario_labels.append(label)
        base_config, array_params = _expand_config_arrays(config)
        scenario_base_configs.append(base_config)
        scenario_array_params.append(array_params)

    if scenario_array_params[0]:
        row_param_name, row_values = scenario_array_params[0][0]
    else:
        row_param_name, row_values = None, [None]

    headers = [row_param_name or ''] + scenario_labels
    rows = []
    for row_val in row_values:
        row = [row_val]
        for base_config, array_params in zip(
                scenario_base_configs, scenario_array_params):
            override = {}
            if row_val is not None and array_params:
                override[array_params[0][0]] = row_val
            args = _build_benchmark_args(base_config, override)
            row.append(entry.benchmark_scenario(args))
        rows.append(row)

    if _is_figure_output(output):
        _plot_benchmark_figure(headers, rows, output, figsize=figsize)
    else:
        _write_benchmark_csv(headers, rows, output)


def _run_single_scenario_benchmark(config, output):
    """Handle benchmarking for a single scenario config."""
    figsize = figsize_from_aspect(config.get('aspect'))
    base_config, array_params = _expand_config_arrays(config)

    if len(array_params) == 0:
        args = _build_benchmark_args(base_config, {})
        time_val = entry.benchmark_scenario(args)
        headers, rows = ['time'], [[time_val]]
    elif len(array_params) == 1:
        param_name, param_values = array_params[0]
        headers = [param_name, 'time']
        rows = []
        for val in param_values:
            args = _build_benchmark_args(base_config, {param_name: val})
            rows.append([val, entry.benchmark_scenario(args)])
    else:
        if len(array_params) > 2:
            warnings.warn(
                "More than two array parameters given. "
                "Only the first two will be used for the benchmark table.",
                UserWarning,
            )
        col_param_name, col_values = array_params[0]  # first  -> columns
        row_param_name, row_values = array_params[1]  # second -> rows

        headers = [row_param_name] + [str(v) for v in col_values]
        rows = []
        for row_val in row_values:
            row = [row_val]
            for col_val in col_values:
                override = {col_param_name: col_val, row_param_name: row_val}
                args = _build_benchmark_args(base_config, override)
                row.append(entry.benchmark_scenario(args))
            rows.append(row)

    if _is_figure_output(output):
        _plot_benchmark_figure(headers, rows, output, figsize=figsize)
    else:
        _write_benchmark_csv(headers, rows, output)


def generate_figures(yaml_file=None, benchmark=False, output=None):
    if yaml_file is None:
        parser = argparse.ArgumentParser(
            description="Generate multiple teachgrav outputs from a YAML file")
        parser.add_argument("yaml_file", help="Path to YAML batch config")
        parser.add_argument(
            "--benchmark",
            action='store_true',
            help="Run in benchmark mode, outputting CSV timing results")
        parser.add_argument(
            "--output",
            default=None,
            help="Output file for benchmark results (CSV or figure path)")
        cli_args = parser.parse_args()
        yaml_file = cli_args.yaml_file
        benchmark = cli_args.benchmark
        output = cli_args.output

    if (benchmark and output
            and not output.endswith('.csv')
            and not _is_figure_output(output)):
        warnings.warn(
            f"Output file '{output}' does not have a .csv extension. "
            "Results may not be formatted correctly.",
            UserWarning,
        )

    with open(yaml_file, "r", encoding="utf-8") as stream:
        configs = yaml.safe_load(stream)

    if not isinstance(configs, list):
        raise ValueError("YAML root must be a list of invocation dictionaries")

    if benchmark:
        run_benchmark(configs, output)
    else:
        # Separate simulation configs from benchmark sweep configs
        sim_configs = [c for c in configs if not c.get('benchmark')]
        bm_configs = [c for c in configs if c.get('benchmark')]

        if sim_configs:
            expanded_sim_configs = []
            for config in sim_configs:
                expanded_sim_configs.extend(_expand_non_benchmark_config(
                    config))
            parsed_args = run_batch(expanded_sim_configs)
            for args in parsed_args:
                entry.execute_scenario(args)

        if bm_configs:
            run_benchmark(bm_configs)
