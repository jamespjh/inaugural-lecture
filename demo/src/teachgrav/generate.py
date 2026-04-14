import argparse
import csv
import re
import sys
import warnings

import numpy as np
import yaml

from . import entry

_VIZ_KEYS = {'visualise', 'video', 'outfile', 'format', 'duration'}


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
    return base_config, array_params


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
    """Run benchmark sweeps defined in YAML configs and write CSV output.

    For a single scenario config, array parameters define the sweep dimensions:
      - First array parameter  -> columns
      - Second array parameter -> rows

    For multiple scenario configs, scenarios form the column axis (labeled by
    the 'key' field), and the first array parameter of each scenario forms the
    row axis.

    Visualization options present in any config trigger a warning and are
    ignored during benchmarking.

    Args:
        configs: list of invocation dicts (parsed from YAML)
        output:  path to output CSV file, or None to write to stdout
    """
    if not isinstance(configs, list):
        raise ValueError("YAML root must be a list of invocation dictionaries")
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("Each YAML entry must be a dictionary")

    seen_viz_keys = set()
    for config in configs:
        for key in config:
            if key in _VIZ_KEYS and key not in seen_viz_keys:
                seen_viz_keys.add(key)
                warnings.warn(
                    f"Visualization option '{key}' is ignored "
                    "in benchmark mode.",
                    UserWarning,
                    stacklevel=2,
                )

    if len(configs) > 1:
        _run_multi_scenario_benchmark(configs, output)
    else:
        _run_single_scenario_benchmark(configs[0], output)


def _run_multi_scenario_benchmark(configs, output):
    """Handle benchmarking when multiple scenario configs are given.

    Scenarios become columns (labeled by 'key'); the first array parameter
    of each scenario becomes the row axis.
    """
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

    _write_benchmark_csv(headers, rows, output)


def _run_single_scenario_benchmark(config, output):
    """Handle benchmarking for a single scenario config."""
    base_config, array_params = _expand_config_arrays(config)

    if len(array_params) == 0:
        args = _build_benchmark_args(base_config, {})
        time_val = entry.benchmark_scenario(args)
        _write_benchmark_csv(['time'], [[time_val]], output)

    elif len(array_params) == 1:
        param_name, param_values = array_params[0]
        headers = [param_name, 'time']
        rows = []
        for val in param_values:
            args = _build_benchmark_args(base_config, {param_name: val})
            rows.append([val, entry.benchmark_scenario(args)])
        _write_benchmark_csv(headers, rows, output)

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
            help="Output CSV file for benchmark results (default: stdout)")
        cli_args = parser.parse_args()
        yaml_file = cli_args.yaml_file
        benchmark = cli_args.benchmark
        output = cli_args.output

    if benchmark and output and not output.endswith('.csv'):
        warnings.warn(
            f"Output file '{output}' does not have a .csv extension. "
            "Results may not be formatted correctly.",
            UserWarning,
        )

    with open(yaml_file, "r", encoding="utf-8") as stream:
        configs = yaml.safe_load(stream)

    if benchmark:
        run_benchmark(configs, output)
    else:
        parsed_args = run_batch(configs)
        for args in parsed_args:
            entry.execute_scenario(args)
