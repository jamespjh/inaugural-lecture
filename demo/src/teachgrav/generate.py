import argparse

import yaml

from . import entry


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


def generate_figures(yaml_file=None):
    if yaml_file is None:
        parser = argparse.ArgumentParser(
            description="Generate multiple teachgrav outputs from a YAML file")
        parser.add_argument("yaml_file", help="Path to YAML batch config")
        cli_args = parser.parse_args()
        yaml_file = cli_args.yaml_file

    with open(yaml_file, "r", encoding="utf-8") as stream:
        configs = yaml.safe_load(stream)

    parsed_args = run_batch(configs)
    for args in parsed_args:
        entry.execute_scenario(args)
