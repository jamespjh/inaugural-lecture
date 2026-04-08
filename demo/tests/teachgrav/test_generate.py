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
