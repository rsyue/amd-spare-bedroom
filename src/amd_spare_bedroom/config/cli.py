"""CLI argument parser integration with YAML configuration files and environment loading.

Enables commands to accept a `--config` / `--cfg` YAML file, automatically sets parser defaults
from the loaded configuration, loads `.env` variables, and allows selective CLI flag overrides.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from amd_spare_bedroom.config.env import load_dotenv
from amd_spare_bedroom.config.settings import ExperimentConfig, load_config


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Ensure --config / --cfg argument is registered in the given parser."""
    existing_options = {opt for action in parser._actions for opt in action.option_strings}
    if "--config" not in existing_options and "--cfg" not in existing_options:
        parser.add_argument(
            "--config",
            "--cfg",
            type=str,
            default=None,
            dest="config",
            help="Path to YAML configuration file containing experiment parameters",
        )


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    args: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command line arguments with optional YAML configuration loading and overrides.

    1. Ensures `--config` / `--cfg` argument exists in parser.
    2. Loads `.env` file into `os.environ` if present.
    3. Pre-parses `--config` flag from arguments.
    4. If `--config` is provided, loads and validates the YAML file, applying values as parser defaults.
    5. Fully parses arguments so explicit CLI flags override configuration file values.

    Args:
        parser: Configured `argparse.ArgumentParser` instance.
        args: Sequence of argument strings, or None to read from `sys.argv[1:]`.

    Returns:
        Populated `argparse.Namespace` with config values and CLI overrides merged.
    """
    add_config_argument(parser)
    load_dotenv()

    # Pre-parse specifically for --config / --cfg
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "--cfg", type=str, default=None, dest="config")
    known, _ = pre_parser.parse_known_args(args)

    if known.config:
        config_path = Path(known.config)
        exp_config: ExperimentConfig = load_config(config_path)
        flat_defaults = exp_config.to_flat_dict()

        # Handle grid search params mapping if specified
        if exp_config.grid_search.params:
            if "lr" in exp_config.grid_search.params:
                flat_defaults["lr_grid"] = ",".join(
                    str(x) for x in exp_config.grid_search.params["lr"]
                )
            if "weight_decay" in exp_config.grid_search.params:
                flat_defaults["weight_decay_grid"] = ",".join(
                    str(x) for x in exp_config.grid_search.params["weight_decay"]
                )

        # Match flat defaults to parser actions by destination attribute name
        known_dests = {action.dest for action in parser._actions}
        valid_defaults: dict[str, Any] = {}
        for k, v in flat_defaults.items():
            normalized_key = k.replace("-", "_")
            if normalized_key in known_dests and v is not None:
                valid_defaults[normalized_key] = v

        if valid_defaults:
            parser.set_defaults(**valid_defaults)

    return parser.parse_args(args)
