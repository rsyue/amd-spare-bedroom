"""CLI argument parser integration with YAML configuration files and environment loading.

Enables commands to accept a `--config` / `--cfg` YAML file, automatically sets parser defaults
from the loaded configuration, loads `.env` variables, and allows selective CLI flag overrides.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from amd_spare_bedroom.config.env import load_dotenv
from amd_spare_bedroom.config.settings import ExperimentConfig


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
        config_path = Path(known.config).expanduser().resolve()
        try:
            if not config_path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            raw_text = config_path.read_text(encoding="utf-8")
            raw_yaml = yaml.safe_load(raw_text) or {}
            if not isinstance(raw_yaml, dict):
                raise ValueError(
                    f"Configuration file '{config_path}' must define a YAML mapping/dictionary."
                )
            exp_config: ExperimentConfig = ExperimentConfig.from_dict(raw_yaml)
        except Exception as exc:
            parser.error(f"Error loading configuration file '{known.config}': {exc}")

        explicit_keys: set[str] = set()
        for k, v in raw_yaml.items():
            k_norm = str(k).replace("-", "_")
            explicit_keys.add(k_norm)
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    sub_norm = str(sub_k).replace("-", "_")
                    explicit_keys.add(sub_norm)
                    if sub_norm == "params" and isinstance(sub_v, dict):
                        for p_k in sub_v:
                            explicit_keys.add(f"{str(p_k).replace('-', '_')}_grid")

        # Map domain aliases into explicit keys
        if "task" in explicit_keys or "task_name" in explicit_keys:
            explicit_keys.update({"task", "task_name"})
        if any(k in explicit_keys for k in ("max_seq_length", "max_seq_len", "seq_len")):
            explicit_keys.update({"max_seq_length", "max_seq_len", "seq_len"})
        if "checkpoint_dir" in explicit_keys or "checkpoint_path" in explicit_keys:
            explicit_keys.update({"checkpoint_dir", "checkpoint_path"})
        if "synthetic" in explicit_keys or "use_synthetic" in explicit_keys:
            explicit_keys.update({"synthetic", "use_synthetic"})
        if "mode" in explicit_keys or "parallelism" in explicit_keys:
            explicit_keys.update({"mode", "parallelism"})
        if any(k in explicit_keys for k in ("use_tiny", "use_tiny_bert", "use_tiny_llama")):
            explicit_keys.update({"use_tiny", "use_tiny_bert", "use_tiny_llama"})
        if "grid_search" in explicit_keys or "enabled" in explicit_keys:
            explicit_keys.add("grid_search")

        flat_defaults = exp_config.to_flat_dict()

        # Handle grid search params mapping dynamically
        if exp_config.grid_search.params:
            for p_k, p_v in exp_config.grid_search.params.items():
                p_grid_key = f"{str(p_k).replace('-', '_')}_grid"
                if isinstance(p_v, (list, tuple)):
                    flat_defaults[p_grid_key] = ",".join(str(x) for x in p_v)
                else:
                    flat_defaults[p_grid_key] = str(p_v)

        # Match flat defaults to parser actions by destination attribute name
        known_dests = {action.dest for action in parser._actions}

        # Propagate use_tiny to destination-specific action names
        if "use_tiny" in flat_defaults and flat_defaults["use_tiny"] is not None:
            if "use_tiny_bert" in known_dests:
                flat_defaults["use_tiny_bert"] = flat_defaults["use_tiny"]
            if "use_tiny_llama" in known_dests:
                flat_defaults["use_tiny_llama"] = flat_defaults["use_tiny"]

        valid_defaults: dict[str, Any] = {}
        for k, v in flat_defaults.items():
            normalized_key = k.replace("-", "_")
            if normalized_key in explicit_keys and normalized_key in known_dests:
                valid_defaults[normalized_key] = v

        if valid_defaults:
            parser.set_defaults(**valid_defaults)

    return parser.parse_args(args)
