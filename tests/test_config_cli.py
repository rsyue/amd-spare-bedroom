"""Unit tests for CLI parser integration with YAML --config and flag overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from amd_spare_bedroom.config.cli import parse_args_with_config


def test_cli_without_config_uses_defaults() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parse_args_with_config(parser, ["--batch-size", "12"])
    assert args.ranks == 4
    assert args.batch_size == 12
    assert args.lr == 2e-5


def test_cli_with_config_loads_values(tmp_path: Path) -> None:
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(
        """
        ranks: 8
        batch_size: 16
        lr: 1e-4
        """,
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parse_args_with_config(parser, ["--config", str(yaml_file)])
    assert args.ranks == 8
    assert args.batch_size == 16
    assert args.lr == 1e-4


def test_cli_with_config_and_single_flag_override(tmp_path: Path) -> None:
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(
        """
        ranks: 8
        batch_size: 16
        lr: 1e-4
        """,
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    # User passes --config custom.yaml --batch-size 32
    args = parse_args_with_config(parser, ["--config", str(yaml_file), "--batch-size", "32"])
    assert args.ranks == 8
    assert args.batch_size == 32  # Overridden!
    assert args.lr == 1e-4  # Preserved from config!


def test_cli_with_cfg_alias(tmp_path: Path) -> None:
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("ranks: 10\n", encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)

    args = parse_args_with_config(parser, ["--cfg", str(yaml_file)])
    assert args.ranks == 10


def test_train_bert_glue_parse_args_with_config(tmp_path: Path) -> None:
    from examples.train_bert_glue import parse_args

    yaml_file = tmp_path / "bert_test.yaml"
    yaml_file.write_text(
        """
        ranks: 6
        batch_size: 16
        lr: 3e-5
        epochs: 4
        parallelism: ddp
        task_name: sst2
        """,
        encoding="utf-8",
    )

    # 1. Load entirely from yaml
    args = parse_args(["--config", str(yaml_file)])
    assert args.ranks == 6
    assert args.batch_size == 16
    assert args.lr == 3e-5
    assert args.epochs == 4
    assert args.parallelism == "ddp"
    assert args.task_name == "sst2"

    # 2. Override single parameter via CLI
    args_overridden = parse_args(["--config", str(yaml_file), "--batch-size", "32"])
    assert args_overridden.ranks == 6
    assert args_overridden.batch_size == 32
    assert args_overridden.lr == 3e-5


def test_train_llama_sft_parse_args_with_config(tmp_path: Path) -> None:
    from examples.train_llama_sft import parse_args

    yaml_file = tmp_path / "llama_test.yaml"
    yaml_file.write_text(
        """
        ranks: 8
        epochs: 3
        batch_size: 2
        lr: 1e-5
        parallelism: ddp
        """,
        encoding="utf-8",
    )

    args = parse_args(["--config", str(yaml_file), "--lr", "5e-5"])
    assert args.ranks == 8
    assert args.epochs == 3
    assert args.batch_size == 2
    assert args.lr == 5e-5  # Overridden!
    assert args.parallelism == "ddp"
    # Ensure unmentioned script defaults are not stomped by ExperimentConfig defaults
    assert args.model_name == "meta-llama/Llama-3.2-1B"
    assert args.eval_metric == "perplexity"


def test_use_tiny_yaml_propagates_to_bert_and_llama(tmp_path: Path) -> None:
    from examples.train_bert_glue import parse_args as parse_bert_args
    from examples.train_llama_sft import parse_args as parse_llama_args

    yaml_file = tmp_path / "tiny_config.yaml"
    yaml_file.write_text(
        """
        model:
          use_tiny: true
        """,
        encoding="utf-8",
    )

    bert_args = parse_bert_args(["--config", str(yaml_file)])
    assert bert_args.use_tiny_bert is True

    llama_args = parse_llama_args(["--config", str(yaml_file)])
    assert llama_args.use_tiny_llama is True


def test_cli_config_error_handling(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)

    # 1. Non-existent file
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, ["--config", str(tmp_path / "nonexistent.yaml")])

    # 2. Corrupt YAML
    corrupt_file = tmp_path / "corrupt.yaml"
    corrupt_file.write_text("invalid: yaml: [", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, ["--config", str(corrupt_file)])
