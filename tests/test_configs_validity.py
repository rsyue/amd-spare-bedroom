"""Tests validating that all default YAML configurations in configs/ parse correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from amd_spare_bedroom.config import ExperimentConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
YAML_FILES = sorted(list(CONFIGS_DIR.glob("*.yaml")) + list(CONFIGS_DIR.glob("*.yml")))


def test_configs_directory_exists_and_not_empty() -> None:
    assert CONFIGS_DIR.is_dir(), f"Expected {CONFIGS_DIR} directory to exist."
    assert len(YAML_FILES) >= 6, (
        f"Expected at least 6 config files in {CONFIGS_DIR}, found {len(YAML_FILES)}."
    )


@pytest.mark.parametrize("config_path", YAML_FILES, ids=lambda p: p.name)
def test_each_yaml_config_parses_as_experiment_config(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.grid.ranks >= 1
    assert cfg.grid.total_vram_gb > 0
    assert cfg.training.epochs >= 1
    assert cfg.training.batch_size >= 1
    assert cfg.model.max_seq_length >= 1
