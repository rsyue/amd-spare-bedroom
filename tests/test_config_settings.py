"""Unit tests for YAML configuration schemas, loading, and merging."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from amd_spare_bedroom.config.settings import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    GridConfig,
    GridSearchConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
    load_config,
)


def test_individual_config_classes() -> None:
    g = GridConfig(ranks=8, total_vram_gb=48.0, device="cpu", seed=10)
    assert g.ranks == 8
    assert g.total_vram_gb == 48.0
    assert g.device == "cpu"
    assert g.seed == 10

    t = TrainingConfig(epochs=4, batch_size=16, lr=1e-4, amp="fp16")
    assert t.epochs == 4
    assert t.batch_size == 16
    assert t.lr == 1e-4
    assert t.amp == "fp16"

    m = ModelConfig(model_name="bert-large-uncased", use_tiny=True, max_seq_length=256)
    assert m.model_name == "bert-large-uncased"
    assert m.use_tiny is True
    assert m.max_seq_length == 256

    d = DataConfig(task_name="sst2", dataset_size=100, synthetic=True)
    assert d.task_name == "sst2"
    assert d.dataset_size == 100
    assert d.synthetic is True

    p = ParallelismConfig(mode="ddp")
    assert p.mode == "ddp"

    c = CheckpointConfig(checkpoint_dir="out/best", eval_metric="accuracy")
    assert c.checkpoint_dir == "out/best"
    assert c.eval_metric == "accuracy"

    s = GridSearchConfig(enabled=True, params={"lr": [1e-5]})
    assert s.enabled is True
    assert s.params == {"lr": [1e-5]}


def test_experiment_config_defaults() -> None:
    cfg = ExperimentConfig()
    assert cfg.grid.ranks == 4
    assert cfg.grid.total_vram_gb == 96.0
    assert cfg.training.batch_size == 8
    assert cfg.training.lr == 2e-5
    assert cfg.parallelism.mode == "fsdp"
    assert cfg.grid_search.enabled is False
    assert cfg.grid_search.params == {}


def test_experiment_config_flat_mapping() -> None:
    data = {
        "ranks": 8,
        "batch_size": 16,
        "lr": 1e-4,
        "parallelism": "ddp",
        "task_name": "sst2",
        "checkpoint_dir": "checkpoints/my_run",
    }
    cfg = ExperimentConfig.from_dict(data)
    assert cfg.grid.ranks == 8
    assert cfg.training.batch_size == 16
    assert cfg.training.lr == 1e-4
    assert cfg.parallelism.mode == "ddp"
    assert cfg.data.task_name == "sst2"
    assert cfg.checkpoint.checkpoint_dir == "checkpoints/my_run"


def test_experiment_config_nested_sections() -> None:
    data = {
        "grid": {"ranks": 2, "device": "cuda"},
        "training": {"epochs": 5, "amp": "bf16"},
        "model": {"model_name": "meta-llama/Llama-3.2-1B", "use_tiny": True},
        "grid_search": {
            "enabled": True,
            "params": {"lr": [1e-5, 2e-5], "batch_size": [4, 8]},
        },
    }
    cfg = ExperimentConfig.from_dict(data)
    assert cfg.grid.ranks == 2
    assert cfg.grid.device == "cuda"
    assert cfg.training.epochs == 5
    assert cfg.training.amp == "bf16"
    assert cfg.model.model_name == "meta-llama/Llama-3.2-1B"
    assert cfg.model.use_tiny is True
    assert cfg.grid_search.enabled is True
    assert cfg.grid_search.params == {"lr": [1e-5, 2e-5], "batch_size": [4, 8]}


def test_load_config_from_yaml_file(tmp_path: Path) -> None:
    yaml_content = """
    ranks: 4
    epochs: 3
    batch_size: 4
    lr: 3e-5
    parallelism: fsdp
    task_name: mrpc
    eval_metric: f1
    grid_search:
      enabled: false
    """
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(yaml_file)
    assert cfg.grid.ranks == 4
    assert cfg.training.epochs == 3
    assert cfg.training.batch_size == 4
    assert cfg.training.lr == 3e-5
    assert cfg.parallelism.mode == "fsdp"
    assert cfg.data.task_name == "mrpc"
    assert cfg.checkpoint.eval_metric == "f1"


def test_load_config_with_overrides(tmp_path: Path) -> None:
    yaml_content = """
    ranks: 4
    batch_size: 8
    lr: 2e-5
    """
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(yaml_file, overrides={"batch_size": 16, "lr": 5e-5})
    assert cfg.training.batch_size == 16
    assert cfg.training.lr == 5e-5
    assert cfg.grid.ranks == 4


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/non/existent/file.yaml")


def test_experiment_config_to_flat_dict() -> None:
    cfg = ExperimentConfig()
    flat = cfg.to_flat_dict()
    assert flat["ranks"] == 4
    assert flat["batch_size"] == 8
    assert flat["lr"] == 2e-5
    assert flat["parallelism"] == "fsdp"
    assert flat["task_name"] == "mrpc"


def test_validation_errors() -> None:
    with pytest.raises(ValidationError):
        # ranks must be positive
        ExperimentConfig(grid=GridConfig(ranks=0))

    with pytest.raises(ValidationError):
        # lr must be positive
        ExperimentConfig(training=TrainingConfig(lr=-1.0))

    with pytest.raises(ValidationError):
        # amp must be valid literal
        TrainingConfig(amp="unsupported_amp")  # type: ignore[arg-type]


def test_load_config_with_nested_overrides_preserves_siblings(tmp_path: Path) -> None:
    yaml_content = """
    training:
      epochs: 4
      batch_size: 16
      amp: bf16
    """
    yaml_file = tmp_path / "test_nested.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(yaml_file, overrides={"training": {"lr": 1e-4}})
    assert cfg.training.epochs == 4
    assert cfg.training.batch_size == 16
    assert cfg.training.amp == "bf16"
    assert cfg.training.lr == 1e-4


def test_grid_config_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMD_SPARE_BEDROOM_TOTAL_VRAM_GB", "48.0")
    monkeypatch.setenv("AMD_SPARE_BEDROOM_DEVICE", "cpu")
    monkeypatch.setenv("AMD_SPARE_BEDROOM_SEED", "123")

    grid = GridConfig()
    assert grid.total_vram_gb == 48.0
    assert grid.device == "cpu"
    assert grid.seed == 123
