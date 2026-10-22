"""Hierarchical Pydantic configuration schemas and YAML loader for experiments.

Supports declarative YAML configurations, automatic mapping between flat CLI-style
keys and structured nested sections, override cascading, and environment fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from amd_spare_bedroom.config.env import get_env_float, get_env_int, get_env_str


class GridConfig(BaseModel):
    """Cluster topology, hardware budgeting, and seed configuration."""

    ranks: int = Field(default=4, ge=1, description="Number of virtual ranks to emulate.")
    total_vram_gb: float = Field(
        default_factory=lambda: get_env_float("AMD_SPARE_BEDROOM_TOTAL_VRAM_GB", 96.0),
        gt=0.0,
        description="Total physical VRAM pool budget in GB.",
    )
    device: str = Field(
        default_factory=lambda: get_env_str("AMD_SPARE_BEDROOM_DEVICE", "auto"),
        description="Execution device backend ('auto', 'cuda', 'cpu').",
    )
    seed: int = Field(
        default_factory=lambda: get_env_int("AMD_SPARE_BEDROOM_SEED", 42),
        ge=0,
        description="Deterministic random seed.",
    )


class TrainingConfig(BaseModel):
    """Optimization hyperparameters and training loop settings."""

    epochs: int = Field(default=2, ge=1, description="Number of training epochs.")
    batch_size: int = Field(default=8, ge=1, description="Per-rank mini-batch size.")
    lr: float = Field(default=2e-5, gt=0.0, description="AdamW optimizer learning rate.")
    weight_decay: float = Field(default=0.01, ge=0.0, description="AdamW weight decay.")
    max_grad_norm: float | None = Field(
        default=1.0, ge=0.0, description="Maximum gradient norm threshold for clipping."
    )
    max_steps: int = Field(
        default=5, description="Maximum training steps per epoch (<=0 for full epoch)."
    )
    amp: Literal["none", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16"] = Field(
        default="none",
        description="Automatic mixed precision mode ('none'/'fp32', 'bf16'/'bfloat16', 'fp16'/'float16').",
    )


class ModelConfig(BaseModel):
    """Model architecture and tokenization parameters."""

    model_name: str = Field(
        default="bert-base-uncased", description="Pretrained model identifier or path."
    )
    use_tiny: bool = Field(
        default=False, description="Use compact architecture for fast offline verification."
    )
    max_seq_length: int = Field(
        default=128, ge=1, description="Maximum sequence length after tokenization."
    )


class DataConfig(BaseModel):
    """Dataset selection, slicing, and synthetic generation flags."""

    task_name: str = Field(
        default="mrpc", description="GLUE benchmark task name for classification."
    )
    dataset_name: str | None = Field(
        default=None, description="HuggingFace dataset name for instruction fine-tuning."
    )
    dataset_size: int | None = Field(
        default=None, ge=1, description="Sample count cap (None for full split)."
    )
    synthetic: bool = Field(
        default=False, description="Use fast in-memory synthetic dataset for smoke tests."
    )


class ParallelismConfig(BaseModel):
    """Distributed execution paradigm."""

    mode: Literal["fsdp", "ddp"] = Field(
        default="fsdp", description="Parallelism mode ('fsdp' for sharded, 'ddp' for replicated)."
    )


class CheckpointConfig(BaseModel):
    """Best-metric model persistence and monitoring settings."""

    checkpoint_dir: str | None = Field(
        default=None, description="Directory path to save single best safetensors checkpoint."
    )
    eval_metric: str = Field(
        default="f1", description="Monitored evaluation metric for best checkpoint overwriting."
    )


class GridSearchConfig(BaseModel):
    """Multi-rank hyperparameter grid search settings."""

    enabled: bool = Field(default=False, description="Whether grid search mode is enabled.")
    params: dict[str, list[Any]] = Field(
        default_factory=dict, description="Cartesian parameter candidate mapping."
    )


# Mapping of top-level flat parameter keys to target section names
FLAT_KEY_TO_SECTION: dict[str, str] = {
    "ranks": "grid",
    "total_vram_gb": "grid",
    "device": "grid",
    "seed": "grid",
    "epochs": "training",
    "batch_size": "training",
    "lr": "training",
    "weight_decay": "training",
    "max_grad_norm": "training",
    "max_steps": "training",
    "amp": "training",
    "model_name": "model",
    "use_tiny": "model",
    "use_tiny_bert": "model",
    "use_tiny_llama": "model",
    "max_seq_length": "model",
    "seq_len": "model",
    "task_name": "data",
    "task": "data",
    "dataset_name": "data",
    "dataset_size": "data",
    "synthetic": "data",
    "use_synthetic": "data",
    "parallelism": "parallelism",
    "checkpoint_dir": "checkpoint",
    "checkpoint_path": "checkpoint",
    "eval_metric": "checkpoint",
    "grid_search": "grid_search",
}


class ExperimentConfig(BaseModel):
    """Unified top-level configuration for distributed runs and hyperparameter search."""

    grid: GridConfig = Field(default_factory=GridConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    grid_search: GridSearchConfig = Field(default_factory=GridSearchConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct an ExperimentConfig from a nested or flat dictionary."""
        sections: dict[str, dict[str, Any]] = {
            "grid": {},
            "training": {},
            "model": {},
            "data": {},
            "parallelism": {},
            "checkpoint": {},
            "grid_search": {},
        }

        # First copy any existing nested sections
        for section_name in sections:
            if section_name in data and isinstance(data[section_name], dict):
                sections[section_name].update(data[section_name])

        # Then distribute flat keys into their respective sections
        for key, value in data.items():
            if key in sections and isinstance(value, dict):
                continue

            # Handle aliases
            if key in ("use_tiny_bert", "use_tiny_llama"):
                sections["model"]["use_tiny"] = bool(value)
            elif key == "seq_len":
                sections["model"]["max_seq_length"] = value
            elif key == "task":
                sections["data"]["task_name"] = value
            elif key == "use_synthetic":
                sections["data"]["synthetic"] = bool(value)
            elif key == "parallelism" and isinstance(value, str):
                sections["parallelism"]["mode"] = value
            elif key == "checkpoint_path":
                sections["checkpoint"]["checkpoint_dir"] = value
            elif key == "grid_search" and isinstance(value, bool):
                sections["grid_search"]["enabled"] = value
            elif key in FLAT_KEY_TO_SECTION:
                target_sec = FLAT_KEY_TO_SECTION[key]
                sections[target_sec][key] = value

        return cls(
            grid=GridConfig(**sections["grid"]),
            training=TrainingConfig(**sections["training"]),
            model=ModelConfig(**sections["model"]),
            data=DataConfig(**sections["data"]),
            parallelism=ParallelismConfig(**sections["parallelism"]),
            checkpoint=CheckpointConfig(**sections["checkpoint"]),
            grid_search=GridSearchConfig(**sections["grid_search"]),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten configuration into a single dictionary matching CLI parameter names."""
        return {
            "ranks": self.grid.ranks,
            "total_vram_gb": self.grid.total_vram_gb,
            "device": self.grid.device,
            "seed": self.grid.seed,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "lr": self.training.lr,
            "weight_decay": self.training.weight_decay,
            "max_grad_norm": self.training.max_grad_norm,
            "max_steps": self.training.max_steps,
            "amp": self.training.amp,
            "model_name": self.model.model_name,
            "use_tiny": self.model.use_tiny,
            "max_seq_length": self.model.max_seq_length,
            "task_name": self.data.task_name,
            "dataset_name": self.data.dataset_name,
            "dataset_size": self.data.dataset_size,
            "synthetic": self.data.synthetic,
            "parallelism": self.parallelism.mode,
            "checkpoint_dir": self.checkpoint.checkpoint_dir,
            "eval_metric": self.checkpoint.eval_metric,
            "grid_search": self.grid_search.enabled,
        }


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update dictionary into base dictionary in-place."""
    for key, val in update.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load, parse, and validate an ExperimentConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of key-value pairs to override YAML values.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML parsing fails.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw_text) or {}
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML configuration from {file_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file {file_path} must define a YAML mapping/dictionary.")

    if overrides:
        _deep_update(parsed, overrides)

    return ExperimentConfig.from_dict(parsed)
