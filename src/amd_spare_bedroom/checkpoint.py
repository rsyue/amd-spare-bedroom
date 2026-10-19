"""Single best-metric HuggingFace safetensors + tokenizer checkpoint saving."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

MetricDirection = Literal["max", "min"]
MetricMode = Literal["auto", "max", "min"]


@dataclass(frozen=True)
class _MetricSpec:
    direction: MetricDirection
    label: str


_METRIC_SPECS: dict[str, _MetricSpec] = {
    "f1": _MetricSpec("max", "F1"),
    "accuracy": _MetricSpec("max", "Accuracy"),
    "precision": _MetricSpec("max", "Precision"),
    "recall": _MetricSpec("max", "Recall"),
    "token_accuracy": _MetricSpec("max", "Token Accuracy"),
    "loss": _MetricSpec("min", "Loss"),
    "eval_loss": _MetricSpec("min", "Eval Loss"),
    "perplexity": _MetricSpec("min", "Perplexity"),
}

METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    name: spec.direction for name, spec in _METRIC_SPECS.items()
}


@dataclass(frozen=True)
class _BestRecord:
    score: float
    epoch: int
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]


def _format_hparam_value(val: Any) -> str:
    """Format a hyperparameter number concisely (e.g., 1e-05 -> '1e-5', 1e-4 -> '1e-4', 0.01 -> '0.01')."""
    if isinstance(val, float):
        if not math.isfinite(val):
            return str(val)
        if 0 < abs(val) < 0.001 or abs(val) >= 1e5:
            text = f"{val:e}".replace("e-0", "e-").replace("e+0", "e+")
            base, exp = text.split("e")
            base = base.rstrip("0").rstrip(".")
            return f"{base}e{exp}"
        text = f"{val:g}"
        return text.replace("e-0", "e-").replace("e+0", "e+")
    return str(val)


class BestCheckpointSaver:
    """Manages a single best-metric HuggingFace safetensors + tokenizer checkpoint directory.

    Overwrites the target checkpoint directory atomically when and only when the
    monitored evaluation metric improves over the best previously recorded score.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        metric_name: str = "f1",
        mode: MetricMode = "auto",
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the single best-metric checkpoint manager.

        Args:
            checkpoint_dir: Target directory path where the best checkpoint is stored.
            metric_name: Evaluation metric to track (e.g. 'f1', 'accuracy', 'perplexity').
            mode: Comparison mode: 'auto' (inferred from metric name), 'max', or 'min'.
            tokenizer: Optional tokenizer instance to serialize alongside model weights.
        """
        path_str = str(checkpoint_dir).strip()
        if not path_str:
            raise ValueError("checkpoint_dir must be a non-empty directory path.")
        resolved = Path(checkpoint_dir).resolve()
        if resolved == Path.cwd():
            raise ValueError(
                f"checkpoint_dir cannot be the current working directory ({resolved}). "
                "Specify a dedicated sub-directory (e.g. 'checkpoints/best_model')."
            )
        if resolved == resolved.parent or resolved == Path.home():
            raise ValueError(
                f"checkpoint_dir cannot be the filesystem root or home directory ({resolved})."
            )
        self.checkpoint_dir = resolved
        self.metric_name = metric_name.strip().lower()
        if mode == "auto":
            if self.metric_name not in METRIC_DIRECTIONS:
                raise ValueError(
                    f"Unknown metric {metric_name!r} for mode='auto'. "
                    f"Known metrics: {sorted(METRIC_DIRECTIONS.keys())}"
                )
            self.mode: MetricDirection = METRIC_DIRECTIONS[self.metric_name]
        elif mode in ("max", "min"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be 'auto', 'max', or 'min', got {mode!r}")

        self.tokenizer = tokenizer
        self._best: _BestRecord | None = None

    @property
    def best_score(self) -> float | None:
        """Return the best recorded metric score, or None if no checkpoint saved yet."""
        return self._best.score if self._best is not None else None

    @property
    def best_epoch(self) -> int | None:
        """Return the epoch index of the best recorded checkpoint."""
        return self._best.epoch if self._best is not None else None

    @property
    def best_hyperparameters(self) -> dict[str, Any] | None:
        """Return the hyperparameters associated with the best recorded checkpoint."""
        return dict(self._best.hyperparameters) if self._best is not None else None

    def is_better(self, score: float) -> bool:
        """Return True if score strictly improves upon the best recorded score (rejecting NaN/Inf)."""
        if not math.isfinite(score):
            return False
        if self._best is None:
            return True
        if self.mode == "max":
            return score > self._best.score
        return score < self._best.score

    def _extract_metric_score(self, metrics_dict: Mapping[str, float]) -> float:
        norm_map = {
            str(k).strip().lower(): float(v) for k, v in metrics_dict.items() if v is not None
        }
        if self.metric_name in norm_map:
            return float(norm_map[self.metric_name])
        if self.metric_name == "eval_loss" and "loss" in norm_map:
            return float(norm_map["loss"])
        if self.metric_name == "loss" and "eval_loss" in norm_map:
            return float(norm_map["eval_loss"])
        raise KeyError(
            f"Monitored metric {self.metric_name!r} not found in metrics {list(metrics_dict.keys())}"
        )

    def save_if_best(
        self,
        model: Any,
        metrics: Mapping[str, float] | Any,
        epoch: int,
        hyperparameters: Mapping[str, Any] | None = None,
        tokenizer: Any | None = None,
    ) -> bool:
        """Save HuggingFace safetensors model, tokenizer, and metadata if metric improved."""
        if hasattr(metrics, "to_dict"):
            raw_dict = metrics.to_dict()
        elif isinstance(metrics, Mapping):
            raw_dict = dict(metrics)
        else:
            raise TypeError(f"Unsupported metrics object: {type(metrics)!r}")

        metrics_dict: dict[str, float] = {
            str(k).strip().lower(): float(v) for k, v in raw_dict.items() if v is not None
        }

        score = self._extract_metric_score(metrics_dict)
        if not self.is_better(score):
            return False

        raw_model = model.unwrap_model() if hasattr(model, "unwrap_model") else model
        active_tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        hparams_dict = dict(hyperparameters) if hyperparameters is not None else {}

        self.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.checkpoint_dir.name}.staging_",
                dir=self.checkpoint_dir.parent,
            )
        )
        backup_dir = self.checkpoint_dir.parent / f".{self.checkpoint_dir.name}.backup"

        try:
            if hasattr(raw_model, "save_pretrained"):
                raw_model.save_pretrained(staging_dir, safe_serialization=True)
            else:
                import safetensors.torch

                safetensors.torch.save_file(
                    raw_model.state_dict(), staging_dir / "model.safetensors"
                )

            if active_tokenizer is not None:
                if hasattr(active_tokenizer, "save_pretrained"):
                    active_tokenizer.save_pretrained(staging_dir)
                elif hasattr(active_tokenizer, "save"):
                    active_tokenizer.save(str(staging_dir / "tokenizer.json"))

            metadata = {
                "metric_name": self.metric_name,
                "best_score": score,
                "epoch": epoch,
                "metrics": metrics_dict,
                "hyperparameters": hparams_dict,
            }
            (staging_dir / "checkpoint_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            if self.checkpoint_dir.exists():
                self.checkpoint_dir.replace(backup_dir)
            staging_dir.replace(self.checkpoint_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
        except BaseException:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if not self.checkpoint_dir.exists() and backup_dir.exists():
                backup_dir.replace(self.checkpoint_dir)
            raise

        self._best = _BestRecord(
            score=score,
            epoch=epoch,
            metrics=metrics_dict,
            hyperparameters=hparams_dict,
        )
        return True

    def load_metadata(self) -> dict[str, Any]:
        """Load checkpoint_metadata.json from the saved checkpoint directory."""
        meta_path = self.checkpoint_dir / "checkpoint_metadata.json"
        return cast(dict[str, Any], json.loads(meta_path.read_text(encoding="utf-8")))

    @staticmethod
    def format_winner(
        hyperparameters: Mapping[str, Any],
        metric_name: str,
        best_score: float,
    ) -> str:
        """Format the canonical winning hyperparameter combination announcement line.

        Example:
            'LR of 1e-5 and weight decay of 0.01 wins at 0.91 F1'
        """
        spec = _METRIC_SPECS.get(metric_name.lower())
        metric_label = spec.label if spec is not None else metric_name.upper()

        hparams = dict(hyperparameters)
        lr_key = (
            "lr" if "lr" in hparams else ("learning_rate" if "learning_rate" in hparams else None)
        )
        wd_key = "weight_decay" if "weight_decay" in hparams else None

        if lr_key is not None and wd_key is not None and len(hparams) == 2:
            lr_str = _format_hparam_value(hparams[lr_key])
            wd_str = _format_hparam_value(hparams[wd_key])
            return f"LR of {lr_str} and weight decay of {wd_str} wins at {best_score:.2f} {metric_label}"

        parts: list[str] = []
        if lr_key is not None:
            parts.append(f"LR of {_format_hparam_value(hparams.pop(lr_key))}")
        if wd_key is not None:
            parts.append(f"weight decay of {_format_hparam_value(hparams.pop(wd_key))}")
        for k, v in sorted(hparams.items()):
            parts.append(f"{k} of {_format_hparam_value(v)}")

        hparam_str = " and ".join(parts) if parts else "default hyperparameters"
        return f"{hparam_str} wins at {best_score:.2f} {metric_label}"

    def format_winner_summary(self) -> str:
        """Format the winning hyperparameter combination announcement for this saver."""
        if self._best is None:
            raise RuntimeError("No checkpoint has been recorded yet.")
        return self.format_winner(
            self._best.hyperparameters,
            self.metric_name,
            self._best.score,
        )
