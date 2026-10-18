"""Classification evaluation metrics and distributed epoch evaluation utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.sampler import VirtualRankSampler

AverageMode = Literal["binary", "macro", "micro", "weighted", "auto"]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Structured classification evaluation metrics computed via scikit-learn."""

    accuracy: float
    f1: float
    precision: float
    recall: float
    loss: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert metrics into a plain dictionary of float scores."""
        result: dict[str, float] = {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }
        if self.loss is not None:
            result["loss"] = self.loss
        return result


def _to_1d_class_array(
    values: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    *,
    allow_logits: bool,
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], int | None]:
    """Convert a sequence, numpy array, or torch Tensor into a 1D int64 numpy array and optional num_classes.

    Args:
        values: Integer class indices or floating-point logits.
        allow_logits: Whether to allow multi-dimensional logits and reduce them via argmax / threshold.

    Returns:
        Tuple of (1D int64 numpy array, optional num_classes).

    Raises:
        ValueError: If values contain non-finite values (NaN/Inf) or if floating-point labels are passed.
    """
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        if not torch.isfinite(tensor).all():
            raise ValueError("Predictions/labels contain non-finite (NaN or Inf) values.")

        num_classes: int | None = None
        if allow_logits:
            if torch.is_floating_point(tensor):
                if tensor.ndim >= 2 and tensor.shape[-1] > 1:
                    num_classes = int(tensor.shape[-1])
                    tensor = torch.argmax(tensor, dim=-1)
                elif tensor.ndim == 1 or (tensor.ndim >= 2 and tensor.shape[-1] == 1):
                    # Single-output binary logits (threshold at 0.0)
                    num_classes = 2
                    tensor = (tensor.squeeze(-1) > 0.0).to(dtype=torch.int64)
                else:
                    tensor = torch.argmax(tensor, dim=-1)
            else:
                if tensor.ndim >= 2:
                    num_classes = int(tensor.shape[-1])
                    tensor = torch.argmax(tensor, dim=-1)
        else:
            if tensor.numel() > 0 and torch.is_floating_point(tensor):
                raise ValueError(
                    "1D ground-truth labels must contain discrete integer class labels, not floats."
                )

        flat = np.asarray(
            tensor.to(dtype=torch.int64, device="cpu").numpy(), dtype=np.int64
        ).reshape(-1)
        return flat, num_classes

    arr = np.asarray(values)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predictions/labels contain non-finite (NaN or Inf) values.")

    num_classes = None
    if allow_logits:
        if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
            if arr.ndim >= 2 and arr.shape[-1] > 1:
                num_classes = int(arr.shape[-1])
                arr = np.argmax(arr, axis=-1)
            elif arr.ndim == 1 or (arr.ndim >= 2 and arr.shape[-1] == 1):
                num_classes = 2
                arr = (np.squeeze(arr, axis=-1 if arr.ndim >= 2 else None) > 0.0).astype(np.int64)
            else:
                arr = np.argmax(arr, axis=-1)
        else:
            if arr.ndim >= 2:
                num_classes = int(arr.shape[-1])
                arr = np.argmax(arr, axis=-1)
    else:
        if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
            raise ValueError(
                "1D ground-truth labels must contain discrete integer class labels, not floats."
            )

    return np.asarray(arr, dtype=np.int64).reshape(-1), num_classes


def compute_classification_metrics(
    preds: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    labels: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    average: AverageMode = "auto",
    pos_label: int = 1,
) -> ClassificationMetrics:
    """Compute accuracy, F1, precision, and recall using scikit-learn.

    Args:
        preds: 1D predicted class indices or 2D+ logits (reduced via argmax(-1) or threshold).
        labels: 1D ground-truth class indices.
        average: Averaging mode for F1/precision/recall. When 'auto', selects 'binary'
            for binary class sets ({0, pos_label}) and 'macro' for multiclass sets.
        pos_label: Positive class label when using binary averaging.

    Returns:
        ClassificationMetrics containing accuracy, f1, precision, and recall.

    Raises:
        ValueError: If prediction count does not match label count or if inputs are empty.
    """
    y_pred, logits_num_classes = _to_1d_class_array(preds, allow_logits=True)
    y_true, _ = _to_1d_class_array(labels, allow_logits=False)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Prediction count ({y_pred.shape[0]}) does not match label count ({y_true.shape[0]})."
        )

    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute classification metrics on empty predictions/labels.")

    resolved_average: str
    if average == "auto":
        unique_classes = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
        is_multiclass_logits = logits_num_classes is not None and logits_num_classes > 2
        if (
            not is_multiclass_logits
            and len(unique_classes) <= 2
            and unique_classes.issubset({0, pos_label})
        ):
            resolved_average = "binary"
        else:
            resolved_average = "macro"
    else:
        resolved_average = average

    effective_pos_label = pos_label if resolved_average == "binary" else 1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            average=resolved_average,
            pos_label=effective_pos_label,
            zero_division=0.0,
        )
    )
    prec = float(
        precision_score(
            y_true,
            y_pred,
            average=resolved_average,
            pos_label=effective_pos_label,
            zero_division=0.0,
        )
    )
    rec = float(
        recall_score(
            y_true,
            y_pred,
            average=resolved_average,
            pos_label=effective_pos_label,
            zero_division=0.0,
        )
    )

    return ClassificationMetrics(
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
    )


def _extract_batch_logits_and_labels(
    dist_model: VirtualDDP[Any] | VirtualFSDP[Any],
    rank: int,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eval_forward on a single rank's mapping batch and return (logits, labels) on CPU."""
    if not isinstance(batch, Mapping):
        raise TypeError(f"Evaluation batch must be a Mapping (e.g. dict), got {type(batch)!r}")
    device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels_tensor = device_batch.get("labels", device_batch.get("label"))
    if not isinstance(labels_tensor, torch.Tensor):
        raise ValueError("Evaluation batch must contain a 'labels' or 'label' Tensor.")

    forward_inputs = {k: v for k, v in device_batch.items() if k not in ("labels", "label")}
    outputs = dist_model.eval_forward(rank, **forward_inputs)

    if isinstance(outputs, torch.Tensor):
        logits_tensor = outputs
    elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        logits_tensor = outputs.logits
    elif isinstance(outputs, Mapping) and isinstance(outputs.get("logits"), torch.Tensor):
        logits_tensor = cast(torch.Tensor, outputs["logits"])
    else:
        raise TypeError(f"Could not extract logits Tensor from model output: {type(outputs)!r}")

    return logits_tensor.detach().cpu(), labels_tensor.detach().cpu()


def evaluate_classification_epoch(
    dist_model: VirtualDDP[Any] | VirtualFSDP[Any],
    eval_dataset: Dataset[Any],
    batch_size: int = 8,
    average: AverageMode = "auto",
    pos_label: int = 1,
) -> ClassificationMetrics:
    """Evaluate a distributed classification model across all virtual ranks.

    Switches `dist_model` to `.eval()` mode, runs lockstep `eval_forward` across all
    ranks using `VirtualRankSampler(shuffle=False, drop_last=False)`, strips sampler
    wrap-around padding via `VirtualRankSampler.unstride_and_unpad` so every validation
    sample is counted once, computes average cross-entropy loss and scikit-learn
    classification metrics, and restores the prior training state.

    Args:
        dist_model: VirtualDDP or VirtualFSDP wrapped model instance.
        eval_dataset: Validation PyTorch Dataset implementing __len__ and yielding mapping batches.
        batch_size: Per-rank mini-batch evaluation size.
        average: Scikit-learn averaging mode ('auto', 'binary', 'macro', etc.).
        pos_label: Positive class label for binary classification.

    Returns:
        ClassificationMetrics containing accuracy, f1, precision, recall, and loss.

    Raises:
        TypeError: If eval_dataset does not implement __len__.
        ValueError: If eval_dataset is empty or all labels are masked.
    """
    if not isinstance(eval_dataset, Sized):
        raise TypeError(f"eval_dataset must implement __len__, got {type(eval_dataset).__name__}")
    dataset_size = len(eval_dataset)
    if dataset_size <= 0:
        raise ValueError(f"eval_dataset must be non-empty, got length {dataset_size}.")

    grid = dist_model.grid
    device = torch.device(grid.device_type)

    samplers = {
        r: VirtualRankSampler(
            dataset_size=dataset_size,
            num_replicas=grid.world_size,
            rank=r,
            shuffle=False,
            drop_last=False,
        )
        for r in grid.ranks
    }
    dataloaders = {
        r: DataLoader(eval_dataset, batch_size=batch_size, sampler=samplers[r]) for r in grid.ranks
    }

    was_training = dist_model.is_training
    dist_model.eval()
    try:
        with torch.no_grad():
            rank_logits: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}
            rank_labels: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}

            for step_batches in zip(*(dataloaders[r] for r in grid.ranks), strict=True):
                for rank, batch in zip(grid.ranks, step_batches, strict=True):
                    logits_cpu, labels_cpu = _extract_batch_logits_and_labels(
                        dist_model, rank, batch, device
                    )
                    rank_logits[rank].append(logits_cpu)
                    rank_labels[rank].append(labels_cpu)

            cat_logits_by_rank = [torch.cat(rank_logits[r], dim=0) for r in grid.ranks]
            cat_labels_by_rank = [torch.cat(rank_labels[r], dim=0) for r in grid.ranks]

            valid_logits = VirtualRankSampler.unstride_and_unpad(cat_logits_by_rank, dataset_size)
            valid_labels = VirtualRankSampler.unstride_and_unpad(
                cat_labels_by_rank, dataset_size
            ).reshape(-1)

            # Filter out ignored/masked labels (< 0, such as -100)
            valid_mask = valid_labels >= 0
            if not valid_mask.any():
                raise ValueError("All evaluation labels are masked (< 0).")

            eval_logits = valid_logits[valid_mask]
            eval_labels = valid_labels[valid_mask]

            if eval_logits.ndim == 1 or (eval_logits.ndim == 2 and eval_logits.shape[-1] == 1):
                eval_loss = float(
                    F.binary_cross_entropy_with_logits(
                        eval_logits.squeeze(-1).float(), eval_labels.float()
                    ).item()
                )
            else:
                eval_loss = float(F.cross_entropy(eval_logits.float(), eval_labels.long()).item())

            base_metrics = compute_classification_metrics(
                preds=eval_logits,
                labels=eval_labels.to(torch.int64),
                average=average,
                pos_label=pos_label,
            )
            return replace(base_metrics, loss=eval_loss)
    finally:
        dist_model.train(was_training)
