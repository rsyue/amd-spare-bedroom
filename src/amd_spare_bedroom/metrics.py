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
    """Convert a sequence, numpy array, or torch Tensor into a 1D int64 numpy array and optional num_classes."""
    if isinstance(values, torch.Tensor):
        tensor = values.detach()
        num_classes = int(tensor.shape[-1]) if (allow_logits and tensor.ndim >= 2) else None
        if allow_logits and tensor.ndim >= 2:
            tensor = torch.argmax(tensor, dim=-1)
        flat = np.asarray(
            tensor.to(dtype=torch.int64, device="cpu").numpy(), dtype=np.int64
        ).reshape(-1)
        return flat, num_classes

    arr = np.asarray(values)
    num_classes = int(arr.shape[-1]) if (allow_logits and arr.ndim >= 2) else None
    if allow_logits and arr.ndim >= 2:
        arr = np.argmax(arr, axis=-1)

    return np.asarray(arr, dtype=np.int64).reshape(-1), num_classes


def compute_classification_metrics(
    preds: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    labels: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    average: AverageMode = "auto",
    pos_label: int = 1,
) -> ClassificationMetrics:
    """Compute accuracy, F1, precision, and recall using scikit-learn.

    Args:
        preds: 1D predicted class indices or 2D+ logits (reduced via argmax(-1)).
        labels: 1D ground-truth class indices.
        average: Averaging mode for F1/precision/recall. When 'auto', selects 'binary'
            for binary class sets ({0, pos_label}) and 'macro' for multiclass sets.
        pos_label: Positive class label when using binary averaging.

    Returns:
        ClassificationMetrics containing accuracy, f1, precision, and recall.
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
    device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels_tensor = device_batch.get("labels", device_batch.get("label"))
    if not isinstance(labels_tensor, torch.Tensor):
        raise ValueError("Evaluation batch must contain a 'labels' or 'label' Tensor.")
    outputs = dist_model.eval_forward(rank, **device_batch)

    if hasattr(outputs, "logits"):
        logits_tensor = cast(torch.Tensor, outputs.logits)
    elif isinstance(outputs, Mapping) and "logits" in outputs:
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
    """
    dataset_size = len(cast(Sized, eval_dataset))
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

        eval_loss = float(F.cross_entropy(valid_logits.float(), valid_labels.long()).item())
        base_metrics = compute_classification_metrics(
            preds=valid_logits,
            labels=valid_labels.to(torch.int64),
            average=average,
            pos_label=pos_label,
        )
        return replace(base_metrics, loss=eval_loss)
    finally:
        dist_model.train(was_training)
