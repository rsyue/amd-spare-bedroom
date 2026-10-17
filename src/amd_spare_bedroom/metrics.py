"""Classification and Causal LM SFT evaluation metrics and distributed epoch evaluation utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from amd_spare_bedroom.amp import AmpGradScalerState, resolve_amp_dtype
from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import VirtualGrid
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


@dataclass(frozen=True)
class SFTMetrics:
    """Structured Causal LM Supervised Fine-Tuning (SFT) evaluation metrics."""

    eval_loss: float
    perplexity: float
    token_accuracy: float

    def to_dict(self) -> dict[str, float]:
        """Convert SFT metrics into a plain dictionary of float scores."""
        return {
            "eval_loss": self.eval_loss,
            "perplexity": self.perplexity,
            "token_accuracy": self.token_accuracy,
        }


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


def compute_sft_metrics(
    eval_loss: float,
    shift_preds: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    shift_labels: Sequence[int] | np.ndarray[Any, Any] | torch.Tensor,
    ignore_index: int = -100,
) -> SFTMetrics:
    """Compute validation perplexity and masked next-token accuracy via scikit-learn.

    Args:
        eval_loss: Mean token-level cross-entropy loss across unmasked target positions.
        shift_preds: Shifted predicted token IDs or logits (`logits[..., :-1, :]`).
        shift_labels: Shifted ground-truth token IDs (`labels[..., 1:]`) with masked positions
            set to `ignore_index`.
        ignore_index: Label value to exclude from accuracy calculation (default: `-100`).

    Returns:
        `SFTMetrics` containing `eval_loss`, `perplexity`, and `token_accuracy`.
    """
    y_true, _ = _to_1d_class_array(shift_labels, allow_logits=False)
    preds_ndim = (
        shift_preds.ndim
        if isinstance(shift_preds, (torch.Tensor, np.ndarray))
        else np.asarray(shift_preds).ndim
    )
    preds_numel = (
        int(shift_preds.numel())
        if isinstance(shift_preds, torch.Tensor)
        else int(np.asarray(shift_preds).size)
    )
    allow_logits = preds_ndim >= 2 and preds_numel != int(y_true.size)
    y_pred, _ = _to_1d_class_array(shift_preds, allow_logits=allow_logits)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Shifted prediction count ({y_pred.shape[0]}) does not match "
            f"shifted label count ({y_true.shape[0]})."
        )

    valid_mask = y_true != ignore_index
    if bool(np.any(valid_mask)):
        token_accuracy = float(accuracy_score(y_true[valid_mask], y_pred[valid_mask]))
    else:
        token_accuracy = 0.0

    loss_val = float(eval_loss)
    perplexity = math.exp(min(loss_val, 20.0)) if not math.isnan(loss_val) else float("nan")
    return SFTMetrics(
        eval_loss=loss_val,
        perplexity=perplexity,
        token_accuracy=token_accuracy,
    )


def _create_eval_dataloaders(
    dist_model: VirtualDDP[Any] | VirtualFSDP[Any],
    eval_dataset: Dataset[Any],
    batch_size: int,
) -> tuple[int, VirtualGrid, torch.device, dict[int, DataLoader[Any]]]:
    """Create non-shuffled per-rank evaluation dataloaders for a distributed model."""
    dataset_size = len(cast(Sized, eval_dataset))
    grid = dist_model.grid
    device = torch.device(grid.device_type)
    dataloaders = {
        r: DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=VirtualRankSampler(
                dataset_size=dataset_size,
                num_replicas=grid.world_size,
                rank=r,
                shuffle=False,
                drop_last=False,
            ),
        )
        for r in grid.ranks
    }
    return dataset_size, grid, device, dataloaders


def _extract_logits_and_labels_from_outputs(
    outputs: Any,
    labels_tensor: torch.Tensor | Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract `(logits_cpu, labels_cpu)` from model outputs and batch labels."""
    if not isinstance(labels_tensor, torch.Tensor):
        raise ValueError("Evaluation batch must contain a 'labels' or 'label' Tensor.")
    if hasattr(outputs, "logits"):
        logits_tensor = cast(torch.Tensor, outputs.logits)
    elif isinstance(outputs, Mapping) and "logits" in outputs:
        logits_tensor = cast(torch.Tensor, outputs["logits"])
    else:
        raise TypeError(f"Could not extract logits Tensor from model output: {type(outputs)!r}")
    return logits_tensor.detach().cpu(), labels_tensor.detach().cpu()


def _extract_batch_logits_and_labels(
    dist_model: VirtualDDP[Any] | VirtualFSDP[Any],
    rank: int,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eval_forward on a single rank's mapping batch and return (logits, labels) on CPU."""
    device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels_tensor = device_batch.get("labels", device_batch.get("label"))
    outputs = dist_model.eval_forward(rank, **device_batch)
    return _extract_logits_and_labels_from_outputs(outputs, labels_tensor)


def _score_causal_lm_batch(
    logits_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute shifted predictions, shifted labels, per-sample loss sums, and valid token counts."""
    shift_logits = logits_cpu[..., :-1, :].contiguous().float()
    shift_labels = labels_cpu[..., 1:].contiguous().long()
    shift_preds = torch.argmax(shift_logits, dim=-1).to(dtype=torch.int64)

    bsz = shift_logits.shape[0]
    vocab_size = shift_logits.shape[-1]
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(bsz, -1)
    sample_loss_sums = token_losses.sum(dim=-1)
    sample_valid_counts = (shift_labels != ignore_index).sum(dim=-1).to(torch.int64)
    return shift_preds, shift_labels, sample_loss_sums, sample_valid_counts


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
    dataset_size, grid, device, dataloaders = _create_eval_dataloaders(
        dist_model, eval_dataset, batch_size
    )

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


def evaluate_causal_lm_model(
    model: nn.Module,
    eval_dataset: Dataset[Any],
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    amp_dtype: torch.dtype | str | None = None,
    ignore_index: int = -100,
) -> SFTMetrics:
    """Evaluate a single Causal LM `nn.Module` on `eval_dataset` and return `SFTMetrics`."""
    resolved_device = torch.device(device) if isinstance(device, str) else device
    resolved_amp = resolve_amp_dtype(amp_dtype)
    scaler = AmpGradScalerState(
        device_type=resolved_device.type,
        ranks=[0],
        amp_dtype=resolved_amp,
    )
    dataloader: DataLoader[Any] = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    was_training = model.training
    model.eval()
    try:
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss_sum = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                device_batch = {
                    k: v.to(resolved_device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                labels_tensor = device_batch.get("labels", device_batch.get("label"))
                with scaler.autocast_context():
                    outputs = model(**device_batch)
                logits_cpu, labels_cpu = _extract_logits_and_labels_from_outputs(
                    outputs, labels_tensor
                )
                shift_preds, shift_labels, sample_loss_sums, sample_valid_counts = (
                    _score_causal_lm_batch(logits_cpu, labels_cpu, ignore_index=ignore_index)
                )
                all_preds.append(shift_preds)
                all_labels.append(shift_labels)
                total_loss_sum += float(sample_loss_sums.sum().item())
                total_valid_tokens += int(sample_valid_counts.sum().item())

        eval_loss = total_loss_sum / total_valid_tokens if total_valid_tokens > 0 else 0.0
        return compute_sft_metrics(
            eval_loss=eval_loss,
            shift_preds=torch.cat(all_preds, dim=0),
            shift_labels=torch.cat(all_labels, dim=0),
            ignore_index=ignore_index,
        )
    finally:
        model.train(was_training)


def evaluate_causal_lm_epoch(
    dist_model: VirtualDDP[Any] | VirtualFSDP[Any],
    eval_dataset: Dataset[Any],
    batch_size: int = 8,
    ignore_index: int = -100,
) -> SFTMetrics:
    """Evaluate a distributed Causal LM across all virtual ranks.

    Switches `dist_model` to `.eval()` mode, runs lockstep `eval_forward` across all
    ranks, shifts `logits[..., :-1, :]` and `labels[..., 1:]` for next-token prediction,
    strips sampler wrap-around padding via `VirtualRankSampler.unstride_and_unpad` to
    exact `len(eval_dataset)`, computes exact token-level cross-entropy loss, perplexity,
    and masked next-token accuracy (`ignore_index` excluded), and restores the prior
    training state.
    """
    dataset_size, grid, device, dataloaders = _create_eval_dataloaders(
        dist_model, eval_dataset, batch_size
    )

    was_training = dist_model.is_training
    dist_model.eval()
    try:
        rank_shift_preds: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}
        rank_shift_labels: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}
        rank_loss_sums: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}
        rank_token_counts: dict[int, list[torch.Tensor]] = {r: [] for r in grid.ranks}

        for step_batches in zip(*(dataloaders[r] for r in grid.ranks), strict=True):
            for rank, batch in zip(grid.ranks, step_batches, strict=True):
                logits_cpu, labels_cpu = _extract_batch_logits_and_labels(
                    dist_model, rank, batch, device
                )
                shift_preds, shift_labels, sample_loss_sums, sample_valid_counts = (
                    _score_causal_lm_batch(logits_cpu, labels_cpu, ignore_index=ignore_index)
                )
                rank_shift_preds[rank].append(shift_preds)
                rank_shift_labels[rank].append(shift_labels)
                rank_loss_sums[rank].append(sample_loss_sums)
                rank_token_counts[rank].append(sample_valid_counts)

        cat_preds = [torch.cat(rank_shift_preds[r], dim=0) for r in grid.ranks]
        cat_labels = [torch.cat(rank_shift_labels[r], dim=0) for r in grid.ranks]
        cat_losses = [torch.cat(rank_loss_sums[r], dim=0) for r in grid.ranks]
        cat_counts = [torch.cat(rank_token_counts[r], dim=0) for r in grid.ranks]

        valid_preds = VirtualRankSampler.unstride_and_unpad(cat_preds, dataset_size)
        valid_labels = VirtualRankSampler.unstride_and_unpad(cat_labels, dataset_size)
        valid_loss_sums = VirtualRankSampler.unstride_and_unpad(cat_losses, dataset_size)
        valid_counts = VirtualRankSampler.unstride_and_unpad(cat_counts, dataset_size)

        total_valid_tokens = int(valid_counts.sum().item())
        eval_loss = (
            float(valid_loss_sums.sum().item() / total_valid_tokens)
            if total_valid_tokens > 0
            else 0.0
        )
        return compute_sft_metrics(
            eval_loss=eval_loss,
            shift_preds=valid_preds,
            shift_labels=valid_labels,
            ignore_index=ignore_index,
        )
    finally:
        dist_model.train(was_training)
