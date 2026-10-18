"""Tests for scikit-learn classification metrics and distributed epoch evaluation."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from amd_spare_bedroom import (
    ClassificationMetrics,
    VirtualDDP,
    VirtualFSDP,
    VirtualGrid,
    compute_classification_metrics,
    evaluate_classification_epoch,
    seed_everything,
)


class _DictClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic 10-sample dict dataset to test non-divisible world_size=4 evaluation."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.features[idx],
            "labels": self.labels[idx],
        }


class _DictLinearClassifier(nn.Module):
    """Simple classifier accepting keyword arguments ('x', 'labels') like HuggingFace models."""

    def __init__(self, in_features: int = 6, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.fc(self.dropout(x))
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits.float(), labels.long())
        return result


def test_compute_classification_metrics_binary_and_logits() -> None:
    """Verify binary accuracy, precision, recall, and F1 against known confusion matrices."""
    # Confusion matrix: TP=3, FP=1, FN=1, TN=3 (total=8)
    # accuracy = 6/8 = 0.75, precision = 3/4 = 0.75, recall = 3/4 = 0.75, f1 = 0.75
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    preds = [1, 1, 1, 0, 1, 0, 0, 0]

    metrics = replace(compute_classification_metrics(preds, labels), loss=0.42)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(0.75)
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.f1 == pytest.approx(0.75)
    assert metrics.loss == pytest.approx(0.42)

    as_dict = metrics.to_dict()
    assert as_dict["accuracy"] == pytest.approx(0.75)
    assert as_dict["f1"] == pytest.approx(0.75)
    assert as_dict["precision"] == pytest.approx(0.75)
    assert as_dict["recall"] == pytest.approx(0.75)
    assert as_dict["loss"] == pytest.approx(0.42)

    # Asymmetric binary matrix via 2D logits tensor:
    # TP=2, FP=2, FN=1, TN=3 (total=8)
    # accuracy = 5/8 = 0.625, precision = 2/4 = 0.5, recall = 2/3, f1 = 2*(0.5*2/3)/(0.5+2/3) = 4/7
    labels_t = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.long)
    # Predicted classes: [1, 1, 0, 1, 1, 0, 0, 0]
    logits_t = torch.tensor(
        [
            [-1.0, 2.0],
            [-0.5, 1.5],
            [3.0, -1.0],
            [-2.0, 1.0],
            [-0.1, 0.2],
            [1.0, -1.0],
            [2.0, 0.0],
            [0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    asym_metrics = compute_classification_metrics(logits_t, labels_t)
    assert asym_metrics.accuracy == pytest.approx(0.625)
    assert asym_metrics.precision == pytest.approx(0.5)
    assert asym_metrics.recall == pytest.approx(2.0 / 3.0)
    assert asym_metrics.f1 == pytest.approx(4.0 / 7.0)
    assert asym_metrics.loss is None


def test_compute_classification_metrics_zero_division_and_multiclass() -> None:
    """Verify zero_division=0.0 when no positives predicted and macro averaging for multiclass."""
    # No positive predictions at all: TP=0, FP=0, FN=2, TN=2
    labels = np.array([1, 1, 0, 0])
    preds = np.array([0, 0, 0, 0])
    zero_metrics = compute_classification_metrics(preds, labels)
    assert zero_metrics.accuracy == pytest.approx(0.5)
    assert zero_metrics.precision == pytest.approx(0.0)
    assert zero_metrics.recall == pytest.approx(0.0)
    assert zero_metrics.f1 == pytest.approx(0.0)

    # 3-class multiclass (e.g., MNLI: 0, 1, 2) with average="auto" -> macro
    # Class 0: TP=2, FP=0, FN=0 -> P=1, R=1, F1=1
    # Class 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    # Class 2: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    # Macro average: P = 2/3, R = 2/3, F1 = 2/3, Accuracy = 4/6 = 2/3
    mc_labels = [0, 0, 1, 1, 2, 2]
    mc_preds = [0, 0, 1, 2, 2, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mc_metrics = compute_classification_metrics(
            mc_preds, mc_labels, average="auto", pos_label=2
        )
    assert mc_metrics.accuracy == pytest.approx(2.0 / 3.0)
    assert mc_metrics.precision == pytest.approx(2.0 / 3.0)
    assert mc_metrics.recall == pytest.approx(2.0 / 3.0)
    assert mc_metrics.f1 == pytest.approx(2.0 / 3.0)

    # 2D 3-class logits (N, 3) where labels/preds in slice only hit {0, 1} still uses macro
    logits_3c = np.array(
        [
            [2.0, -1.0, -2.0],  # pred 0, true 0
            [-1.0, 2.0, -2.0],  # pred 1, true 0
        ]
    )
    labels_subset = [0, 0]
    # Under binary (pos_label=1): TP=0, FP=1, FN=0 -> F1=0.0
    # Under macro across {0, 1}: Class 0 (TP=1,FN=1,FP=0 -> F1=2/3), Class 1 (F1=0) -> macro F1=1/3
    subset_3c_metrics = compute_classification_metrics(logits_3c, labels_subset, average="auto")
    assert subset_3c_metrics.f1 == pytest.approx(1.0 / 3.0)


def test_evaluate_classification_epoch_ddp_fsdp_parity_and_unpadding() -> None:
    """Verify 4-rank DDP and FSDP evaluation match exact 10-sample single-device evaluation."""
    seed_everything(123)
    features = torch.randn(10, 6)
    labels = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.long)
    dataset = _DictClassificationDataset(features, labels)

    base_model = _DictLinearClassifier(in_features=6, num_classes=2)
    base_model.eval()
    with torch.no_grad():
        ref_out = base_model(x=features, labels=labels)
        ref_logits = ref_out["logits"]
        ref_loss = float(ref_out["loss"].item())
    expected = replace(compute_classification_metrics(ref_logits, labels), loss=ref_loss)

    grid = VirtualGrid.create(world_size=4, total_memory_gb=8.0, device_type="cpu", seed=123)
    ddp: VirtualDDP[_DictLinearClassifier] = VirtualDDP(base_model, grid=grid)
    fsdp: VirtualFSDP[_DictLinearClassifier] = VirtualFSDP(base_model, grid=grid)

    ddp.train()
    fsdp.train()
    assert ddp.is_training is True
    assert fsdp.is_training is True

    ddp_metrics = evaluate_classification_epoch(ddp, dataset, batch_size=2)
    fsdp_metrics = evaluate_classification_epoch(fsdp, dataset, batch_size=2)

    # Restores training mode after evaluation
    assert ddp.is_training is True
    assert fsdp.is_training is True

    # Exact parity with single-device unpadded 10-sample evaluation
    assert fsdp_metrics.accuracy == pytest.approx(expected.accuracy)
    assert fsdp_metrics.f1 == pytest.approx(expected.f1)
    assert fsdp_metrics.precision == pytest.approx(expected.precision)
    assert fsdp_metrics.recall == pytest.approx(expected.recall)
    assert fsdp_metrics.loss == pytest.approx(expected.loss, rel=1e-5)

    assert ddp_metrics.accuracy == pytest.approx(expected.accuracy)
    assert ddp_metrics.f1 == pytest.approx(expected.f1)
    assert ddp_metrics.precision == pytest.approx(expected.precision)
    assert ddp_metrics.recall == pytest.approx(expected.recall)
    assert ddp_metrics.loss == pytest.approx(expected.loss, rel=1e-5)

    # Also verify bfloat16 AMP evaluation converts bfloat16 logits cleanly
    ddp_bf16: VirtualDDP[_DictLinearClassifier] = VirtualDDP(
        base_model, grid=grid, amp_dtype="bfloat16"
    )
    bf16_metrics = evaluate_classification_epoch(ddp_bf16, dataset, batch_size=2)
    assert bf16_metrics.loss is not None
    assert bf16_metrics.loss == pytest.approx(expected.loss, rel=1e-2)


def test_compute_classification_metrics_exceptions() -> None:
    """Verify exceptions for mismatched lengths, empty inputs, float labels, and non-finite values."""
    with pytest.raises(ValueError, match="Prediction count .* does not match label count"):
        compute_classification_metrics([1, 0, 1], [1, 0])

    with pytest.raises(ValueError, match="Cannot compute classification metrics on empty"):
        compute_classification_metrics([], [])

    with pytest.raises(ValueError, match="1D ground-truth labels must contain discrete integer"):
        compute_classification_metrics(np.array([1, 0]), np.array([0.5, 1.5]))

    with pytest.raises(ValueError, match="1D ground-truth labels must contain discrete integer"):
        compute_classification_metrics(torch.tensor([1, 0]), torch.tensor([0.2, 0.8]))

    with pytest.raises(ValueError, match="Predictions/labels contain non-finite"):
        compute_classification_metrics(np.array([float("nan"), 1.0]), np.array([0, 1]))

    with pytest.raises(ValueError, match="Predictions/labels contain non-finite"):
        compute_classification_metrics(torch.tensor([float("inf"), 1.0]), torch.tensor([0, 1]))


def test_compute_classification_metrics_1d_float_and_single_logit() -> None:
    """Verify 1D float logits and (N, 1) single-output binary logits threshold at 0.0."""
    # 1D float logits: positive -> class 1, non-positive -> class 0
    float_logits = torch.tensor([2.5, -1.0, 0.5, -3.0], dtype=torch.float32)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    metrics_1d = compute_classification_metrics(float_logits, labels)
    assert metrics_1d.accuracy == pytest.approx(1.0)
    assert metrics_1d.f1 == pytest.approx(1.0)

    # 2D single-output binary logits (N, 1)
    single_col_logits = torch.tensor([[2.5], [-1.0], [0.5], [-3.0]], dtype=torch.float32)
    metrics_2d_1col = compute_classification_metrics(single_col_logits, labels)
    assert metrics_2d_1col.accuracy == pytest.approx(1.0)
    assert metrics_2d_1col.f1 == pytest.approx(1.0)

    # Binary with pos_label != 1: classes {0, 2} with pos_label=2
    custom_labels = [0, 2, 2, 0]
    custom_preds = [0, 2, 0, 0]  # TP=1, FP=0, FN=1, TN=2
    custom_metrics = compute_classification_metrics(
        custom_preds, custom_labels, average="auto", pos_label=2
    )
    assert custom_metrics.accuracy == pytest.approx(0.75)
    assert custom_metrics.precision == pytest.approx(1.0)
    assert custom_metrics.recall == pytest.approx(0.5)
    assert custom_metrics.f1 == pytest.approx(2.0 / 3.0)


class _RawTensorModule(nn.Module):
    """Standard PyTorch module returning raw torch.Tensor logits without accepting 'labels'."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
        nn.init.eye_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


class _SingleLogitModule(nn.Module):
    """Binary classifier returning single logit (N, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def test_evaluate_classification_epoch_raw_tensor_single_rank_and_masking() -> None:
    """Verify evaluation works with raw Tensor outputs, single-rank grid, and -100 label masking."""
    grid = VirtualGrid.create(world_size=1, total_memory_gb=2.0, device_type="cpu", seed=42)
    model = _RawTensorModule()
    ddp = VirtualDDP(model, grid=grid)

    # 4 samples with 4 features: first 2 are unmasked, 3rd has label -100 (masked)
    features = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],  # argmax -> 0
            [0.0, 3.0, 0.0, 0.0],  # argmax -> 1
            [0.0, 3.0, 0.0, 0.0],  # argmax -> 1, but masked with -100!
        ]
    )
    labels = torch.tensor([0, 1, -100], dtype=torch.long)
    dataset = _DictClassificationDataset(features, labels)

    metrics = evaluate_classification_epoch(ddp, dataset, batch_size=2)
    # The masked sample is ignored; remaining 2 samples both match exactly
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.loss is not None
    assert metrics.loss >= 0.0

    # Binary single-column module using binary_cross_entropy_with_logits
    bin_model = _SingleLogitModule()
    bin_ddp = VirtualDDP(bin_model, grid=grid)
    bin_features = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    bin_labels = torch.tensor([1, 0], dtype=torch.long)
    bin_dataset = _DictClassificationDataset(bin_features, bin_labels)

    bin_metrics = evaluate_classification_epoch(bin_ddp, bin_dataset, batch_size=2)
    assert bin_metrics.accuracy == pytest.approx(1.0)
    assert bin_metrics.loss is not None


def test_evaluate_classification_epoch_exceptions() -> None:
    """Verify input validation and error raises in evaluate_classification_epoch."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=2.0, device_type="cpu", seed=42)
    model = _RawTensorModule()
    ddp = VirtualDDP(model, grid=grid)

    # 1. Non-sized dataset
    class _UnsizedDataset:
        pass

    with pytest.raises(TypeError, match="eval_dataset must implement __len__"):
        evaluate_classification_epoch(ddp, _UnsizedDataset())  # type: ignore[arg-type]

    # 2. Empty dataset
    empty_dataset = _DictClassificationDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="eval_dataset must be non-empty"):
        evaluate_classification_epoch(ddp, empty_dataset)

    # 3. Batch not a Mapping
    class _TupleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (torch.randn(4), torch.tensor(0))

    with pytest.raises(TypeError, match="Evaluation batch must be a Mapping"):
        evaluate_classification_epoch(ddp, _TupleDataset())

    # 4. Batch missing 'labels' or 'label'
    class _MissingLabelsDataset(Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"x": torch.randn(4)}

    with pytest.raises(ValueError, match="Evaluation batch must contain a 'labels' or 'label'"):
        evaluate_classification_epoch(ddp, _MissingLabelsDataset())

    # 5. All labels masked with -100
    all_masked = _DictClassificationDataset(torch.randn(2, 4), torch.tensor([-100, -100]))
    with pytest.raises(ValueError, match="All evaluation labels are masked"):
        evaluate_classification_epoch(ddp, all_masked)
