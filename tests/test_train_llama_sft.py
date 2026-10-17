"""Tests for Llama 3.2 1B Supervised Fine-Tuning (SFT) workflow and evaluation metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Sized
from typing import Any, cast

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amd_spare_bedroom import (
    SFTMetrics,
    VirtualDDP,
    VirtualFSDP,
    VirtualGrid,
    compute_sft_metrics,
    evaluate_causal_lm_epoch,
    seed_everything,
)
from amd_spare_bedroom.metrics import evaluate_causal_lm_model
from examples.train_llama_sft import (
    SyntheticSFTDataset,
    get_llama_model,
    get_sft_dataset,
    get_tokenizer,
    main,
    parse_args,
)


def test_compute_sft_metrics_known_values_and_masking() -> None:
    """Seam 1: Verify compute_sft_metrics with known loss and masked label handling."""
    # Loss = ln(4) -> perplexity == 4.0
    known_loss = math.log(4.0)
    # 6 positions: 4 unmasked, 2 masked with -100
    shift_labels = torch.tensor([10, 20, 30, -100, 40, -100], dtype=torch.long)
    # 3 correct out of 4 valid unmasked tokens: accuracy = 3/4 = 0.75
    shift_preds = torch.tensor([10, 20, 99, 50, 40, 88], dtype=torch.long)

    metrics = compute_sft_metrics(
        eval_loss=known_loss,
        shift_preds=shift_preds,
        shift_labels=shift_labels,
        ignore_index=-100,
    )
    assert isinstance(metrics, SFTMetrics)
    assert metrics.eval_loss == pytest.approx(known_loss)
    assert metrics.perplexity == pytest.approx(4.0)
    assert metrics.token_accuracy == pytest.approx(0.75)

    as_dict = metrics.to_dict()
    assert as_dict["eval_loss"] == pytest.approx(known_loss)
    assert as_dict["perplexity"] == pytest.approx(4.0)
    assert as_dict["token_accuracy"] == pytest.approx(0.75)

    # 3D logits (1, 6, vocab_size=100) are reduced via argmax(-1)
    logits_3d = torch.zeros((1, 6, 100), dtype=torch.float32)
    for i, pred_tok in enumerate(shift_preds.tolist()):
        logits_3d[0, i, pred_tok] = 10.0
    logits_metrics = compute_sft_metrics(
        eval_loss=known_loss,
        shift_preds=logits_3d,
        shift_labels=shift_labels.unsqueeze(0),
        ignore_index=-100,
    )
    assert logits_metrics.token_accuracy == pytest.approx(0.75)

    # 2D token IDs (2, 3) vs 1D labels (6,) when total elements match (not logits)
    preds_2d_ids = torch.tensor([[10, 20, 99], [50, 40, 88]], dtype=torch.long)
    metrics_2d_ids = compute_sft_metrics(
        eval_loss=known_loss,
        shift_preds=preds_2d_ids,
        shift_labels=shift_labels,
        ignore_index=-100,
    )
    assert metrics_2d_ids.token_accuracy == pytest.approx(0.75)

    # All tokens masked returns 0.0 accuracy without crashing
    all_masked_labels = torch.full((6,), -100, dtype=torch.long)
    masked_metrics = compute_sft_metrics(
        eval_loss=1.0,
        shift_preds=shift_preds,
        shift_labels=all_masked_labels,
        ignore_index=-100,
    )
    assert masked_metrics.token_accuracy == pytest.approx(0.0)

    # High loss is clamped at math.exp(20.0) to prevent overflow
    overflow_metrics = compute_sft_metrics(
        eval_loss=100.0,
        shift_preds=shift_preds,
        shift_labels=shift_labels,
    )
    assert overflow_metrics.perplexity == pytest.approx(math.exp(20.0))


def test_evaluate_causal_lm_epoch_ddp_fsdp_parity_and_unpadding() -> None:
    """Seam 2: Verify 2-rank DDP and FSDP evaluation match single-device evaluation on 7 samples."""
    seed_everything(42)
    dataset = SyntheticSFTDataset(num_samples=7, seq_len=16, vocab_size=100)

    # Compact 2-layer Llama model with GQA (4 query heads, 2 kv heads)
    base_model = get_llama_model(model_name="tiny-llama", use_tiny=True)

    expected = evaluate_causal_lm_model(
        model=base_model,
        eval_dataset=dataset,
        batch_size=2,
        device="cpu",
    )
    assert expected.token_accuracy >= 0.0
    assert math.isfinite(expected.perplexity)

    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    ddp = VirtualDDP(base_model, grid=grid)
    fsdp = VirtualFSDP(base_model, grid=grid)

    ddp.train()
    fsdp.train()

    ddp_metrics = evaluate_causal_lm_epoch(ddp, dataset, batch_size=2)
    fsdp_metrics = evaluate_causal_lm_epoch(fsdp, dataset, batch_size=2)

    # Model training mode is restored
    assert ddp.is_training is True
    assert fsdp.is_training is True

    # Exact parity across single-device, DDP, and FSDP with 7 samples (non-divisible by 2)
    assert ddp_metrics.eval_loss == pytest.approx(expected.eval_loss, rel=1e-5)
    assert ddp_metrics.perplexity == pytest.approx(expected.perplexity, rel=1e-5)
    assert ddp_metrics.token_accuracy == pytest.approx(expected.token_accuracy, rel=1e-5)

    assert fsdp_metrics.eval_loss == pytest.approx(expected.eval_loss, rel=1e-5)
    assert fsdp_metrics.perplexity == pytest.approx(expected.perplexity, rel=1e-5)
    assert fsdp_metrics.token_accuracy == pytest.approx(expected.token_accuracy, rel=1e-5)


def test_parse_args_defaults_and_custom() -> None:
    """Verify parse_args sets sensible SFT defaults and handles custom flags and aliases."""
    args = parse_args([])
    assert args.ranks == 4
    assert args.total_vram_gb == 96.0
    assert args.batch_size == 4
    assert args.epochs == 2
    assert args.lr == 2e-5
    assert args.weight_decay == 0.01
    assert args.max_grad_norm == 1.0
    assert args.parallelism == "fsdp"
    assert args.model_name == "meta-llama/Llama-3.2-1B"
    assert args.use_tiny_llama is False
    assert args.dataset_name == "yahma/alpaca-cleaned"
    assert args.max_seq_length == 256
    assert args.max_steps == 5
    assert args.synthetic is False
    assert args.seed == 42
    assert args.eval_metric == "perplexity"
    assert args.checkpoint_dir is None
    assert args.grid_search is False

    custom_args = parse_args(
        [
            "--ranks",
            "2",
            "--batch-size",
            "8",
            "--epochs",
            "3",
            "--lr",
            "1e-4",
            "--weight-decay",
            "0.05",
            "--max-grad-norm",
            "0.5",
            "--parallelism",
            "ddp",
            "--use-tiny-llama",
            "--use-synthetic",
            "--seq-len",
            "128",
            "--max-steps",
            "10",
            "--precision",
            "bf16",
            "--eval-metric",
            "token_accuracy",
            "--checkpoint-path",
            "/tmp/llama_ckpt",
            "--grid-search",
            "--lr-grid",
            "1e-5,2e-5",
            "--weight-decay-grid",
            "0.0,0.01",
        ]
    )
    assert custom_args.ranks == 2
    assert custom_args.batch_size == 8
    assert custom_args.epochs == 3
    assert custom_args.lr == 1e-4
    assert custom_args.weight_decay == 0.05
    assert custom_args.max_grad_norm == 0.5
    assert custom_args.parallelism == "ddp"
    assert custom_args.use_tiny_llama is True
    assert custom_args.synthetic is True
    assert custom_args.max_seq_length == 128
    assert custom_args.max_steps == 10
    assert custom_args.amp == "bf16"
    assert custom_args.eval_metric == "token_accuracy"
    assert custom_args.checkpoint_dir == "/tmp/llama_ckpt"
    assert custom_args.grid_search is True
    assert custom_args.lr_grid == "1e-5,2e-5"
    assert custom_args.weight_decay_grid == "0.0,0.01"


def test_synthetic_sft_dataset_shapes_and_masking() -> None:
    """Verify SyntheticSFTDataset produces valid shapes and -100 label masks for prompt/padding."""
    ds = SyntheticSFTDataset(num_samples=16, seq_len=32, vocab_size=1000)
    assert len(ds) == 16
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["labels"].shape == (32,)

    # Labels must contain -100 for masked prompt/padding tokens and non -100 for active response tokens
    labels = item["labels"]
    assert -100 in labels.tolist()
    unmasked = labels[labels != -100]
    assert len(unmasked) > 0
    assert (unmasked >= 0).all().item()


def test_get_sft_dataset_synthetic_mode() -> None:
    """Verify get_sft_dataset returns SyntheticSFTDataset when synthetic=True."""
    ds = get_sft_dataset(synthetic=True, max_seq_length=48, dataset_size=10)
    assert isinstance(ds, SyntheticSFTDataset)
    assert len(ds) == 10
    assert ds[0]["input_ids"].shape == (48,)


def test_get_llama_model_tiny() -> None:
    """Verify get_llama_model instantiates compact Llama with GQA."""
    model = cast(Any, get_llama_model(use_tiny=True))
    assert model.config.model_type == "llama"
    assert model.config.hidden_size == 128
    assert model.config.num_hidden_layers == 2
    assert model.config.num_attention_heads == 4
    assert model.config.num_key_value_heads == 2


def test_get_tokenizer_synthetic() -> None:
    """Verify get_tokenizer produces a tokenizer with pad/bos/eos tokens."""
    tokenizer = get_tokenizer(synthetic=True)
    assert tokenizer.pad_token is not None
    assert tokenizer.bos_token is not None
    assert tokenizer.eos_token is not None


def test_main_end_to_end_fsdp_smoke_and_checkpoint(capsys: Any, tmp_path: Any) -> None:
    """Seam 3: Verify train_llama_sft.py main() executes in FSDP mode with AMP and saves checkpoint."""
    ckpt_dir = tmp_path / "llama_sft_ckpt"
    main(
        [
            "--synthetic",
            "--use-tiny-llama",
            "--ranks",
            "2",
            "--batch-size",
            "2",
            "--dataset-size",
            "6",
            "--seq-len",
            "24",
            "--epochs",
            "2",
            "--max-steps",
            "2",
            "--parallelism",
            "fsdp",
            "--amp",
            "bf16",
            "--device",
            "cpu",
            "--seed",
            "42",
            "--eval-metric",
            "perplexity",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "AMP Precision Mode:   BF16" in out
    assert "Virtual FSDP Training" in out
    assert "--- Epoch 1 Eval |" in out
    assert "--- Epoch 2 Eval |" in out
    assert "Eval Loss:" in out
    assert "Perplexity:" in out
    assert "Token Accuracy:" in out
    assert "Training finished successfully" in out

    # Verify single HuggingFace safetensors checkpoint files exist and can be loaded
    assert (ckpt_dir / "model.safetensors").is_file()
    assert (ckpt_dir / "config.json").is_file()
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "checkpoint_metadata.json").is_file()

    loaded_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    assert loaded_model.config.model_type == "llama"
    assert loaded_tokenizer.pad_token is not None


def test_main_end_to_end_ddp_smoke(capsys: Any) -> None:
    """Seam 3: Verify train_llama_sft.py main() executes in DDP mode."""
    main(
        [
            "--synthetic",
            "--use-tiny-llama",
            "--ranks",
            "2",
            "--batch-size",
            "2",
            "--dataset-size",
            "4",
            "--seq-len",
            "16",
            "--epochs",
            "1",
            "--max-steps",
            "1",
            "--parallelism",
            "ddp",
            "--device",
            "cpu",
            "--seed",
            "123",
        ]
    )
    out = capsys.readouterr().out
    assert "Virtual DDP Training" in out
    assert "--- Epoch 1 Eval |" in out
    assert "Eval Loss:" in out
    assert "Perplexity:" in out
    assert "Token Accuracy:" in out
    assert "Training finished successfully" in out


def test_main_grid_search_causal_lm(capsys: Any, tmp_path: Any) -> None:
    """Seam 3: Verify train_llama_sft.py --grid-search executes and announces winner on final line."""
    ckpt_dir = tmp_path / "llama_grid_ckpt"
    main(
        [
            "--grid-search",
            "--lr-grid",
            "1e-4,2e-4",
            "--weight-decay-grid",
            "0.0,0.01",
            "--ranks",
            "2",
            "--synthetic",
            "--use-tiny-llama",
            "--batch-size",
            "2",
            "--dataset-size",
            "4",
            "--seq-len",
            "16",
            "--epochs",
            "1",
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--seed",
            "42",
            "--eval-metric",
            "perplexity",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    assert len(lines) >= 2
    assert re.match(
        r"^LR of .+ and weight decay of .+ wins at [0-9]+\.[0-9]{2} Perplexity$",
        lines[-1],
    )
    assert (ckpt_dir / "model.safetensors").is_file()


def test_get_sft_dataset_fallback_splits(monkeypatch: Any) -> None:
    """Verify get_sft_dataset partitions fallback train split into non-overlapping train/val splits."""
    from datasets import Dataset as HFDataset

    samples = [{"instruction": f"Inst {i}", "input": "", "output": f"Resp {i}"} for i in range(20)]
    mock_hf = HFDataset.from_list(samples)

    def mock_load_dataset(name: str, split: str = "train") -> Any:
        if split == "validation":
            raise ValueError("No validation split")
        return mock_hf

    monkeypatch.setattr("examples.train_llama_sft.load_dataset", mock_load_dataset)
    monkeypatch.setattr(
        "examples.train_llama_sft.get_tokenizer",
        lambda **kw: get_tokenizer(synthetic=True),
    )

    train_ds = get_sft_dataset(
        dataset_name="mock_alpaca", max_seq_length=16, synthetic=False, split="train"
    )
    val_ds = get_sft_dataset(
        dataset_name="mock_alpaca", max_seq_length=16, synthetic=False, split="validation"
    )

    # 20 samples -> val_size = max(int(20 * 0.1), 1) = 2, train_size = 18
    assert len(cast(Sized, train_ds)) == 18
    assert len(cast(Sized, val_ds)) == 2
