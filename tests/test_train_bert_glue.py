"""Tests for example training script argparse plumbing and GLUE dataset loading."""

from __future__ import annotations

from collections.abc import Sized
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from examples.train_bert_glue import (
    GLUE_TASK_NUM_LABELS,
    GLUE_TASK_TO_KEYS,
    SyntheticGLUEDataset,
    get_dataset,
    get_model,
    main,
    parse_args,
)


def test_parse_args_defaults() -> None:
    """Verify default arguments are properly set."""
    args = parse_args([])
    assert args.ranks == 4
    assert args.total_vram_gb == 96.0
    assert args.batch_size == 8
    assert args.epochs == 2
    assert args.lr == 2e-5
    assert args.device == "auto"
    assert args.dataset_size is None
    assert args.max_steps == 5
    assert args.parallelism == "fsdp"
    assert args.use_tiny_bert is False
    assert args.synthetic is False
    assert args.task_name == "mrpc"
    assert args.max_seq_length == 128
    assert args.model_name == "bert-base-uncased"
    assert args.seed == 42
    assert args.amp == "none"
    assert args.max_grad_norm == 1.0
    assert args.weight_decay == 0.01
    assert args.eval_metric == "f1"
    assert args.checkpoint_dir is None
    assert args.grid_search is False
    assert args.lr_grid == "1e-5,2e-5,3e-5,5e-5,1e-4"
    assert args.weight_decay_grid == "0.0,0.001,0.01,0.05,0.1"


def test_parse_args_custom_and_aliases() -> None:
    """Verify CLI flags, custom parameters, and flag aliases are plumbed correctly."""
    cli_args = [
        "--ranks",
        "8",
        "--total-vram-gb",
        "128.0",
        "--batch-size",
        "16",
        "--epochs",
        "4",
        "--lr",
        "1e-4",
        "--weight-decay",
        "0.05",
        "--device",
        "cpu",
        "--dataset-size",
        "200",
        "--max-steps",
        "10",
        "--parallelism",
        "ddp",
        "--use-tiny-bert",
        "--synthetic",
        "--task",
        "sst2",
        "--seq-len",
        "64",
        "--model-name",
        "bert-base-cased",
        "--seed",
        "999",
        "--amp",
        "bf16",
        "--max-grad-norm",
        "0.5",
        "--eval-metric",
        "accuracy",
        "--checkpoint-dir",
        "/tmp/ckpt_test",
        "--grid-search",
        "--lr-grid",
        "1e-5,2e-5",
        "--weight-decay-grid",
        "0.0,0.01",
    ]
    args = parse_args(cli_args)
    assert args.ranks == 8
    assert args.total_vram_gb == 128.0
    assert args.batch_size == 16
    assert args.epochs == 4
    assert args.lr == 1e-4
    assert args.weight_decay == 0.05
    assert args.device == "cpu"
    assert args.dataset_size == 200
    assert args.max_steps == 10
    assert args.parallelism == "ddp"
    assert args.use_tiny_bert is True
    assert args.synthetic is True
    assert args.task_name == "sst2"
    assert args.max_seq_length == 64
    assert args.model_name == "bert-base-cased"
    assert args.seed == 999
    assert args.amp == "bf16"
    assert args.max_grad_norm == 0.5
    assert args.eval_metric == "accuracy"
    assert args.checkpoint_dir == "/tmp/ckpt_test"
    assert args.grid_search is True
    assert args.lr_grid == "1e-5,2e-5"
    assert args.weight_decay_grid == "0.0,0.01"


def test_parse_args_use_synthetic_alias() -> None:
    """Verify --use-synthetic, --precision, and --checkpoint-path aliases are parsed properly."""
    args = parse_args(
        ["--use-synthetic", "--precision", "fp16", "--checkpoint-path", "/tmp/ckpt_alias"]
    )
    assert args.synthetic is True
    assert args.amp == "fp16"
    assert args.checkpoint_dir == "/tmp/ckpt_alias"

    args_none = parse_args(["--amp", "none"])
    assert args_none.amp == "none"


def test_synthetic_glue_dataset_shapes_and_keys() -> None:
    """Verify SyntheticGLUEDataset yields expected tensors."""
    ds = SyntheticGLUEDataset(num_samples=16, seq_len=32, vocab_size=1000, num_labels=2)
    assert len(ds) == 16
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["labels"].dtype == torch.long


def test_synthetic_glue_dataset_regression() -> None:
    """Verify SyntheticGLUEDataset supports regression (num_labels=1)."""
    ds = SyntheticGLUEDataset(num_samples=8, seq_len=16, num_labels=1)
    assert len(ds) == 8
    item = ds[0]
    assert item["labels"].dtype == torch.float32


def test_get_dataset_synthetic_mode() -> None:
    """Verify get_dataset returns synthetic dataset when synthetic=True."""
    ds = get_dataset(
        task_name="mrpc",
        max_seq_length=48,
        dataset_size=20,
        synthetic=True,
    )
    assert isinstance(ds, SyntheticGLUEDataset)
    assert len(ds) == 20
    item = ds[0]
    assert item["input_ids"].shape == (48,)


def test_get_dataset_actual_glue() -> None:
    """Verify get_dataset loads and tokenizes actual GLUE MRPC samples."""
    ds = get_dataset(
        task_name="mrpc",
        model_name="bert-base-uncased",
        max_seq_length=32,
        dataset_size=4,
        synthetic=False,
    )
    assert len(cast(Sized, ds)) == 4
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    assert batch["labels"].shape == (2,)


def test_get_model_tiny_and_custom_labels() -> None:
    """Verify get_model handles tiny BERT and variable number of labels."""
    model_2 = cast(Any, get_model(use_tiny=True, num_labels=2))
    assert model_2.config.num_labels == 2
    assert model_2.config.hidden_size == 128

    model_3 = cast(Any, get_model(use_tiny=True, num_labels=3))
    assert model_3.config.num_labels == 3


def test_glue_task_mappings() -> None:
    """Verify GLUE task dictionaries have standard keys."""
    assert "mrpc" in GLUE_TASK_TO_KEYS
    assert "sst2" in GLUE_TASK_TO_KEYS
    assert GLUE_TASK_NUM_LABELS["mrpc"] == 2
    assert GLUE_TASK_NUM_LABELS["mnli"] == 3
    assert GLUE_TASK_NUM_LABELS["stsb"] == 1


def test_main_end_to_end_smoke_with_amp(capsys: Any, tmp_path: Any) -> None:
    """Verify main() executes end-to-end with AMP, seeding, clip_grad_norm, per-epoch eval, and checkpointing."""
    ckpt_dir = tmp_path / "bert_best_ckpt"
    main(
        [
            "--synthetic",
            "--use-tiny-bert",
            "--ranks",
            "2",
            "--batch-size",
            "4",
            "--dataset-size",
            "8",
            "--seq-len",
            "16",
            "--epochs",
            "2",
            "--max-steps",
            "2",
            "--amp",
            "bf16",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--weight-decay",
            "0.01",
            "--eval-metric",
            "f1",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "AMP Precision Mode:   BF16" in out
    assert "Grad Norm:" in out
    assert "--- Epoch 1 Eval | Loss:" in out
    assert "--- Epoch 2 Eval | Loss:" in out
    assert "| Acc:" in out
    assert "| F1:" in out
    assert "| Precision:" in out
    assert "| Recall:" in out
    assert (ckpt_dir / "model.safetensors").is_file()
    assert (ckpt_dir / "config.json").is_file()
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "checkpoint_metadata.json").is_file()
    assert "Training finished successfully" in out


def test_main_grid_search_end_to_end_and_checkpoint_persistence(capsys: Any, tmp_path: Any) -> None:
    """Seam 2 (#8): End-to-end --grid-search saves single safetensors checkpoint and announces winner last."""
    import re

    import pytest
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from examples.train_bert_glue import parse_float_grid

    assert parse_float_grid("1e-5, 2e-5,3e-5") == [1e-5, 2e-5, 3e-5]
    assert parse_float_grid("0.0 0.01 0.1") == [0.0, 0.01, 0.1]
    with pytest.raises(ValueError, match="must contain at least one numeric value"):
        parse_float_grid(" , ")

    ckpt_dir = tmp_path / "best_ckpt"
    main(
        [
            "--grid-search",
            "--lr-grid",
            "1e-5,1e-3",
            "--weight-decay-grid",
            "0.0,0.01",
            "--ranks",
            "2",
            "--synthetic",
            "--use-tiny-bert",
            "--batch-size",
            "4",
            "--dataset-size",
            "8",
            "--seq-len",
            "16",
            "--epochs",
            "1",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--eval-metric",
            "f1",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    assert len(lines) >= 2
    assert re.match(
        r"^LR of .+ and weight decay of .+ wins at [0-9]+\.[0-9]{2} F1$",
        lines[-1],
    )

    assert (ckpt_dir / "model.safetensors").is_file()
    assert (ckpt_dir / "config.json").is_file()
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "checkpoint_metadata.json").is_file()

    loaded_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    assert loaded_model.config.num_labels == 2
    assert loaded_tokenizer.pad_token == "[PAD]"
