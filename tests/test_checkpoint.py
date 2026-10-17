"""Tests for HuggingFace safetensors + tokenizer best-metric checkpoint saving."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from amd_spare_bedroom import (
    BestCheckpointSaver,
    ClassificationMetrics,
    VirtualDDP,
    VirtualFSDP,
    VirtualGrid,
    seed_everything,
)


class _TinyLinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(torch.relu(self.fc1(x)))
        return out


def _build_tiny_hf_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5, "amd": 6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _build_tiny_hf_bert() -> nn.Module:
    config = AutoConfig.for_model(
        "bert",
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    return AutoModelForSequenceClassification.from_config(config)  # type: ignore[no-any-return]


def test_unwrap_model_state_dict_and_save_pretrained_ddp_and_fsdp(tmp_path: Path) -> None:
    """Verify unwrap_model(), state_dict(), load_state_dict(), and save_pretrained() on DDP & FSDP."""
    seed_everything(42)
    grid = VirtualGrid.create(world_size=4, total_memory_gb=8.0, device_type="cpu", seed=42)
    base_model = _TinyLinearNet()
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    ddp: VirtualDDP[_TinyLinearNet] = VirtualDDP(copy.deepcopy(base_model), grid=grid)
    fsdp: VirtualFSDP[_TinyLinearNet] = VirtualFSDP(copy.deepcopy(base_model), grid=grid)

    ddp_opts = ddp.create_optimizers(torch.optim.SGD, lr=0.1)
    fsdp_opts = fsdp.create_optimizers(torch.optim.SGD, lr=0.1)

    for rank in grid.ranks:
        ddp.backward(
            rank,
            nn.functional.cross_entropy(ddp.forward(rank, x[rank : rank + 1]), y[rank : rank + 1]),
        )
        fsdp.backward(
            rank,
            nn.functional.cross_entropy(fsdp.forward(rank, x[rank : rank + 1]), y[rank : rank + 1]),
        )

    ddp.sync_gradients()
    fsdp.sync_gradients()
    ddp.step(ddp_opts)
    fsdp.step(fsdp_opts)

    unwrapped_ddp = ddp.unwrap_model()
    unwrapped_fsdp = fsdp.unwrap_model()

    with torch.no_grad():
        out_ddp = unwrapped_ddp(x)
        out_fsdp = unwrapped_fsdp(x)
        out_fsdp_eval = fsdp.eval_forward(0, x)

    assert torch.allclose(out_ddp, out_fsdp, atol=1e-6)
    assert torch.allclose(out_fsdp, out_fsdp_eval, atol=1e-6)

    # Verify state_dict() and load_state_dict() round-trip on both VirtualDDP and VirtualFSDP
    ddp_state = ddp.state_dict()
    fresh_ddp: VirtualDDP[_TinyLinearNet] = VirtualDDP(_TinyLinearNet(), grid=grid)
    fresh_ddp.load_state_dict(ddp_state)
    with torch.no_grad():
        assert torch.allclose(fresh_ddp.eval_forward(0, x), out_ddp, atol=1e-6)

    fsdp_state = fsdp.state_dict()
    fresh_fsdp: VirtualFSDP[_TinyLinearNet] = VirtualFSDP(_TinyLinearNet(), grid=grid)
    fresh_fsdp.load_state_dict(fsdp_state)
    with torch.no_grad():
        assert torch.allclose(fresh_fsdp.eval_forward(0, x), out_fsdp, atol=1e-6)

    # Verify save_pretrained() directly on VirtualDDP and VirtualFSDP with HuggingFace model
    hf_bert = _build_tiny_hf_bert()
    ddp_hf: VirtualDDP[nn.Module] = VirtualDDP(copy.deepcopy(hf_bert), grid=grid)
    fsdp_hf: VirtualFSDP[nn.Module] = VirtualFSDP(copy.deepcopy(hf_bert), grid=grid)

    ddp_dir = tmp_path / "ddp_pretrained"
    fsdp_dir = tmp_path / "fsdp_pretrained"
    ddp_hf.save_pretrained(ddp_dir)
    fsdp_hf.save_pretrained(fsdp_dir)
    assert (ddp_dir / "model.safetensors").is_file()
    assert (fsdp_dir / "model.safetensors").is_file()


def test_best_checkpoint_saver_hf_safetensors_and_tokenizer_overwrite(tmp_path: Path) -> None:
    """Verify BestCheckpointSaver saves loadable HF safetensors + tokenizer and overwrites on improvement."""
    seed_everything(99)
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=99)
    hf_model = _build_tiny_hf_bert()
    tokenizer = _build_tiny_hf_tokenizer()
    fsdp: VirtualFSDP[nn.Module] = VirtualFSDP(hf_model, grid=grid)
    opts = fsdp.create_optimizers(torch.optim.SGD, lr=0.5)

    input_ids = torch.tensor([[2, 4, 5, 3], [2, 6, 5, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([1, 0], dtype=torch.long)

    ckpt_dir = tmp_path / "best_checkpoint"
    saver = BestCheckpointSaver(ckpt_dir, metric_name="f1", tokenizer=tokenizer)

    # Epoch 1: f1 = 0.75 -> initial save
    m1 = ClassificationMetrics(accuracy=0.75, f1=0.75, precision=0.75, recall=0.75, loss=0.5)
    saved_1 = saver.save_if_best(
        fsdp,
        m1,
        epoch=1,
        hyperparameters={"lr": 1e-4, "weight_decay": 0.01},
    )
    assert saved_1 is True
    assert (ckpt_dir / "model.safetensors").is_file()
    assert (ckpt_dir / "config.json").is_file()
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "checkpoint_metadata.json").is_file()
    assert not list(ckpt_dir.glob("*.bin"))
    assert not list(ckpt_dir.glob("*.pt"))

    # Verify reloadable via HuggingFace AutoModel and AutoTokenizer
    reloaded_1 = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    reloaded_tok = AutoTokenizer.from_pretrained(ckpt_dir)
    assert reloaded_tok.encode("hello world") == tokenizer.encode("hello world")

    reloaded_1.eval()
    with torch.no_grad():
        logits_epoch1 = reloaded_1(input_ids=input_ids, attention_mask=attention_mask).logits
        fsdp_logits_1 = fsdp.eval_forward(
            0, input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert torch.allclose(logits_epoch1, fsdp_logits_1, atol=1e-6)

    # Mutate FSDP weights via an optimization step
    for rank in grid.ranks:
        out = fsdp.forward(
            rank,
            input_ids=input_ids[rank : rank + 1],
            attention_mask=attention_mask[rank : rank + 1],
            labels=labels[rank : rank + 1],
        )
        fsdp.backward(rank, out.loss)
    fsdp.sync_gradients()
    fsdp.step(opts)

    # Epoch 2a: tied f1 = 0.75 -> must NOT overwrite epoch 1 checkpoint
    assert (
        saver.save_if_best(fsdp, m1, epoch=2, hyperparameters={"lr": 1e-4, "weight_decay": 0.01})
        is False
    )

    # Epoch 2b: worse f1 = 0.70 -> must NOT overwrite epoch 1 checkpoint
    m2 = ClassificationMetrics(accuracy=0.70, f1=0.70, precision=0.70, recall=0.70, loss=0.6)
    saved_2 = saver.save_if_best(
        fsdp,
        m2,
        epoch=2,
        hyperparameters={"lr": 1e-4, "weight_decay": 0.01},
    )
    assert saved_2 is False
    assert saver.best_score == pytest.approx(0.75)
    assert saver.best_epoch == 1
    reloaded_after_skip = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    reloaded_after_skip.eval()
    with torch.no_grad():
        logits_after_skip = reloaded_after_skip(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert torch.allclose(logits_after_skip, logits_epoch1, atol=1e-6)

    # Epoch 3: improved f1 = 0.912 -> overwrites single checkpoint directory
    m3 = ClassificationMetrics(accuracy=0.92, f1=0.912, precision=0.91, recall=0.915, loss=0.2)
    saved_3 = saver.save_if_best(
        fsdp,
        m3,
        epoch=3,
        hyperparameters={"lr": 1e-5, "weight_decay": 0.01},
    )
    assert saved_3 is True
    assert list(tmp_path.iterdir()) == [ckpt_dir]
    meta = saver.load_metadata()
    assert meta["best_score"] == pytest.approx(0.912)
    assert meta["epoch"] == 3
    assert meta["hyperparameters"] == {"lr": 1e-5, "weight_decay": 0.01}
    assert saver.best_hyperparameters == {"lr": 1e-5, "weight_decay": 0.01}

    reloaded_3 = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    reloaded_3.eval()
    with torch.no_grad():
        logits_epoch3 = reloaded_3(input_ids=input_ids, attention_mask=attention_mask).logits
        fsdp_logits_3 = fsdp.eval_forward(
            0, input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert torch.allclose(logits_epoch3, fsdp_logits_3, atol=1e-6)
    assert not torch.allclose(logits_epoch3, logits_epoch1, atol=1e-4)

    assert saver.format_winner_summary() == "LR of 1e-5 and weight decay of 0.01 wins at 0.91 F1"
    assert (
        BestCheckpointSaver.format_winner({"lr": 1e-5, "weight_decay": 0.01}, "f1", 0.912)
        == "LR of 1e-5 and weight decay of 0.01 wins at 0.91 F1"
    )


def test_best_checkpoint_saver_min_metric_direction(tmp_path: Path) -> None:
    """Verify mode='auto' selects 'min' for perplexity/eval_loss and rejects NaN or ties."""
    hf_model = _build_tiny_hf_bert()
    ckpt_dir = tmp_path / "sft_ckpt"
    saver = BestCheckpointSaver(ckpt_dir, metric_name="perplexity", mode="auto")
    assert saver.mode == "min"

    assert saver.save_if_best(hf_model, {"perplexity": float("nan")}, epoch=1) is False
    assert saver.save_if_best(hf_model, {"perplexity": 12.5}, epoch=1) is True
    assert saver.best_score == pytest.approx(12.5)
    assert saver.save_if_best(hf_model, {"perplexity": 12.5}, epoch=2) is False
    assert saver.save_if_best(hf_model, {"perplexity": 14.0}, epoch=2) is False
    assert saver.save_if_best(hf_model, {"perplexity": 8.2}, epoch=3) is True
    assert saver.best_score == pytest.approx(8.2)
    assert saver.best_epoch == 3

    # Verify eval_loss extracts loss from ClassificationMetrics
    loss_saver = BestCheckpointSaver(tmp_path / "loss_ckpt", metric_name="eval_loss")
    cm = ClassificationMetrics(accuracy=0.8, f1=0.8, precision=0.8, recall=0.8, loss=0.35)
    assert loss_saver.save_if_best(hf_model, cm, epoch=1) is True
    assert loss_saver.best_score == pytest.approx(0.35)
