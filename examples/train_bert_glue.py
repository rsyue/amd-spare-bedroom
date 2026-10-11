"""Example: Train BERT on GLUE across emulated AMD GPU ranks using VirtualDDP or VirtualFSDP.

Demonstrates emulating multi-rank DDP and FSDP inside the 96GB unified memory pool
on AMD ROCm (targeting gfx_1151 / Radeon 8060S / Strix Halo).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sized
from typing import Any, cast

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import VirtualGrid
from amd_spare_bedroom.sampler import VirtualRankSampler

GLUE_TASK_TO_KEYS: dict[str, tuple[str, ...]] = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASK_NUM_LABELS: dict[str, int] = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
    "wnli": 2,
}


class SyntheticGLUEDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory synthetic tokenized dataset matching BERT GLUE format with masking."""

    def __init__(
        self,
        num_samples: int = 128,
        seq_len: int = 64,
        vocab_size: int = 30522,
        num_labels: int = 2,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
        self.attention_mask = torch.zeros((num_samples, seq_len), dtype=torch.long)
        self.token_type_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)

        for i in range(num_samples):
            # Variable sequence length with padding and masking up to seq_len
            actual_len = int(torch.randint(max(2, seq_len // 4), seq_len + 1, (1,)).item())
            if actual_len > 2:
                self.input_ids[i, 1 : actual_len - 1] = torch.randint(
                    104, vocab_size, (actual_len - 2,)
                )
            self.input_ids[i, 0] = 101  # [CLS] token
            self.input_ids[i, actual_len - 1] = 102  # [SEP] token
            # Active tokens receive 1; masked padding tokens beyond actual_len remain 0
            self.attention_mask[i, :actual_len] = 1

        if num_labels == 1:
            self.labels = torch.rand((num_samples,), dtype=torch.float32) * 5.0
        else:
            self.labels = torch.randint(0, num_labels, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }


def get_dataset(
    task_name: str = "mrpc",
    model_name: str = "bert-base-uncased",
    max_seq_length: int = 128,
    dataset_size: int | None = None,
    synthetic: bool = False,
) -> Dataset[dict[str, torch.Tensor]]:
    """Load actual GLUE dataset (tokenized) or create synthetic dataset for smoke tests."""
    num_labels = GLUE_TASK_NUM_LABELS.get(task_name, 2)
    if synthetic:
        n_samples = dataset_size if dataset_size is not None else 64
        return SyntheticGLUEDataset(
            num_samples=n_samples,
            seq_len=max_seq_length,
            vocab_size=30522,
            num_labels=num_labels,
        )

    print(f"[*] Loading actual GLUE dataset for task '{task_name}'...")
    try:
        try:
            raw_dataset = load_dataset("nyu-mll/glue", task_name, split="train")
        except Exception:
            raw_dataset = load_dataset("glue", task_name, split="train")
    except Exception as exc:
        print(f"[!] Error loading GLUE dataset '{task_name}': {exc}")
        print("[!] Tip: For offline runs or fast verification, use --synthetic")
        raise

    if dataset_size is not None:
        actual_count = min(dataset_size, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(actual_count))

    print(f"[*] Tokenizing {len(raw_dataset)} samples with '{model_name}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence_keys = GLUE_TASK_TO_KEYS.get(task_name, ("sentence1", "sentence2"))

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        args = [examples[key] for key in sentence_keys]
        return tokenizer(  # type: ignore[no-any-return]
            *args,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_token_type_ids=True,
        )

    tokenized = raw_dataset.map(tokenize_fn, batched=True)
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")

    columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")

    tokenized.set_format(type="torch", columns=columns)
    return tokenized  # type: ignore[no-any-return]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT on GLUE using emulated VirtualDDP or VirtualFSDP ranks."
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=4,
        help="Number of virtual ranks to emulate in the grid (default: 4)",
    )
    parser.add_argument(
        "--total-vram-gb",
        type=float,
        default=96.0,
        help="Total physical VRAM budget in GB (default: 96.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-rank batch size (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0; <=0 to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device backend ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Number of samples to use (default: None for full dataset, or 64 if --synthetic)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum training steps per epoch for demonstration (default: 5; <= 0 for all steps in epoch)",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="fsdp",
        choices=["fsdp", "ddp"],
        help="Distributed parallelism mode ('fsdp' for sharded weights, 'ddp' for full replication)",
    )
    parser.add_argument(
        "--use-tiny-bert",
        action="store_true",
        help="Use a compact BERT architecture for fast verification",
    )
    parser.add_argument(
        "--synthetic",
        "--use-synthetic",
        action="store_true",
        dest="synthetic",
        help="Use synthetic dataset instead of downloading actual GLUE dataset (for quick run/smoke test)",
    )
    parser.add_argument(
        "--task-name",
        "--task",
        type=str,
        default="mrpc",
        dest="task_name",
        choices=list(GLUE_TASK_TO_KEYS.keys()),
        help="GLUE task name (default: 'mrpc')",
    )
    parser.add_argument(
        "--max-seq-length",
        "--seq-len",
        type=int,
        default=128,
        dest="max_seq_length",
        help="Maximum sequence length after tokenization (default: 128)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT model name or path (default: 'bert-base-uncased')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and sampling (default: 42)",
    )
    return parser.parse_args(args)


def get_model(
    model_name: str = "bert-base-uncased",
    use_tiny: bool = True,
    num_labels: int = 2,
) -> nn.Module:
    """Instantiate a BERT classification model."""
    if use_tiny:
        # Compact 2-layer BERT for fast local runs & tests
        config = AutoConfig.for_model(
            "bert",
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            num_labels=num_labels,
        )
        return cast(
            nn.Module,
            AutoModelForSequenceClassification.from_config(config),  # type: ignore[no-untyped-call]
        )

    # Standard pretrained model weights for downstream fine-tuning
    return cast(
        nn.Module,
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ),
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"AMD SPARE BEDROOM: Virtual {args.parallelism.upper()} Training on ROCm (gfx_1151)")
    print("=" * 70)

    # 1. Initialize Virtual Grid
    grid = VirtualGrid.create(
        world_size=args.ranks,
        total_memory_gb=args.total_vram_gb,
        device_type=args.device,
    )

    per_rank_gb = grid.per_rank_memory_bytes / (1024**3)
    print(f"[*] Physical VRAM Budget: {args.total_vram_gb:.1f} GB")
    print(f"[*] Virtual World Size:   {grid.world_size} ranks")
    print(f"[*] Per-Rank Budget:      {per_rank_gb:.2f} GB per virtual device")
    print(f"[*] Active Device Type:   {grid.device_type.upper()}")

    if grid.device_type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_detected = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] ROCm Physical Device: {gpu_name} ({vram_detected:.1f} GB detected)")
    print("-" * 70)

    # 2. Prepare Model and wrap with VirtualFSDP or VirtualDDP
    print("[*] Instantiating BERT sequence classification model...")
    num_labels = GLUE_TASK_NUM_LABELS.get(args.task_name, 2)
    base_model = get_model(
        model_name=args.model_name,
        use_tiny=args.use_tiny_bert,
        num_labels=num_labels,
    )
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Model Parameter Count: {param_count:,}")

    if args.parallelism == "fsdp":
        print(
            f"[*] Wrapping model across {grid.world_size} virtual ranks via VirtualFSDP (Sharded)..."
        )
        dist_model: Any = VirtualFSDP(base_model, grid=grid)
        per_rank_params = dist_model.get_rank_parameter_numel(0)
        pct = (per_rank_params / param_count) * 100
        print(
            f"[*] Memory Optimization: {per_rank_params:,} parameters stored per rank ({pct:.1f}% of full model)"
        )
    else:
        print(
            f"[*] Wrapping model across {grid.world_size} virtual ranks via VirtualDDP (Replicated)..."
        )
        dist_model = VirtualDDP(base_model, grid=grid)

    optimizers = dist_model.create_optimizers(torch.optim.AdamW, lr=args.lr)

    # 3. Create Dataset and Rank-Sharded Samplers
    dataset = get_dataset(
        task_name=args.task_name,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dataset_size=args.dataset_size,
        synthetic=args.synthetic,
    )

    rank_loaders: dict[int, DataLoader[dict[str, torch.Tensor]]] = {}
    samplers: dict[int, VirtualRankSampler] = {}

    dataset_sized = cast(Sized, dataset)
    for rank in grid.ranks:
        sampler = VirtualRankSampler(
            dataset_size=len(dataset_sized),
            num_replicas=grid.world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        samplers[rank] = sampler
        rank_loaders[rank] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
        )

    target_device = torch.device(grid.device_type)
    dataset_type = "Synthetic" if args.synthetic else f"GLUE ({args.task_name.upper()})"
    print(f"[*] Dataset:              {dataset_type} ({len(dataset_sized)} samples)")
    has_max_steps = args.max_steps is not None and args.max_steps > 0
    max_steps_str = f"max {args.max_steps} steps/epoch" if has_max_steps else "full epoch"
    print(f"[*] Training configured for {args.epochs} epoch(s), {max_steps_str}")
    print("=" * 70)

    # 4. Training Loop with Lockstep All-Reduce Synchronization
    for epoch in range(args.epochs):
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs}")
        for rank in grid.ranks:
            samplers[rank].set_epoch(epoch)

        rank_iters = {rank: iter(rank_loaders[rank]) for rank in grid.ranks}
        step = 0
        epoch_start = time.perf_counter()

        while not has_max_steps or step < args.max_steps:
            # Check if any rank ran out of batches
            try:
                batches = {rank: next(rank_iters[rank]) for rank in grid.ranks}
            except StopIteration:
                break

            step += 1
            step_losses: dict[int, float] = {}

            # A. Independent Forward & Backward pass on each virtual rank
            for rank in grid.ranks:
                batch = batches[rank]
                input_ids = batch["input_ids"].to(target_device)
                attention_mask = batch["attention_mask"].to(target_device)
                labels = batch["labels"].to(target_device)
                token_type_ids = (
                    batch["token_type_ids"].to(target_device) if "token_type_ids" in batch else None
                )

                forward_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if token_type_ids is not None:
                    forward_kwargs["token_type_ids"] = token_type_ids

                outputs: Any = dist_model.forward(rank, **forward_kwargs)
                loss = outputs.loss
                step_losses[rank] = float(loss.item())

                # Backward pass computes local gradients
                loss.backward()

            # B. Collective Gradient Synchronization (All-Reduce for DDP, Reduce-Scatter for FSDP)
            dist_model.sync_gradients()

            # Clip gradients to ensure numerical stability across virtual ranks
            if args.max_grad_norm > 0:
                for rank in grid.ranks:
                    if args.parallelism == "fsdp":
                        sharded_params = dist_model.get_sharded_parameters(rank)
                        for p in sharded_params:
                            if p.grad is not None:
                                p.grad.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                        torch.nn.utils.clip_grad_norm_(
                            sharded_params,
                            args.max_grad_norm,
                        )
                    else:
                        replica_params = list(dist_model.replicas[rank].parameters())
                        for p in replica_params:
                            if p.grad is not None:
                                p.grad.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                        torch.nn.utils.clip_grad_norm_(
                            replica_params,
                            args.max_grad_norm,
                        )

            # C. Lockstep Optimizer Step
            dist_model.step(optimizers)

            # D. Verify Multi-Rank Weight Parity
            if hasattr(dist_model, "assert_parameters_synced"):
                synced = dist_model.assert_parameters_synced()
                if not synced:
                    print(f"[!] Warning: Parameter desynchronization detected at step {step}!")
                    ref_p = list(dist_model.replicas[grid.ranks[0]].named_parameters())
                    for r in grid.ranks[1:]:
                        rep_p = dict(dist_model.replicas[r].named_parameters())
                        for n, p0 in ref_p:
                            pr = rep_p[n]
                            if not torch.allclose(p0, pr, atol=1e-6):
                                print(
                                    f"    Desync on Rank {r} for {n}: max diff = {(p0 - pr).abs().max().item()}, p0 nan={torch.isnan(p0).any().item()}, pr nan={torch.isnan(pr).any().item()}"
                                )
                    sys.exit(1)

            avg_loss = sum(step_losses.values()) / len(step_losses)
            loss_breakdown = ", ".join(
                f"R{rank_id}: {loss_val:.4f}" for rank_id, loss_val in step_losses.items()
            )
            print(
                f"  Step {step:02d} | Avg Loss: {avg_loss:.4f} | Ranks: [{loss_breakdown}] | Mode: {args.parallelism.upper()}"
            )

        epoch_time = time.perf_counter() - epoch_start
        print(f"--- Epoch {epoch + 1} completed in {epoch_time:.2f}s ---")

    print("\n" + "=" * 70)
    print(
        f"Training finished successfully across all emulated ranks using {args.parallelism.upper()}!"
    )
    if hasattr(dist_model, "assert_parameters_synced"):
        print(f"Final parameter lockstep check: {dist_model.assert_parameters_synced()}")
    else:
        full_p = dist_model.get_full_parameters()
        print(f"Final sharded parameter reconstruction check: {len(full_p)} parameters active")
    print("=" * 70)


if __name__ == "__main__":
    main()
