"""Example: Train BERT on GLUE across emulated AMD GPU ranks using VirtualDDP.

Demonstrates emulating multi-rank DDP inside the 96GB unified memory pool
on AMD ROCm (targeting gfx_1151 / Radeon 8060S / Strix Halo).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import VirtualGrid
from amd_spare_bedroom.sampler import VirtualRankSampler


class SyntheticGLUEDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory synthetic tokenized dataset matching BERT GLUE format."""

    def __init__(
        self,
        num_samples: int = 128,
        seq_len: int = 64,
        vocab_size: int = 30522,
        num_labels: int = 2,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, num_labels, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT on GLUE using emulated VirtualDDP ranks."
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device backend ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=64,
        help="Number of synthetic samples (default: 64)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum training steps per epoch for demonstration (default: 5)",
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
    return parser.parse_args()


def get_model(use_tiny: bool = True) -> nn.Module:
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
            num_labels=2,
        )
        return AutoModelForSequenceClassification.from_config(config)

    # Standard bert-base-uncased configuration
    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
    return AutoModelForSequenceClassification.from_config(config)


def main() -> None:
    args = parse_args()

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
    base_model = get_model(use_tiny=args.use_tiny_bert)
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
    dataset = SyntheticGLUEDataset(num_samples=args.dataset_size, seq_len=64)

    rank_loaders: dict[int, DataLoader[dict[str, torch.Tensor]]] = {}
    samplers: dict[int, VirtualRankSampler] = {}

    for rank in grid.ranks:
        sampler = VirtualRankSampler(
            dataset_size=len(dataset),
            num_replicas=grid.world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        samplers[rank] = sampler
        rank_loaders[rank] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
        )

    target_device = torch.device(grid.device_type)
    print(f"[*] Training configured for {args.epochs} epoch(s), max {args.max_steps} steps/epoch")
    print("=" * 70)

    # 4. Training Loop with Lockstep All-Reduce Synchronization
    for epoch in range(args.epochs):
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs}")
        for rank in grid.ranks:
            samplers[rank].set_epoch(epoch)

        rank_iters = {rank: iter(rank_loaders[rank]) for rank in grid.ranks}
        step = 0
        epoch_start = time.perf_counter()

        while step < args.max_steps:
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

                outputs: Any = dist_model.forward(
                    rank,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                step_losses[rank] = float(loss.item())

                # Backward pass computes local gradients
                loss.backward()

            # B. Collective Gradient Synchronization (All-Reduce for DDP, Reduce-Scatter for FSDP)
            dist_model.sync_gradients()

            # C. Lockstep Optimizer Step
            dist_model.step(optimizers)

            # D. Verify Multi-Rank Weight Parity
            if hasattr(dist_model, "assert_parameters_synced"):
                synced = dist_model.assert_parameters_synced()
                if not synced:
                    print(f"[!] Warning: Parameter desynchronization detected at step {step}!")
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
