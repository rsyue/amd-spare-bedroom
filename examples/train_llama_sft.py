"""Example: Supervised Fine-Tuning (SFT) on Llama 3.2 1B across emulated AMD GPU ranks.

Demonstrates emulating multi-rank VirtualFSDP and VirtualDDP on AMD ROCm
(targeting gfx_1151 / Radeon 8060S / Strix Halo unified memory APUs).
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from collections.abc import Sized
from typing import Any, cast

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from amd_spare_bedroom.checkpoint import BestCheckpointSaver
from amd_spare_bedroom.config import parse_args_with_config
from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import VirtualGrid, seed_everything
from amd_spare_bedroom.grid_search import (
    DEFAULT_LR_GRID_STR,
    DEFAULT_WEIGHT_DECAY_GRID_STR,
    VirtualGridSearch,
    parse_float_grid,
)
from amd_spare_bedroom.metrics import evaluate_causal_lm_epoch, evaluate_causal_lm_model
from amd_spare_bedroom.sampler import VirtualRankSampler


class SyntheticSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory synthetic tokenized instruction-following dataset with label masking."""

    def __init__(
        self,
        num_samples: int = 64,
        seq_len: int = 64,
        vocab_size: int = 32000,
    ) -> None:
        if seq_len < 4:
            raise ValueError(f"seq_len must be >= 4, got {seq_len}")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
        self.attention_mask = torch.zeros((num_samples, seq_len), dtype=torch.long)
        self.labels = torch.full((num_samples, seq_len), -100, dtype=torch.long)

        for i in range(num_samples):
            actual_len = (
                int(torch.randint(max(4, seq_len // 2), seq_len + 1, (1,)).item())
                if seq_len >= 4
                else seq_len
            )
            prompt_len = max(1, actual_len // 3)

            self.input_ids[i, 0] = 1  # BOS token
            if actual_len > 2:
                self.input_ids[i, 1 : actual_len - 1] = torch.randint(
                    3, vocab_size, (actual_len - 2,)
                )
            self.input_ids[i, actual_len - 1] = 2  # EOS token
            self.attention_mask[i, :actual_len] = 1

            # Active target response tokens receive token IDs; prompt and padding are masked with -100
            self.labels[i, prompt_len:actual_len] = self.input_ids[i, prompt_len:actual_len]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def get_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-1B",
    synthetic: bool = False,
) -> Any:
    """Instantiate a Llama tokenizer (or a self-contained offline tokenizer when synthetic=True)."""
    if synthetic:
        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "amd": 4,
            "rocm": 5,
            "fsdp": 6,
            "llama": 7,
        }
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
            tokenizer_object=tok,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_llama_model(
    model_name: str = "meta-llama/Llama-3.2-1B",
    use_tiny: bool = False,
) -> nn.Module:
    """Instantiate a Llama causal language model."""
    if use_tiny:
        config = AutoConfig.for_model(
            "llama",
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        return cast(
            nn.Module,
            AutoModelForCausalLM.from_config(config),  # type: ignore[no-untyped-call]
        )
    return cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(model_name),
    )


def get_sft_dataset(
    dataset_name: str = "yahma/alpaca-cleaned",
    model_name: str = "meta-llama/Llama-3.2-1B",
    max_seq_length: int = 256,
    dataset_size: int | None = None,
    synthetic: bool = False,
    split: str = "train",
) -> Dataset[dict[str, torch.Tensor]]:
    """Load instruction dataset or create synthetic SFT dataset with causal label masking."""
    if synthetic:
        n_samples = dataset_size if dataset_size is not None else 64
        return SyntheticSFTDataset(
            num_samples=n_samples,
            seq_len=max_seq_length,
            vocab_size=32000,
        )

    print(f"[*] Loading instruction dataset '{dataset_name}' (split='{split}')...")
    try:
        if split == "train":
            try:
                load_dataset(dataset_name, split="validation")
                raw_dataset = load_dataset(dataset_name, split="train")
            except Exception:
                full_train = load_dataset(dataset_name, split="train")
                val_size = min(max(int(len(full_train) * 0.1), 1), 500)
                raw_dataset = full_train.select(range(len(full_train) - val_size))
        else:
            raw_dataset = load_dataset(dataset_name, split=split)
    except Exception:
        full_train = load_dataset(dataset_name, split="train")
        val_size = min(max(int(len(full_train) * 0.1), 1), 500)
        if split == "validation":
            raw_dataset = full_train.select(range(len(full_train) - val_size, len(full_train)))
        else:
            raw_dataset = full_train.select(range(len(full_train) - val_size))

    if dataset_size is not None:
        actual_count = min(dataset_size, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(actual_count))

    print(f"[*] Tokenizing {len(raw_dataset)} SFT samples with '{model_name}' tokenizer...")
    tokenizer = get_tokenizer(model_name=model_name, synthetic=False)

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        instructions = examples.get("instruction", [""] * len(examples[list(examples.keys())[0]]))
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples.get("output", [""] * len(instructions))

        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_labels: list[list[int]] = []

        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

        for inst, inp, out in zip(instructions, inputs, outputs, strict=False):
            if inp.strip():
                prompt_text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt_text = f"### Instruction:\n{inst}\n\n### Response:\n"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            response_ids = tokenizer.encode(f"{out}{tokenizer.eos_token}", add_special_tokens=False)
            if len(prompt_ids) >= max_seq_length:
                prompt_ids = prompt_ids[: max_seq_length - 1]
            full_ids = (prompt_ids + response_ids)[:max_seq_length]

            prompt_len = min(len(prompt_ids), len(full_ids))
            seq_len = len(full_ids)
            pad_len = max_seq_length - seq_len

            input_ids = full_ids + [pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
            labels = [-100] * prompt_len + full_ids[prompt_len:] + [-100] * pad_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    tokenized = raw_dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized  # type: ignore[no-any-return]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    default_amp = "bf16" if torch.cuda.is_available() else "none"

    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning (SFT) on Llama 3.2 1B using emulated VirtualFSDP or VirtualDDP."
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
        default=4,
        help="Per-rank batch size (default: 4)",
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
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
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
        help="Maximum training steps per epoch (default: 5; <= 0 for full epoch)",
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="fsdp",
        choices=["fsdp", "ddp"],
        help="Distributed parallelism mode ('fsdp' for sharded weights, 'ddp' for full replication)",
    )
    parser.add_argument(
        "--use-tiny-llama",
        action="store_true",
        help="Use a compact 2-layer Llama architecture for fast verification",
    )
    parser.add_argument(
        "--synthetic",
        "--use-synthetic",
        action="store_true",
        dest="synthetic",
        help="Use synthetic dataset for quick offline verification",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yahma/alpaca-cleaned",
        help="HuggingFace instruction dataset name (default: 'yahma/alpaca-cleaned')",
    )
    parser.add_argument(
        "--max-seq-length",
        "--seq-len",
        type=int,
        default=256,
        dest="max_seq_length",
        help="Maximum sequence length after tokenization (default: 256)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Pretrained Llama model name or path (default: 'meta-llama/Llama-3.2-1B')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and sampling (default: 42)",
    )
    parser.add_argument(
        "--amp",
        "--precision",
        type=str,
        default=default_amp,
        dest="amp",
        choices=["none", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
        help="Automatic mixed precision mode ('none'/'fp32', 'bf16'/'bfloat16', 'fp16'/'float16')",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="perplexity",
        choices=["perplexity", "token_accuracy", "eval_loss", "loss"],
        help="Evaluation metric to monitor for best checkpoint saving (default: 'perplexity')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint-path",
        type=str,
        default=None,
        dest="checkpoint_dir",
        help="Directory path to save the single best HuggingFace safetensors + tokenizer checkpoint",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run multi-rank hyperparameter grid search across sequential waves",
    )
    parser.add_argument(
        "--lr-grid",
        type=str,
        default=DEFAULT_LR_GRID_STR,
        help=f"Comma- or space-separated learning rates for grid search (default: '{DEFAULT_LR_GRID_STR}')",
    )
    parser.add_argument(
        "--weight-decay-grid",
        type=str,
        default=DEFAULT_WEIGHT_DECAY_GRID_STR,
        help=f"Comma- or space-separated weight decay values for grid search (default: '{DEFAULT_WEIGHT_DECAY_GRID_STR}')",
    )
    parsed = parse_args_with_config(parser, args)
    if parsed.ranks < 1:
        raise ValueError(f"ranks must be >= 1, got {parsed.ranks}")
    if not math.isfinite(parsed.total_vram_gb) or parsed.total_vram_gb <= 0:
        raise ValueError(
            f"total-vram-gb must be a positive finite float, got {parsed.total_vram_gb}"
        )
    if parsed.batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {parsed.batch_size}")
    if parsed.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {parsed.epochs}")
    if not math.isfinite(parsed.lr) or parsed.lr <= 0:
        raise ValueError(f"lr must be a positive finite float, got {parsed.lr}")
    if not math.isfinite(parsed.weight_decay) or parsed.weight_decay < 0:
        raise ValueError(
            f"weight-decay must be a non-negative finite float, got {parsed.weight_decay}"
        )
    if parsed.max_grad_norm is not None:
        if not math.isfinite(parsed.max_grad_norm) or parsed.max_grad_norm < 0:
            raise ValueError(
                f"max-grad-norm must be a non-negative finite float, got {parsed.max_grad_norm}"
            )
    if parsed.max_seq_length < 1:
        raise ValueError(f"max-seq-length must be >= 1, got {parsed.max_seq_length}")
    if parsed.dataset_size is not None and parsed.dataset_size < 1:
        raise ValueError(f"dataset-size must be >= 1 if provided, got {parsed.dataset_size}")
    if not (0 <= parsed.seed < 2**32):
        raise ValueError(f"seed must be between 0 and {2**32 - 1}, got {parsed.seed}")
    if parsed.checkpoint_dir is not None and not str(parsed.checkpoint_dir).strip():
        raise ValueError("checkpoint-dir cannot be an empty path string if provided.")

    if parsed.grid_search:
        parse_float_grid(
            parsed.lr_grid,
            name="--lr-grid",
            min_val=0.0,
            allow_zero=False,
        )
        parse_float_grid(
            parsed.weight_decay_grid,
            name="--weight-decay-grid",
            min_val=0.0,
            allow_zero=True,
        )
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    seed_everything(args.seed)

    print("=" * 70)
    print(f"AMD SPARE BEDROOM: Virtual {args.parallelism.upper()} Training on ROCm (gfx_1151)")
    print("Supervised Fine-Tuning (SFT) for Causal Language Models")
    print("=" * 70)

    # 1. Initialize Virtual Grid
    grid = VirtualGrid.create(
        world_size=args.ranks,
        total_memory_gb=args.total_vram_gb,
        device_type=args.device,
        seed=args.seed,
    )

    per_rank_gb = grid.per_rank_memory_bytes / (1024**3)
    print(f"[*] Physical VRAM Budget: {args.total_vram_gb:.1f} GB")
    print(f"[*] Virtual World Size:   {grid.world_size} ranks")
    print(f"[*] Per-Rank Budget:      {per_rank_gb:.2f} GB per virtual device")
    print(f"[*] Active Device Type:   {grid.device_type.upper()}")
    print(f"[*] AMP Precision Mode:   {args.amp.upper()}")

    if grid.device_type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_detected = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] ROCm Physical Device: {gpu_name} ({vram_detected:.1f} GB detected)")
    print("-" * 70)

    # 2. Prepare Model
    print(f"[*] Instantiating Llama causal language model ({args.model_name})...")
    base_model = get_llama_model(
        model_name=args.model_name,
        use_tiny=args.use_tiny_llama,
    )
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Model Parameter Count: {param_count:,}")

    # 3. Create Datasets
    dataset = get_sft_dataset(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dataset_size=args.dataset_size,
        synthetic=args.synthetic,
        split="train",
    )
    eval_dataset = get_sft_dataset(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dataset_size=args.dataset_size,
        synthetic=args.synthetic,
        split="validation",
    )

    dataset_sized = cast(Sized, dataset)
    target_device = torch.device(grid.device_type)
    dataset_type = "Synthetic SFT" if args.synthetic else f"Instruction ({args.dataset_name})"
    print(f"[*] Dataset:              {dataset_type} ({len(dataset_sized)} samples)")
    has_max_steps = args.max_steps is not None and args.max_steps > 0
    max_steps_str = f"max {args.max_steps} steps/epoch" if has_max_steps else "full epoch"
    print(f"[*] Training configured for {args.epochs} epoch(s), {max_steps_str}")

    resolved_checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else ("checkpoints/best_sft_model" if args.grid_search else None)
    )
    checkpoint_saver: BestCheckpointSaver | None = None
    if resolved_checkpoint_dir is not None:
        tokenizer = get_tokenizer(args.model_name, synthetic=args.synthetic)
        checkpoint_saver = BestCheckpointSaver(
            checkpoint_dir=resolved_checkpoint_dir,
            metric_name=args.eval_metric,
            tokenizer=tokenizer,
        )

    # 4. Multi-Rank Grid Search Mode
    if args.grid_search:
        lrs = parse_float_grid(args.lr_grid, name="--lr-grid")
        weight_decays = parse_float_grid(args.weight_decay_grid, name="--weight-decay-grid")
        total_trials = len(lrs) * len(weight_decays)
        print(
            f"[*] Running VirtualGridSearch: {total_trials} trials across {grid.world_size} virtual ranks"
        )
        print("=" * 70)
        search = VirtualGridSearch(
            base_model=base_model,
            grid=grid,
            param_grid={"lr": lrs, "weight_decay": weight_decays},
            metric_name=args.eval_metric,
            amp_dtype=args.amp,
            max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            seed=args.seed,
            checkpoint_saver=checkpoint_saver,
        )
        result = search.run(
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_steps=args.max_steps,
            eval_fn=evaluate_causal_lm_model,
        )
        print(
            f"[*] Grid search completed {len(result.trials)} trials in {result.num_waves} wave(s)."
        )
        print(result.format_winner_announcement())
        return

    # 5. Distributed Model Parallelism Setup (FSDP or DDP)
    if args.parallelism == "fsdp":
        print(
            f"[*] Wrapping model across {grid.world_size} virtual ranks via VirtualFSDP (Sharded)..."
        )
        dist_model: Any = VirtualFSDP(base_model, grid=grid, amp_dtype=args.amp)
        per_rank_params = dist_model.get_rank_parameter_numel(0)
        pct = (per_rank_params / param_count) * 100
        print(
            f"[*] Memory Optimization: {per_rank_params:,} parameters stored per rank ({pct:.1f}% of full model)"
        )
    else:
        print(
            f"[*] Wrapping model across {grid.world_size} virtual ranks via VirtualDDP (Replicated)..."
        )
        dist_model = VirtualDDP(base_model, grid=grid, amp_dtype=args.amp)

    del base_model
    gc.collect()

    dist_model.train()
    optimizers = dist_model.create_optimizers(
        torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay
    )

    rank_loaders: dict[int, DataLoader[dict[str, torch.Tensor]]] = {}
    samplers: dict[int, VirtualRankSampler] = {}
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
    print("=" * 70)

    # 6. SFT Training Loop with Lockstep All-Reduce Synchronization
    for epoch in range(args.epochs):
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs}")
        dist_model.train()
        for rank in grid.ranks:
            samplers[rank].set_epoch(epoch)

        rank_iters = {rank: iter(rank_loaders[rank]) for rank in grid.ranks}
        step = 0
        epoch_start = time.perf_counter()

        while not has_max_steps or step < args.max_steps:
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

                forward_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                outputs: Any = dist_model.forward(rank, **forward_kwargs)
                loss = outputs.loss
                step_losses[rank] = float(loss.item())

                # Backward pass computes local gradients (with GradScaler when FP16 AMP is active)
                dist_model.backward(rank, loss)

            # B. Collective Gradient Synchronization
            dist_model.sync_gradients()

            # C. Lockstep Optimizer Step with Gradient Clipping via library seam
            grad_norm_val = 0.0
            grad_norms = dist_model.step(
                optimizers,
                max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            )
            if grad_norms is not None:
                grad_norm_val = grad_norms[grid.ranks[0]]

            # D. Verify Multi-Rank Weight Parity
            if hasattr(dist_model, "assert_parameters_synced"):
                synced = dist_model.assert_parameters_synced()
                if not synced:
                    raise RuntimeError(f"Parameter desynchronization detected at step {step}!")

            avg_loss = sum(step_losses.values()) / len(step_losses)
            loss_breakdown = ", ".join(
                f"R{rank_id}: {loss_val:.4f}" for rank_id, loss_val in step_losses.items()
            )
            print(
                f"  Step {step:02d} | Avg Loss: {avg_loss:.4f} | Grad Norm: {grad_norm_val:.4f} | Ranks: [{loss_breakdown}] | Mode: {args.parallelism.upper()}"
            )

        del rank_iters
        if "batches" in locals():
            del batches
        gc.collect()
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_time = time.perf_counter() - epoch_start
        print(f"--- Epoch {epoch + 1} completed in {epoch_time:.2f}s ---")

        eval_metrics = evaluate_causal_lm_epoch(
            dist_model=dist_model,
            eval_dataset=eval_dataset,
            batch_size=args.batch_size,
        )
        print(
            f"--- Epoch {epoch + 1} Eval | Eval Loss: {eval_metrics.eval_loss:.4f} | "
            f"Perplexity: {eval_metrics.perplexity:.4f} | Token Accuracy: {eval_metrics.token_accuracy:.4f} ---"
        )
        if checkpoint_saver is not None:
            saved = checkpoint_saver.save_if_best(
                model=dist_model,
                metrics=eval_metrics,
                epoch=epoch + 1,
                hyperparameters={"lr": args.lr, "weight_decay": args.weight_decay},
            )
            if saved:
                print(
                    f"[*] Saved new best checkpoint to '{args.checkpoint_dir}' "
                    f"({args.eval_metric}={checkpoint_saver.best_score:.4f})"
                )

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
