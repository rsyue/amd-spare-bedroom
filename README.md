# AMD Spare Bedroom (`amd-spare-bedroom`)

Emulate multiple AMD GPU devices and distributed ranks (DDP, FSDP) with isolated address spaces inside a unified high-capacity VRAM pool on AMD ROCm (targeting `gfx_1151` / AMD Strix Halo).

---

## Overview

This repo positions itself as a playground for parallelism techniques for those who might have more VRAM than they need for a given workflow. Think of a [spare bedroom](https://advancedhouseplans.com/blogs/the-best-things-to-do-with-your-spare-bedroom) you might have in your house. If no one's using it, you could put a home office, gym or art studio in there. Well this repo is kind of like that. Except the goal is to experiment with different virtual devices in lockstep across the unified VRAM you have to spare.

Modern unified-memory APUs—such as the **AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx_1151`)**—provide massive VRAM allocations (up to **96 GB** or **128 GB** visible via `amdgpu_top` and `rocm-smi`). 

While this hardware typically appears to the operating system as a single monolithic GPU (`cuda:0` / `hip:0`), 96 GB is more than enough memory to host entire multi-node distributed training topologies.

**`amd-spare-bedroom`** partitions this unified physical VRAM into an emulated grid of virtual AMD GPU ranks:
- **Virtual Devices & Address Spaces**: Each rank operates inside its own bounded virtual address space and memory quota.
- **True Distributed Mechanics**: Replicates Distributed Data Parallel (**DDP**) and Fully Sharded Data Parallel (**FSDP**) mechanisms without requiring a multi-GPU cluster.
- **Lockstep Execution & Synchronization**: Synchronizes forward, backward, and optimizer passes across ranks using lockstep execution barriers and collective all-reduce operations.
- **Rank-Isolated Data Pipelines**: Feeds each virtual device with rank-specific, seedable datasets and mini-batch shards.
- **Real-World Workloads**: Prototype and validate real-world models (e.g., training `bert-base-uncased` on GLUE tasks) locally under exact distributed conditions.

---

## Key Concepts & Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Unified Physical VRAM (96 GB, gfx_1151)                   │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  Virtual Rank 0 (Address Space 0)    │  Virtual Rank 1 (Address Space 1)    │
│  - Model Replica                     │  - Model Replica                     │
│  - Sharded Dataset Batch 0           │  - Sharded Dataset Batch 1           │
│  - Local Gradients                   │  - Local Gradients                   │
├──────────────────────────────────────┴──────────────────────────────────────┤
│                    Lockstep Barrier & Communicator                          │
│         - Gradient All-Reduce (Ring / Direct Unified Reduction)             │
│         - Rank Parameter Synchronization                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **`VirtualGrid`**: Defines the virtual cluster topology (world size, virtual ranks, memory quota per rank, device backend).
2. **`VirtualDevice`**: Represents an emulated GPU rank with isolated tensor namespaces, memory limits, and stream coordination.
3. **`VirtualCommunicator`**: Provides collective communication primitives (`all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, `barrier`).
4. **`VirtualDDP`**: A drop-in wrapper mirroring `torch.nn.parallel.DistributedDataParallel` that coordinates gradient reduction across virtual ranks.
5. **`VirtualFSDP`**: Fully Sharded Data Parallel wrapper sharding parameters, gradients, and optimizer states across virtual ranks, executing All-Gather for forward passes and Reduce-Scatter for gradients to reduce memory footprint by $\sim 1 / \text{world\_size}$.
6. **`VirtualRankSampler`**: Distributes non-overlapping, seedable dataset slices to each virtual rank.

---

## Tooling & Prerequisites

- **Python**: `>=3.12`
- **Environment Manager**: [`uv`](https://github.com/astral-sh/uv)
- **Hardware Target**: AMD ROCm GPU (initial target: `gfx_1151` / RDNA 3.5 / Radeon 8060S with up to 96 GB VRAM)
- **Monitoring**: `amdgpu_top` or `rocm-smi`

---

## Getting Started

### 1. Initialize Virtual Environment with `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 2. Install Dependencies

The project is pre-configured in [`pyproject.toml`](pyproject.toml) with the official AMD ROCm nightly index for `gfx1151` (`https://rocm.nightlies.amd.com/v2/gfx1151`). Simply sync the environment:

```bash
uv sync --extra dev
```

To manually install or update ROCm PyTorch wheels for `gfx_1151`:

```bash
uv pip install --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151 torch
```

---

## Running Examples

An end-to-end multi-rank BERT classification training script is provided in [`examples/train_bert_glue.py`](examples/train_bert_glue.py), supporting both **FSDP** (parameter sharded) and **DDP** (replicated) on actual GLUE benchmarks or fast in-memory synthetic datasets:

```bash
# Run with actual GLUE (MRPC) across 4 virtual ranks using VirtualFSDP:
./hooks/run examples/train_bert_glue.py --parallelism fsdp --use-tiny-bert --ranks 4 --epochs 2 --max-steps 5

# Run with VirtualDDP (replicated model with All-Reduce gradient synchronization):
./hooks/run examples/train_bert_glue.py --parallelism ddp --use-tiny-bert --ranks 4 --epochs 2 --max-steps 5

# Fast smoke test using synthetic dataset (no network/download required):
./hooks/run examples/train_bert_glue.py --synthetic --use-tiny-bert --ranks 2 --max-steps 2

# Full BERT-base sequence classification across 8 virtual ranks with FSDP on SST-2:
./hooks/run examples/train_bert_glue.py --parallelism fsdp --ranks 8 --batch-size 4 --epochs 3 --task-name sst2
```

Key CLI options:
- `--parallelism`: Distributed mode (`fsdp` for sharded memory, `ddp` for replicated; default: `fsdp`).
- `--synthetic` / `--use-synthetic`: Use fast in-memory synthetic dataset instead of downloading GLUE.
- `--task-name` / `--task`: GLUE task (`mrpc`, `sst2`, `cola`, `qqp`, `mnli`, `qnli`, `rte`, `wnli`, `stsb`; default: `mrpc`).
- `--dataset-size`: Sample cap (default: full dataset for GLUE, 64 for synthetic).
- `--max-seq-length` / `--seq-len`: Maximum sequence length (default: 128).
- `--max-steps`: Maximum training steps per epoch (`<= 0` for full epoch; default: 5).
- `--epochs`: Number of training epochs (default: 2).
- `--batch-size`: Per-rank batch size (default: 8).
- `--lr`: Optimizer learning rate (default: 2e-5).
- `--max-grad-norm`: Maximum gradient norm threshold for clipping (default: 1.0).
- `--ranks`: Number of virtual devices/ranks to emulate in the grid (default: 4).
- `--total-vram-gb`: Physical VRAM memory pool allocation in GB (default: 96.0).
- `--device`: Target device type (`cuda`, `cpu`, or auto-detected; default: auto-detect).
- `--model-name`: HuggingFace model identifier (default: `bert-base-uncased`).
- `--use-tiny-bert`: Use compact 2-layer BERT for fast local verification.
- `--seed`: Random seed for data shuffling and sampling (default: 42).

---

## Hook Scripts

The repository includes a set of lightweight wrapper scripts in [`hooks/`](hooks/) that execute commands through `uv run`:

| Script | Command | Purpose |
| :--- | :--- | :--- |
| [`hooks/run`](hooks/run) | `uv run python "$@"` | Run any Python script with the environment active |
| [`hooks/lint`](hooks/lint) | `uv run ruff check "$@"` | Run Ruff linter checks |
| [`hooks/format`](hooks/format) | `uv run ruff format "$@"` | Run Ruff code formatter |
| [`hooks/typecheck`](hooks/typecheck) | `uv run mypy "$@"` | Run Mypy static type checking |
| [`hooks/test`](hooks/test) | `uv run pytest "$@"` | Run test suite |
| [`hooks/check-all`](hooks/check-all) | Run all of the above | Full pre-commit / pre-push verification |

Usage example:
```bash
./hooks/run examples/train_bert_glue.py --parallelism fsdp --use-tiny-bert --ranks 4 --epochs 2
./hooks/check-all
```

---

## Test-Driven Development (TDD) & Agreed Seams

In accordance with our [TDD Guidelines](AGENTS.md), development proceeds in vertical slices against pre-agreed seams:

- **Seam 1: `VirtualGrid` & Topology**: Allocation, rank assignments, and address space memory budgeting.
- **Seam 2: `VirtualCommunicator` & `LockstepBarrier`**: Inter-rank barrier, tensor all-reduce, and synchronization invariants.
- **Seam 3: `VirtualRankSampler`**: Deterministic dataset partitioning without data leakage across virtual ranks.
- **Seam 4: `VirtualDDP`**: Gradient hook attachment, parameter synchronization, and backward pass all-reduce.
- **Seam 5: `VirtualFSDP` & Collective Sharding**: Parameter sharding ($\sim 1/N$ memory), on-demand All-Gather, and gradient Reduce-Scatter.

---

## Continuous Integration (CI)

A credential-free GitHub Actions workflow is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) to automatically run:
- Ruff linting and formatting verification.
- Mypy strict type checking across `src/` and `tests/`.
- Pytest suite on CPU / mocked virtual grids (GPU tests marked with `@pytest.mark.gpu` are skipped in standard CI runners).

---

## Author

Created by [Richard Yue](https://github.com/rsyue) ([@rsyue](https://github.com/rsyue)).

