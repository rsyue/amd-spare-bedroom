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
3. **`Communicator`**: Provides collective communication primitives (`all_reduce`, `broadcast`, `gather`, `scatter`, `barrier`).
4. **`VirtualDDP`**: A drop-in wrapper mirroring `torch.nn.parallel.DistributedDataParallel` that coordinates gradient reduction across virtual ranks.
5. **`VirtualSampler`**: Distributes non-overlapping, seedable dataset slices to each virtual rank.

---

## Tooling & Prerequisites

- **Python**: `>=3.12`
- **Environment Manager**: [`uv`](https://github.com/astral-sh/uv)
- **Hardware Target**: AMD ROCm GPU (initial target: `gfx_1151` / RDNA 3.5 / Radeon 8060S)
- **Monitoring**: `amdgpu_top` or `rocm-smi`

---

## Getting Started

### 1. Initialize Virtual Environment with `uv`

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 2. Install Dependencies

Install the core package along with development tools (linting, formatting, type checking, pytest):

```bash
uv sync --extra dev
```

For ROCm-accelerated PyTorch wheels (ROCm 6.2+):

```bash
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
```

---

## Hook Scripts

The repository includes a set of lightweight wrapper scripts in [`hooks/`](file:///home/rtotheich/amd-spare-bedroom/hooks) that execute commands through `uv run`:

| Script | Command | Purpose |
| :--- | :--- | :--- |
| [`hooks/run`](file:///home/rtotheich/amd-spare-bedroom/hooks/run) | `uv run python "$@"` | Run any Python script with the environment active |
| [`hooks/lint`](file:///home/rtotheich/amd-spare-bedroom/hooks/lint) | `uv run ruff check "$@"` | Run Ruff linter checks |
| [`hooks/format`](file:///home/rtotheich/amd-spare-bedroom/hooks/format) | `uv run ruff format "$@"` | Run Ruff code formatter |
| [`hooks/typecheck`](file:///home/rtotheich/amd-spare-bedroom/hooks/typecheck) | `uv run mypy "$@"` | Run Mypy static type checking |
| [`hooks/test`](file:///home/rtotheich/amd-spare-bedroom/hooks/test) | `uv run pytest "$@"` | Run test suite |
| [`hooks/check-all`](file:///home/rtotheich/amd-spare-bedroom/hooks/check-all) | Run all of the above | Full pre-commit / pre-push verification |

Usage example:
```bash
./hooks/run examples/train_bert_ddp.py --ranks 4 --epochs 3
./hooks/check-all
```

---

## Test-Driven Development (TDD) & Agreed Seams

In accordance with our [TDD Guidelines](file:///home/rtotheich/amd-spare-bedroom/AGENTS.md), development proceeds in vertical slices against pre-agreed seams:

- **Seam 1: `VirtualGrid` & Topology**: Allocation, rank assignments, and address space memory budgeting.
- **Seam 2: `Communicator` & `LockstepBarrier`**: Inter-rank barrier, tensor all-reduce, and synchronization invariants.
- **Seam 3: `VirtualSampler`**: Deterministic dataset partitioning without data leakage across virtual ranks.
- **Seam 4: `VirtualDDP`**: Gradient hook attachment, parameter synchronization, and backward pass all-reduce.

---

## Continuous Integration (CI)

A credential-free GitHub Actions workflow is configured in [`.github/workflows/ci.yml`](file:///home/rtotheich/amd-spare-bedroom/.github/workflows/ci.yml) to automatically run:
- Ruff linting and formatting verification.
- Mypy strict type checking across `src/` and `tests/`.
- Pytest suite on CPU / mocked virtual grids (GPU tests marked with `@pytest.mark.gpu` are skipped in standard CI runners).

---

## Author

Created by [Richard Yue](https://github.com/rsyue) ([@rsyue](https://github.com/rsyue)).

