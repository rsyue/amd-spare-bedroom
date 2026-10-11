# Glossary

Canonical domain vocabulary for `amd-spare-bedroom`. Use these terms consistently in code, tests, docs, and issue discussions.

## Terms

### Virtual Grid
The simulated multi-rank cluster topology mapped onto a single physical AMD ROCm GPU (such as `gfx1151` with 96 GB unified VRAM). Defines the world size, rank membership, and total memory allocation.

### Virtual Rank
An integer identifier ($0 \le \text{rank} < \text{world\_size}$) assigned to an emulated worker process/thread within the Virtual Grid.

### Virtual Device & Address Space
The isolated runtime context allocated to a single Virtual Rank. Enforces rank-scoped memory limits and namespace separation on top of the underlying physical ROCm device (`hip:0` / `cuda:0`).

### Lockstep Barrier
A synchronization barrier ensuring all virtual ranks execute forward, backward, and optimization steps in deterministic lockstep, simulating synchronized multi-device execution.

### Virtual Communicator
The collective communications engine (implemented in `VirtualCommunicator`) providing inter-rank primitives (`all_reduce`, `broadcast`, `gather`, `scatter`, `barrier`). Leverages zero-copy / unified memory buffers on APUs for fast local reductions.


### Virtual DDP (Distributed Data Parallel)
Emulation of PyTorch's `DistributedDataParallel`. Holds a full model replica per virtual rank, feeds rank-sharded data, hooks backward gradients to trigger All-Reduce across ranks, and executes synchronized parameter updates.

### Virtual FSDP (Fully Sharded Data Parallel)
Emulation of PyTorch's FSDP (implemented in `VirtualFSDP`). Shards model parameters, gradients, and optimizer states across virtual ranks, executing all-gather before forward/backward computations and reduce-scatter on gradients to achieve $\sim 1 / \text{world\_size}$ per-rank memory scaling.


### Virtual Rank Sampler
A deterministic, seedable dataset partitioner that produces non-overlapping batch splits for each Virtual Rank without data duplication or cross-rank leakage.
