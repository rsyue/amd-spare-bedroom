# Glossary

Canonical domain vocabulary for `amd-spare-bedroom`. Use these terms consistently in code, tests, docs, and issue discussions.

## Terms

### Virtual Grid
The simulated multi-rank cluster topology mapped onto a single physical AMD ROCm GPU (such as `gfx1151` with 96 GB unified VRAM). Defines the world size, rank membership, and total memory allocation.

### Virtual Rank
An integer identifier (`0 <= rank < world_size`) assigned to an emulated worker process/thread within the Virtual Grid.

### Virtual Device & Address Space
The isolated runtime context allocated to a single Virtual Rank. Enforces rank-scoped memory limits and namespace separation on top of the underlying physical ROCm device (`hip:0` / `cuda:0`).

### Lockstep Barrier
A synchronization barrier ensuring all virtual ranks execute forward, backward, and optimization steps in deterministic lockstep, simulating synchronized multi-device execution.

### Virtual Communicator
The collective communications engine (implemented in `VirtualCommunicator`) providing inter-rank primitives (`all_reduce`, `broadcast`, `gather`, `scatter`, `barrier`). Leverages zero-copy / unified memory buffers on APUs for fast local reductions.

### Virtual DDP (Distributed Data Parallel)
Emulation of PyTorch's `DistributedDataParallel`. Holds a full model replica per virtual rank, feeds rank-sharded data, hooks backward gradients to trigger All-Reduce across ranks, and executes synchronized parameter updates.

### Virtual FSDP (Fully Sharded Data Parallel)
Emulation of PyTorch's FSDP (implemented in `VirtualFSDP`). Shards model parameters, gradients, and optimizer states across virtual ranks, executing all-gather before forward/backward computations and reduce-scatter on gradients to achieve ~`1 / world_size` per-rank memory scaling.

### Virtual Rank Sampler
A deterministic, seedable dataset partitioner that produces non-overlapping batch splits for each Virtual Rank without data duplication or cross-rank leakage.

### Classification Metrics
Structured evaluation scores (`ClassificationMetrics`: `accuracy`, `f1`, `precision`, `recall`, `loss`) computed via `scikit-learn` (`compute_classification_metrics`) with automatic binary/multiclass resolution and zero-division protection.

### Distributed Epoch Evaluation
Lockstep multi-rank validation evaluation (`evaluate_classification_epoch`) orchestrating `VirtualDDP` and `VirtualFSDP` under `torch.no_grad()`, reconstructing sequential dataset order and stripping wrap-around padding via `VirtualRankSampler.unstride_and_unpad`.

### Best-Metric Checkpoint Saver
A persistence manager (`BestCheckpointSaver`) that maintains a single loadable HuggingFace `safetensors` model (`model.safetensors`, `config.json`), tokenizer, and `checkpoint_metadata.json` directory, atomically overwriting it when the monitored metric improves (`f1` by default for classification) and formatting the winning hyperparameter summary (`BestCheckpointSaver.format_winner_summary`).

### Virtual Grid Search
A parallel hyperparameter exploration mode (`HyperparameterGrid`, `WaveSchedule`, `TrialConfig`) in which Cartesian hyperparameter trials are scheduled across a `VirtualGrid`'s virtual ranks in sequential execution waves of up to `world_size` concurrent trials.

## Architecture & Concept Relationships

```mermaid
flowchart TD
    VG["VirtualGrid\n(World Size & Memory Pool)"] --> VR["Virtual Rank\n(0 <= rank < world_size)"]
    VR --> VD["Virtual Device & Address Space\n(Rank Quota & Memory Isolation)"]
    VG --> VC["VirtualCommunicator & Barrier\n(All-Reduce, All-Gather, Reduce-Scatter)"]
    VG --> RS["VirtualRankSampler\n(Non-overlapping Data Slices)"]
    VD --> DDP["VirtualDDP\n(Replicated Models)"]
    VD --> FSDP["VirtualFSDP\n(Sharded ~1/N Memory)"]
    VC -. "Synchronizes Gradients" .-> DDP
    VC -. "Shards & Gathers" .-> FSDP
    DDP --> EVAL["evaluate_classification_epoch\n(Distributed Validation)"]
    FSDP --> EVAL
    RS -. "unstride_and_unpad" .-> EVAL
    EVAL --> METRICS["ClassificationMetrics\n(scikit-learn)"]
    METRICS --> CKPT["BestCheckpointSaver\n(Safetensors & Metadata)"]
    DDP -. "unwrap_model" .-> CKPT
    FSDP -. "unwrap_model" .-> CKPT
    VG --> HG["HyperparameterGrid\n(Cartesian Search Space)"]
    HG --> TC["TrialConfig\n(trial_id & params)"]
    HG --> WS["WaveSchedule\n(plan_waves)"]
    WS -. "assigns trials per wave" .-> VR
```
