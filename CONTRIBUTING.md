# Contributing to AMD Spare Bedroom

Thank you for your interest in contributing to **AMD Spare Bedroom**! This project provides virtual distributed training (DDP, FSDP) and multi-rank hyperparameter optimization across unified physical VRAM on AMD ROCm (targeting `gfx_1151` / Strix Halo).

---

## 1. Development Environment Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management and virtual environment orchestration.

### Prerequisites
- **Python**: `==3.12.*`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (`>=0.4.0`)
- **Git**

### Installation
1. Clone the repository and navigate into the root directory:
   ```bash
   git clone https://github.com/rsyue/amd-spare-bedroom.git
   cd amd-spare-bedroom
   ```

2. Initialize virtual environment and install dependencies:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync --extra dev
   ```

3. Configure environment settings:
   ```bash
   cp .env.example .env
   ```

---

## 2. Local Quality & Verification Commands

All quality assurance checks run directly through `uv run`. No custom shell scripts or wrapper hooks are required:

### Linting & Formatting
```bash
# Check code style and rules with Ruff:
uv run ruff check .

# Automatically apply Ruff lint fixes:
uv run ruff check --fix .

# Verify code formatting:
uv run ruff format --check .

# Auto-format codebase:
uv run ruff format .
```

### Static Type Checking
Strict type checking is enforced across `src/` and `tests/` using Mypy:
```bash
uv run mypy src tests
```

### Unit Tests
The test suite covers communication primitives, mock grids, sampler logic, and metrics without requiring GPU hardware:
```bash
# Run all CPU / non-GPU tests:
uv run pytest -m "not gpu"

# Run tests with test coverage reporting:
uv run pytest -m "not gpu" --cov=src --cov-report=term-missing
```

### Fast Local Smoke Tests
Run end-to-end integration passes using compact 2-layer models and synthetic data:
```bash
# BERT sequence classification smoke test:
uv run python examples/train_bert_glue.py --synthetic --use-tiny-bert --ranks 2 --max-steps 2

# Llama 3.2 1B SFT smoke test:
uv run python examples/train_llama_sft.py --synthetic --use-tiny-llama --ranks 2 --max-steps 2
```

---

## 3. Test-Driven Development (TDD) & Architectural Seams

Development proceeds in vertical slices against pre-agreed architectural seams:

- **Seam 1: `VirtualGrid` & Topology**: Allocation, rank assignments, and address space memory budgeting.
- **Seam 2: `VirtualCommunicator` & `LockstepBarrier`**: Inter-rank barrier, tensor all-reduce, and synchronization invariants.
- **Seam 3: `VirtualRankSampler`**: Deterministic dataset partitioning without data leakage across virtual ranks.
- **Seam 4: `VirtualDDP`**: Gradient hook attachment, parameter synchronization, and backward pass all-reduce.
- **Seam 5: `VirtualFSDP` & Collective Sharding**: Parameter sharding (~`1/N` memory), on-demand All-Gather, and gradient Reduce-Scatter.
- **Seam 6: `ClassificationMetrics` & Scikit-Learn Adapters**: Zero-division protection, binary/multiclass auto-averaging, and `ClassificationMetrics` dataclass conversions.
- **Seam 7: Distributed Lockstep Evaluation (`evaluate_classification_epoch`)**: Multi-rank lockstep validation under `eval_forward()`, rank unpadding via `VirtualRankSampler.unstride_and_unpad`, and training state restoration.
- **Seam 8: Single Best-Metric Checkpoint Saving (`BestCheckpointSaver`)**: Monitored metric tracking, HuggingFace `safetensors` + tokenizer persistence, atomic directory overwrite, and winning trial formatting.
- **Seam 9: Hyperparameter Search Space Expansion & Wave Scheduler (`HyperparameterGrid`, `WaveSchedule`, `TrialConfig`)**: Deterministic Cartesian trial expansion, sequential wave partitioning across virtual ranks, and idle rank tracking for partial waves.
- **Seam 10: Multi-Rank Hyperparameter Grid Search Execution Engine (`VirtualGridSearch`, `TrialResult`, `GridSearchResult`)**: Per-rank replica and optimizer isolation, lockstep wave execution with step limits, evaluation aggregation, and single-best safetensors checkpointing.
- **Seam 11: CLI Hyperparameter Grid Search & Winner Announcement (`--grid-search`, `format_winner_announcement`)**: Multi-rank wave execution orchestration from CLI, single best-model safetensors persistence, and canonical final-line winner announcement format.
- **Seam 12: Causal LM SFT Pipeline & Next-Token Metrics (`SFTMetrics`, `evaluate_causal_lm_epoch`)**: Instruction fine-tuning for Llama 3.2 1B, Alpaca label masking (`ignore_index=-100`), validation perplexity, and masked token accuracy.
- **Seam 13: Configuration Schemas, Environment Abstraction & CLI Overrides (`ExperimentConfig`, `load_config`, `parse_args_with_config`, `load_dotenv`)**: Encapsulated configuration management in `amd_spare_bedroom.config`, Pydantic validation, hierarchical `.env` fallbacks, selective CLI defaults propagation, and declarative YAML experiment recipes.

---

## 4. Code Placement & Architectural Standards

Before submitting any code change:
- **Avoid Convenience Sinks**: Place logic where its invariants and domain responsibilities naturally belong. Do not dump unrelated utility code into existing modules.
- **Maintain Modular Boundaries**: Modules should have narrow, deep interfaces hiding internal implementation complexity.
- **Documentation Freshness**: Keep inline docstrings, type annotations, and documentation in sync with code behavior.

---

## 5. Pull Request Process

1. Create a descriptive feature branch:
   ```bash
   git checkout -b your-name/feature-description
   ```
2. Verify all quality gates pass locally before opening a PR:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src tests
   uv run pytest -m "not gpu"
   ```
3. Open a Pull Request against `main`. Continuous Integration in GitHub Actions will run parallel `lint`, `typecheck`, and `test` jobs.
