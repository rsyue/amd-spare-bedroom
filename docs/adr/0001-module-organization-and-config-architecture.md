# ADR-0001: Encapsulate Configuration & Environment in Deep Submodule

## Status

Accepted

## Context

Prior to this decision, `src/amd_spare_bedroom/` contained 9 flat Python modules:
- `grid.py`: Virtual grid cluster topology and isolated device address spaces.
- `communicator.py`: Collective communication engine and lockstep barriers.
- `ddp.py`: Distributed Data Parallelism with gradient reduction.
- `fsdp.py`: Fully Sharded Data Parallelism with collective sharding.
- `amp.py`: Automatic mixed precision and loss scaling coordination.
- `sampler.py`: Deterministic virtual rank data partitioning.
- `metrics.py`: Classification and causal LM evaluation metrics.
- `checkpoint.py`: Single best-metric HuggingFace safetensors persistence.
- `grid_search.py`: Multi-rank hyperparameter grid search engine.

When introducing experiment configuration management (Pydantic schemas, YAML parsing, CLI overrides, and `.env` discovery), placing these new files directly in the root of `src/amd_spare_bedroom/` would increase folder density to 12+ flat files. According to our Code Placement guidelines, flat directories exceeding 7–10 files with mixed abstraction levels lead to cognitive sprawl and erosion of boundaries.

We evaluated two architectural alternatives:
1. **Full Domain Partitioning**: Move all existing files into nested subpackages (e.g. `src/amd_spare_bedroom/core/`, `distributed/`, `search/`, `config/`).
2. **Targeted Submodule Encapsulation**: Keep the existing 9 domain modules at the package root to preserve established seams and backwards compatibility, while encapsulating all configuration, environment, and CLI parsing logic in a dedicated deep module `src/amd_spare_bedroom/config/`.

## Decision

We chose **Alternative 2: Targeted Submodule Encapsulation**:
- Encapsulate the configuration subsystem in a deep subpackage: `src/amd_spare_bedroom/config/`
  - `src/amd_spare_bedroom/config/env.py`: `.env` file discovery, parser, and runtime environment variable abstraction.
  - `src/amd_spare_bedroom/config/settings.py`: Pydantic models (`ExperimentConfig`, `GridConfig`, `TrainingConfig`, etc.) and YAML deserializer.
  - `src/amd_spare_bedroom/config/cli.py`: CLI `--config` integration, automatic `.env` loading, and parameter overrides.
- Expose clean entry points through `src/amd_spare_bedroom/config/__init__.py`:
  - `load_config(path: str | Path) -> ExperimentConfig`
  - `parse_args_with_config(parser, args) -> argparse.Namespace`
  - `load_dotenv(path=None, override=False) -> dict[str, str]`
  - `ExperimentConfig` and component schemas.
- Preserve existing root domain modules and top-level public exports in `src/amd_spare_bedroom/__init__.py`.

## Consequences

### Positive
- **Seam Stability**: Existing test suites, examples, and user scripts importing from `amd_spare_bedroom` or `amd_spare_bedroom.ddp` remain 100% backward compatible without breaking changes.
- **Deep Module Architecture**: The configuration subsystem hides complex YAML mapping, Pydantic type conversions, and precedence resolution behind a narrow, simple interface (`parse_args_with_config`, `load_config`).
- **Controlled Density**: Directory density in `src/amd_spare_bedroom/` is maintained without sprawl.

### Negative / Trade-offs
- The root of `src/amd_spare_bedroom/` retains its 9 domain modules alongside the new `config/` subpackage. Should the core execution engine expand further, future ADRs may partition the execution engines into `src/amd_spare_bedroom/distributed/`.
