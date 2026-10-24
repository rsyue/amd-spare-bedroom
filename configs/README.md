# Configuration Reference & Recipes (`configs/`)

This directory provides pre-configured YAML experiment recipes for `amd-spare-bedroom`, along with an exhaustive reference of all configuration parameters, environment variables, and command-line flags.

---

## Quick Start

Run any pre-built experiment recipe with `uv run`:

```bash
# 1. BERT Sequence Classification on MRPC with VirtualDDP
uv run python examples/train_bert_glue.py --config configs/bert_mrpc_ddp.yaml

# 2. BERT Sequence Classification on MRPC with VirtualFSDP
uv run python examples/train_bert_glue.py --config configs/bert_mrpc_fsdp.yaml

# 3. Llama 3.2 1B Instruction Fine-Tuning with VirtualFSDP
uv run python examples/train_llama_sft.py --config configs/llama_sft_fsdp.yaml

# 4. Multi-Rank Hyperparameter Grid Search for BERT
uv run python examples/train_bert_glue.py --config configs/bert_mrpc_grid.yaml
```

### Overriding Parameters from CLI

Any parameter defined in a YAML configuration file can be overridden on the command line without modifying the file:

```bash
# Override batch size and learning rate on top of the bert_mrpc_ddp config:
uv run python examples/train_bert_glue.py --config configs/bert_mrpc_ddp.yaml --batch-size 16 --lr 5e-5

# Override number of epochs and target device:
uv run python examples/train_llama_sft.py --config configs/llama_sft_fsdp.yaml --epochs 1 --device cuda
```

---

## Configuration Precedence

When resolving settings at runtime, `amd-spare-bedroom` applies strict hierarchical precedence:

```
CLI Arguments  >  YAML Config File  >  Environment Variables (.env)  >  Built-in Code Defaults
 (Highest)                                                                    (Lowest)
```

1. **CLI Arguments**: Explicit flags passed to the script (e.g., `--batch-size 16`).
2. **YAML Config File**: Key-values defined in the file passed via `--config <path>`.
3. **Environment Variables**: Variables defined in `.env` or exported in the shell.
4. **Code Defaults**: Internal schema defaults defined in `src/amd_spare_bedroom/config/settings.py`.

---

## Available Pre-Built Configurations

| File | Workload | Parallelism | Target Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| [`bert_mrpc_ddp.yaml`](bert_mrpc_ddp.yaml) | GLUE MRPC | DDP (Replicated) | `bert-base-uncased` | 4 ranks, batch size 8, bf16 AMP, F1 eval |
| [`bert_mrpc_fsdp.yaml`](bert_mrpc_fsdp.yaml) | GLUE MRPC | FSDP (Sharded) | `bert-base-uncased` | 4 ranks, batch size 8, bf16 AMP, ~1/4 memory scaling |
| [`bert_mrpc_grid.yaml`](bert_mrpc_grid.yaml) | GLUE MRPC | Grid Search | `bert-base-uncased` | 10 ranks, search across 5 LRs x 5 weight decays x 2 batch sizes |
| [`llama_sft_ddp.yaml`](llama_sft_ddp.yaml) | Alpaca SFT | DDP (Replicated) | `meta-llama/Llama-3.2-1B` | 4 ranks, batch size 4, bf16 AMP, perplexity eval |
| [`llama_sft_fsdp.yaml`](llama_sft_fsdp.yaml) | Alpaca SFT | FSDP (Sharded) | `meta-llama/Llama-3.2-1B` | 4 ranks, batch size 4, parameter-sharded fine-tuning |
| [`llama_sft_grid.yaml`](llama_sft_grid.yaml) | Alpaca SFT | Grid Search | `meta-llama/Llama-3.2-1B` | 4 ranks, search across 3 LRs x 2 weight decays x 2 batch sizes |

---

## Parameter Catalog & Schema Reference

Configurations can be authored using either **nested sections** (e.g. `grid.ranks`) or **flat keys** (e.g. `ranks`). Both formats are automatically parsed and validated by `ExperimentConfig`.

### 1. Virtual Grid & Hardware Topology

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `grid.ranks` (`ranks`) | `--ranks` | - | `int` | `4` | Number of virtual devices/ranks to emulate in the unified memory pool. |
| `grid.total_vram_gb` (`total_vram_gb`) | `--total-vram-gb` | `AMD_SPARE_BEDROOM_TOTAL_VRAM_GB` | `float` | `96.0` | Total physical VRAM allocation budget in GB. |
| `grid.device` (`device`) | `--device` | `AMD_SPARE_BEDROOM_DEVICE` | `str` | `"auto"` | Target execution device backend (`"auto"`, `"cuda"`, or `"cpu"`). |
| `grid.seed` (`seed`) | `--seed` | `AMD_SPARE_BEDROOM_SEED` | `int` | `42` | Global random seed for deterministic sampling and initialization. |

### 2. Model & Tokenization

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `model.model_name` (`model_name`) | `--model-name` | - | `str` | `"bert-base-uncased"` | HuggingFace pretrained model name or local path. |
| `model.use_tiny` (`use_tiny`) | `--use-tiny-bert`, `--use-tiny-llama` | - | `bool` | `false` | Use compact 2-layer architecture for fast smoke tests and CI. |
| `model.max_seq_length` (`max_seq_length`) | `--max-seq-length`, `--seq-len` | - | `int` | `128` (BERT) / `256` (SFT) | Maximum sequence length after tokenization with padding. |

### 3. Dataset & Data Loading

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data.task_name` (`task_name`) | `--task-name`, `--task` | - | `str` | `"mrpc"` | GLUE benchmark classification task (`"mrpc"`, `"sst2"`, `"cola"`, etc.). |
| `data.dataset_name` (`dataset_name`) | `--dataset-name` | - | `str` | `"yahma/alpaca-cleaned"` | HuggingFace instruction dataset for causal LM SFT. |
| `data.dataset_size` (`dataset_size`) | `--dataset-size` | - | `int \| null` | `null` | Sample cap limit (defaults to full split or 64 if synthetic). |
| `data.synthetic` (`synthetic`) | `--synthetic`, `--use-synthetic` | - | `bool` | `false` | Generate fast in-memory synthetic data without network downloads. |

### 4. Training & Optimization

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `training.epochs` (`epochs`) | `--epochs` | - | `int` | `2` | Number of full training epochs to execute. |
| `training.batch_size` (`batch_size`) | `--batch-size` | - | `int` | `8` (BERT) / `4` (SFT) | Per-rank mini-batch size. |
| `training.lr` (`lr`) | `--lr` | - | `float` | `2.0e-5` | Optimizer learning rate for AdamW. |
| `training.weight_decay` (`weight_decay`) | `--weight-decay` | - | `float` | `0.01` | Weight decay regularization for AdamW. |
| `training.max_grad_norm` (`max_grad_norm`) | `--max-grad-norm` | - | `float \| null` | `1.0` | Maximum gradient norm threshold for clipping (`<= 0` to disable). |
| `training.max_steps` (`max_steps`) | `--max-steps` | - | `int` | `5` | Maximum training steps per epoch (`<= 0` for full epoch). |
| `training.amp` (`amp`) | `--amp`, `--precision` | - | `str` | `"none"` | Automatic mixed precision (`"none"`, `"bf16"`, `"fp16"`). |

### 5. Parallelism Mode

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `parallelism.mode` (`parallelism`) | `--parallelism` | - | `str` | `"fsdp"` | Distributed mode (`"fsdp"` for sharded parameters, `"ddp"` for full replication). |

### 6. Checkpoint Persistence

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `checkpoint.checkpoint_dir` (`checkpoint_dir`) | `--checkpoint-dir`, `--checkpoint-path` | - | `str \| null` | `null` | Output directory path to persist the single best safetensors model + tokenizer. |
| `checkpoint.eval_metric` (`eval_metric`) | `--eval-metric` | - | `str` | `"f1"` (BERT) / `"perplexity"` (SFT) | Monitored evaluation metric for saving winning checkpoint (`f1`, `accuracy`, `perplexity`, `loss`). |

### 7. Multi-Rank Grid Search

| YAML Key | CLI Flag | Env Fallback | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `grid_search.enabled` (`grid_search`) | `--grid-search` | - | `bool` | `false` | Enable multi-rank hyperparameter grid search across sequential waves. |
| `grid_search.params.lr` | `--lr-grid` | - | `list[float]` | `[1e-5, 2e-5, 3e-5, 5e-5, 1e-4]` | Candidate learning rates to explore. |
| `grid_search.params.weight_decay` | `--weight-decay-grid` | - | `list[float]` | `[0.0, 0.001, 0.01, 0.05, 0.1]` | Candidate weight decays to explore. |
| `grid_search.params.batch_size` | `--batch-size-grid` | - | `list[int]` | `null` | Candidate batch sizes to evaluate per rank. |
| `grid_search.params.max_grad_norm` | `--max-grad-norm-grid` | - | `list[float]` | `null` | Candidate gradient clipping thresholds to evaluate. |

---

## Writing Custom Recipes

To create a custom experiment, create a new YAML file in `configs/` or any local directory:

```yaml
# configs/my_custom_experiment.yaml
grid:
  ranks: 4
  total_vram_gb: 96.0

training:
  epochs: 3
  batch_size: 16
  lr: 3.0e-5
  amp: bf16

model:
  model_name: bert-base-uncased

data:
  task_name: sst2

parallelism:
  mode: fsdp

checkpoint:
  checkpoint_dir: checkpoints/my_sst2_model
  eval_metric: accuracy
```

Run it directly with:

```bash
uv run python examples/train_bert_glue.py --config configs/my_custom_experiment.yaml
```
