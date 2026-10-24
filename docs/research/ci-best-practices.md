# Continuous Integration Best Practices: uv, Ruff, Mypy, and Pytest

This document synthesizes primary-source recommendations and industry standards for Continuous Integration (CI) in modern Python libraries using **uv**, **Ruff**, **Mypy**, and **Pytest**.

---

## 1. Official Astral & `uv` GitHub Actions Integration

### 1.1 The `astral-sh/setup-uv` Action
Astral provides the official GitHub Action [`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv) for setting up `uv` in CI workflows ([Astral Docs: GitHub Integration](https://docs.astral.sh/uv/guides/integration/github/)).

Key configuration capabilities:
* **Version Management:** Pinning the action (e.g. `astral-sh/setup-uv@v5`) ensures reproducible workflows. Specifying `version: "0.5.x"` locks the `uv` binary, while omitting `version` auto-detects required versions from `pyproject.toml` or `uv.toml`.
* **Automatic Python Provisioning:** By supplying `python-version: "3.12"`, `setup-uv` sets the `UV_PYTHON` environment variable. `uv` will automatically download and manage standalone CPython builds on demand, eliminating the need for `actions/setup-python` or explicit `uv python install` steps.
* **Built-in Caching (`enable-cache: true`):** Automatically saves and restores `uv`'s package cache (`~/.cache/uv`), keyed by default against `uv.lock`.

### 1.2 Lockfile Synchronization: `--locked` vs `--frozen`
In CI pipelines, dependency resolution must be deterministic and verifiable ([Astral Docs: Sync CLI](https://docs.astral.sh/uv/concepts/projects/sync/)):
* **`uv sync --locked` (Recommended for CI):** Asserts that `uv.lock` is in exact synchronization with `pyproject.toml`. If a developer modified dependencies in `pyproject.toml` but neglected to update `uv.lock`, `--locked` fails immediately. This prevents accidental silent re-resolution in CI.
* **`uv sync --frozen`:** Installs strictly from `uv.lock` without checking if `pyproject.toml` was modified. This is suitable for Docker multi-stage builds where `pyproject.toml` may not be copied yet, but `--locked` is superior for CI gates.

### 1.3 Caching Mechanics & Avoiding Cache Collisions
* **Parallel Job Isolation (`cache-suffix`):** When multiple jobs run concurrently (such as separate `lint`, `typecheck`, and `test` jobs), they attempt to save to the GitHub Actions cache at the end of their run. If all jobs share the same cache key, GitHub Actions can throw a non-retryable `409 Conflict` error. Using `cache-suffix` isolates cache namespaces per job (e.g., `cache-suffix: "lint"`).
* **Cache Optimization (`prune-cache: true`):** Running `uv cache prune --ci` (or configuring `prune-cache: true` in `setup-uv`) removes pre-built downloaded wheels while preserving wheels compiled from source ([Astral Docs: Cache Pruning](https://docs.astral.sh/uv/concepts/cache/)). Because pre-built wheels can be quickly re-fetched, pruning avoids hitting GitHub's 10 GB repository cache quota.

---

## 2. CI Workflow Architecture & Job Structuring

### 2.1 Multi-Job Architecture vs Single-Job
Professional Python libraries structure CI into discrete parallel jobs:

| Criterion | Multi-Job (`lint`, `typecheck`, `test`) | Single Combined Job |
| :--- | :--- | :--- |
| **Feedback Latency** | **Fast (seconds):** Lint issues surface in ~5s without waiting for Mypy or Pytest. | **Slow:** Lint failure waits on dependency installation or previous steps. |
| **Failure Triage** | Clear visual badges in PR UI (shows exactly whether formatting, typing, or tests broke). | Opaque; contributor must scroll through mixed logs. |
| **Branch Protection** | Granular required status checks per concern. | Monolithic pass/fail. |
| **Matrix Scalability** | Test matrix (`3.11`, `3.12`, `3.13`) runs without duplicating lint/typecheck runs. | Matrix re-runs linters redundantly on every Python version. |

### 2.2 Tool-Specific Best Practices

#### Ruff (Linting & Formatting)
* **Execution:** Run both linter and formatter verification:
  ```bash
  uv sync --locked --only-group dev --no-install-project
  uv run --locked --no-sync ruff check --output-format=github .
  uv run --locked --no-sync ruff format --check .
  ```
* **Dependency Minimization:** Ruff is a standalone Rust binary that does not depend on project runtime dependencies or heavy PyTorch/ROCm wheels. Synchronizing only `--only-group dev --no-install-project` speeds up CI runner setup and eliminates network vulnerability to external wheel registries.
* **GitHub Annotations:** The `--output-format=github` flag outputs workflow command syntax (`::error file=...`), creating native inline code annotations directly on GitHub PR diffs ([Ruff Docs: Output Formats](https://docs.astral.sh/ruff/configuration/#output-format)).

#### Mypy (Static Type Checking)
* **Execution:**
  ```bash
  uv sync --locked
  uv run --locked mypy src tests
  ```
* **Dependency Context:** Unlike Ruff, Mypy performs inter-module type checking and inspects type hints across external dependencies (`torch`, `pydantic`, `transformers`). Therefore, `uv sync --locked` must be executed before Mypy.

#### Pytest (Unit Tests & Coverage)
* **Execution:**
  ```bash
  uv sync --locked
  uv run --locked pytest -m "not gpu" --cov=src --cov-report=term-missing
  ```
* **Hardware Marker Filtering:** For libraries supporting specialized hardware (e.g. AMD ROCm / GPUs), CPU-only runners must deselect hardware tests (`-m "not gpu"`).
* **Matrix Testing:** For libraries supporting multiple Python runtimes, define `strategy: matrix: python-version: ["3.11", "3.12"]` on the `test` job only. (If the library explicitly pins a single version, such as `requires-python = "==3.12.*"`, matrixing is omitted).

### 2.3 Workflow Triggers and Concurrency Management
* **Events:** Trigger on `push` to `main`/`master` and `pull_request` targeting `main`/`master`.
* **Safe Concurrency Pattern:**
  ```yaml
  concurrency:
    group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}
    cancel-in-progress: true
  ```
  * For **Pull Requests**: `github.event.pull_request.number` evaluates to the unique PR number. If a developer pushes three commits in rapid succession, obsolete runs are automatically cancelled, saving runner minutes. Keying on PR number avoids collision across external forks with common branch names (such as `main` or `patch-1`).
  * For **Main Branch Pushes**: `github.event.pull_request.number` is empty, falling back to `github.run_id` (globally unique). Pushes and merge commits to `main` will *never* cancel each other and will always complete.

---

## 3. Local Git Hooks vs. CI: Modern Consensus

### 3.1 Why Custom `./hooks` Shell Scripts are an Anti-Pattern
Some repositories maintain a directory of custom bash wrapper scripts (e.g., `./hooks/lint`, `./hooks/format`, `./hooks/test`, `./hooks/check-all` containing `exec uv run <tool> "$@"`). In modern engineering, this pattern is considered unnecessary overhead and an anti-pattern:

1. **The "Pseudo-Hook" Deception:** Git natively reads hooks exclusively from `.git/hooks/`. A root `./hooks/` folder is **not** an active Git hook unless explicitly activated by `git config core.hooksPath hooks`. This misleads contributors and automated agents into believing hooks are enforcing quality locally when they are completely inert.
2. **Redundant Indirection:** `uv` is already a fast tool runner. Writing a 3-line bash script to wrap `uv run ruff check "$@"` introduces a layer of indirection that adds maintenance burden while providing zero architectural value.
3. **Cross-Platform Incompatibility:** Shell scripts starting with `#!/usr/bin/env bash` fail on native Windows environments (PowerShell / cmd.exe) without WSL or Git Bash. `uv` is a self-contained, statically compiled Rust binary that functions identically across Windows, macOS, and Linux.
4. **Configuration Drift:** Shell wrappers inevitably drift from CI. For example, CI may enforce `pytest -m "not gpu" --cov=src`, while `./hooks/test` executes `pytest "$@"` (which would fail on machines without GPU hardware).
5. **No Unified Task Interface:** Custom shell scripts lack argument validation, auto-documentation (`--help`), and standardized listing.

### 3.2 Division of Responsibilities
* **Local Hooks / Dev Workflows (Shift-Left Convenience):**
  * Fast feedback: Local checks must execute in <2 seconds (e.g. Ruff formatting and linting).
  * Can always be bypassed by developers (`git commit --no-verify`), web edits, or AI agents.
  * Standard alternative: If automated Git hooks are desired, use declarative `pre-commit` managed via `uvx pre-commit`, or standard task runners like `just` (`Justfile`).
* **Continuous Integration (Immutable Gatekeeper):**
  * CI is the single source of truth.
  * Runs on pristine, reproducible runners with locked dependencies (`uv sync --locked`).
  * Enforces branch protection, coverage thresholds, multi-OS/multi-Python matrices, and hardware tests.

---

## 4. Production-Ready CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]

concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.run_id }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint & Format (Ruff)
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-suffix: "lint"
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --locked

      - name: Run Ruff Lint (with GitHub PR Annotations)
        run: uv run --locked ruff check --output-format=github .

      - name: Verify Ruff Code Formatting
        run: uv run --locked ruff format --check .

  typecheck:
    name: Typecheck (Mypy)
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-suffix: "typecheck"
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --locked

      - name: Run Mypy Type Checking
        run: uv run --locked mypy src tests

  test:
    name: Unit Tests (Pytest)
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-suffix: "test"
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --locked

      - name: Run Pytest Suite (CPU / Non-GPU)
        run: uv run --locked pytest -m "not gpu" --cov=src --cov-report=term-missing
```

---

## 5. Primary References & Documentation

1. **Astral `uv` GitHub Integration Guide:**  
   [https://docs.astral.sh/uv/guides/integration/github/](https://docs.astral.sh/uv/guides/integration/github/)
2. **`astral-sh/setup-uv` Official GitHub Action:**  
   [https://github.com/astral-sh/setup-uv](https://github.com/astral-sh/setup-uv)
3. **Astral `uv` Projects & Lockfiles (`--locked` vs `--frozen`):**  
   [https://docs.astral.sh/uv/concepts/projects/sync/](https://docs.astral.sh/uv/concepts/projects/sync/)
4. **Astral `uv` Cache Management (`uv cache prune --ci`):**  
   [https://docs.astral.sh/uv/concepts/cache/](https://docs.astral.sh/uv/concepts/cache/)
5. **Ruff GitHub Actions Integration & Output Formats:**  
   [https://docs.astral.sh/ruff/configuration/#output-format](https://docs.astral.sh/ruff/configuration/#output-format)
6. **Mypy Continuous Integration Practices:**  
   [https://mypy.readthedocs.io/en/stable/running_mypy.html](https://mypy.readthedocs.io/en/stable/running_mypy.html)
