# Agent Guidelines

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five-role triage vocabulary mapped directly 1:1 (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` at repo root). See `docs/agents/domain.md`.

## Development & Verification Standards

- **Environment & Command Runner**: Always execute commands via `uv run` (e.g. `uv run python ...`, `uv run pytest ...`). Do not create or depend on ad-hoc shell script wrappers in `./hooks`.
- **Quality Gates**:
  - **Lint**: `uv run ruff check .` (auto-fix with `--fix`).
  - **Format**: `uv run ruff format --check .` (auto-format with `uv run ruff format .`).
  - **Typecheck**: `uv run mypy src tests` (strict mode).
  - **Test**: `uv run pytest -m "not gpu"` (non-GPU unit test suite).
- **Contributing & Architectural Seams**: Consult `CONTRIBUTING.md` for pre-integration code placement and agreed vertical seams (Seams 1–13).
