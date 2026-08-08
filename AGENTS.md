# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Mule-Hunt (`mule-hunt` on PyPI) is a graph-neural-network fraud-detection
system for UPI-style payment graphs. It ships a CLI (`upifraud`), a FastAPI
risk service with a dashboard, an MCP investigation server for coding agents,
and honest benchmark experiments — including documented negative results.

## Commands

- Lint: `.venv/bin/ruff check src tests`
- Tests: `.venv/bin/python -m pytest -q -p no:warnings`
- Reinstall after `pyproject.toml` changes: `uv pip install -e .`
- Run an experiment: `.venv/bin/upifraud <cmd> --help` for
  `benchmark`, `temporal`, `attack`, `query`, `mcp`, `demo`
- Fast, dependency-free smoke test: `make demo`

## Architecture

| Module | Role |
| --- | --- |
| `src/upifraud/generate.py` | Synthetic bursty transaction graph (per-ring time windows) |
| `src/upifraud/dataset.py` | CSV -> PyG `Data`, ring-aware splits, edge timestamps, snapshots |
| `src/upifraud/features.py` | Node/edge feature builders (structural, optional amount/cycle/temporal) |
| `src/upifraud/models.py` | GCN / GraphSAGE / GATv2 with Jumping Knowledge + edge head |
| `src/upifraud/train.py` | Training loop, calibration, cold-start fallback, edge loss |
| `src/upifraud/evaluate.py` | AUC/AP/brier, ring recovery, F1 operating point |
| `src/upifraud/cli.py` | All `upifraud` subcommands (argparse) |
| `src/upifraud/api.py` | FastAPI risk service + dashboard endpoints |
| `src/upifraud/assistant.py` | Grounded natural-language investigation reports |
| `src/upifraud/mcp_server.py` | MCP (stdio) tools over the risk graph |
| `frontend/` | Static dashboard HTML/JS served by the API |

## Conventions

- No code comments unless the user asks for them; prefer docstrings for
  public functions and honest caveats.
- Match existing style: type hints, `from __future__ import annotations`,
  f-strings, 100-char lines (ruff).
- Never commit: `data/`, `models/`, `runs/`, `bench_*/`, `dist/`, `.venv/`.
- `load_graph(data_dir)` requires a `pathlib.Path` — a `str` raises.
- Changing results-generating code requires updating `results/benchmark.json`
  and the README table in the same change.
- Every feature needs tests in `tests/`; the `lint-and-test` CI check must
  stay green (ruff + pytest).
- Reporting is honest by design: document negative results and explain why
  they occur instead of omitting them.

## Release process

`pyproject.toml` version -> PR -> merge -> tag `v0.X.Y` -> the
`.github/workflows/workflow.yml` job publishes to PyPI (trusted publishing)
and verifies the tag matches the version. Keep the version bump in the PR
that ships the feature.
