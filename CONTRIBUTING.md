# Contributing to Mule-Hunt

Thanks for helping out. This is a small, research-adjacent project — the fastest
way to help is a good issue, a reviewed PR, or a benchmark improvement.

## Ground rules

- **One PR per concern.** Small, reviewable diffs get merged fast.
- **Tests must pass** for any change touching code: `uv run pytest`.
- **Lint clean:** `uv run ruff check .` before pushing.
- **No secrets, no real financial data.** The pipeline is synthetic by design.
- If you change results-generating code, update `results/benchmark.json` and the
  README table in the same PR.

## Finding something to do

Check the [issues](https://github.com/sohamvjadhav/Mule-Hunt/issues) —
items labeled `good first issue` are picked for newcomers. Anything under
"Roadmap" in the README is fair game.

## Setup

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest
```

## Working on the benchmark

Benchmark runs are expensive (minutes per hardness level). Use the toy
generator for fast iteration:

```bash
upifraud demo --toy
```

and only run the full matrix when the numbers are ready to commit:

```bash
upifraud benchmark --root bench --rings 50 --test-rings 10
```

## Submitting

1. Fork the repo, branch off `main`.
2. Make the change, add a test where it makes sense.
3. Run `pytest` and `ruff`.
4. Open a PR describing what and why.

Maintainer reviews PRs on a rolling basis.
