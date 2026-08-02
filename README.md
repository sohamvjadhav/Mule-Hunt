# Mulehunt — UPI Fraud Detection with Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/sohamvjadhav/mulehunt/actions/workflows/ci.yml/badge.svg)](https://github.com/sohamvjadhav/mulehunt/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Graph-neural-network fraud detection for **UPI-style payment graphs**: model
transactions as a graph (accounts = nodes, transfers = edges) and detect
coordinated fraud rings — mule accounts, shell merchants, cash-out chains —
that per-transaction rules and tabular ML miss because every individual
transaction looks normal in isolation.

```
accounts ---(transactions)---> accounts           account is a fraud-ring member?
   |                                                |
   v                                                v
  tabular features (amount, time)              graph structure (who sends
   |                                            money to whom, in loops)
   v                                                v
  rules / per-tx ML  ~~~~~misses rings~~~~>   GNN message passing detects
                                               coordinated multi-account rings
```

## Why graph-based detection

UPI (India's real-time payment rail, ~10B+ transactions/month) is targeted by
organized fraud: mule accounts that receive, split, and forward stolen funds;
rings of accounts cycling money to obscure the source. Two signatures only
exist at the *network* level:

- **Cyclic flow** — money circles back to the originator through several hops
  (money-laundering "ring" pattern).
- **Unusual connectivity** — mule accounts are densely wired to a small set of
  counterparties; normal accounts are not.

Per-transaction features (amount, time, location) cannot see either signature.
Graph representations + message-passing can.

## Repository layout

```
src/upifraud/
├── generate.py    # wraps SantanderAI/gen-fraud-graph (synthetic data, 100% private)
├── dataset.py     # CSV graphs -> PyTorch Geometric Data + ring-aware train/test split
├── features.py    # node features; drops zero-variance columns
├── models.py      # GCN and GraphSAGE (PyG)
├── train.py       # GNN training: class-imbalance weighting, early stopping
├── baseline.py    # RandomForest / HistGradientBoosting / XGBoost on flattened features
├── evaluate.py    # AUC, average precision, ring-recovery metrics
├── api.py         # FastAPI risk-scoring service
└── cli.py         # `upifraud` command line
```

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

# 1. generate a synthetic UPI transaction graph (10k accounts, 90k transfers, 10 fraud rings)
upifraud generate --scale 0.001 --output data/raw

# 2. train the GNN + baselines + compare on the same held-out rings
upifraud demo --scale 0.001
```

The `demo` command runs the full pipeline: generate → train GNN → train
baselines → comparison table → top risk accounts.

### Individual commands

```bash
upifraud train-gnn     --data data/raw --model sage   # gcn | sage
upifraud train-baseline --data data/raw --model rf    # rf | hgb | xgb
upifraud evaluate --out-dir models
```

### Risk-scoring API

```bash
upifraud serve --out-dir models
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/risk/account/acc_42
curl -X POST http://127.0.0.1:8000/risk/batch -H 'Content-Type: application/json' \
     -d '{"account_ids": ["acc_42", "acc_99"]}'
```

Returns a `risk_score` (0–1), a `risk_band` (low/medium/high), and the account's
risk rank across the graph.

## Data: Santander `gen-fraud-graph`

All experiments run on the open-source synthetic graph generator from
[SantanderAI/gen-fraud-graph](https://github.com/SantanderAI/gen-fraud-graph)
(Apache-2.0). It emits account/transaction CSVs plus planted **fraud rings**
(cyclic money-laundering patterns, 4–7 hops) and their metadata. No real
financial data is used — the whole pipeline is private by construction.

| Flag | Meaning |
|---|---|
| `--scale` | `0.0001` ≈ 1k accounts / 9k tx; `0.001` ≈ 10k / 90k; `1.0` ≈ 10M / 90M |
| `--toy` | tiny built-in generator (same CSV schema) for tests and quick smoke runs |

## Evaluation protocol

Two design choices make the benchmark honest — and are worth defending:

1. **Ring-aware split.** Whole rings are held out for testing, never individual
   accounts. A random node split would leak ring structure into training and
   flatter the results.
2. **Leak-free features.** `gen-fraud-graph` encodes ring transfers with a
   distinctive amount (9999). Feeding amount aggregates into the model makes
   detection *trivial and unrealistic*:

   | Features | GNN AUC | RF AUC |
   |---|---|---|
   | + amount stats (leaky) | 1.00 | 1.00 |
   | structural only (honest) | 0.63 | 0.61 |

   The default pipeline uses structural + account features only, so the
   measured performance reflects *network-signal* detection, not a synthetic
   artifact.

### Results (10k accounts, 90k transfers, 10 rings; 3 rings held out)

| Model | AUC | AP | Mean ring recall | Fraud hit@K |
|---|---|---|---|---|
| GraphSAGE | **0.631** | 0.023 | **0.367** | **0.357** |
| HistGradientBoosting | 0.616 | **0.041** | 0.350 | 0.357 |
| RandomForest | 0.607 | 0.024 | 0.283 | 0.286 |
| GCN | 0.585 | 0.012 | 0.150 | 0.143 |

AP ~0.02–0.04 looks small but the test set is ~1% fraud — the GNN's precision
is ~2–4x the base rate. Numbers vary slightly between runs because the
generator does not expose a seed (it uses its own RNG).

## Known limitations (honest list)

- **Synthetic data.** Results transfer imperfectly to production; `gen-fraud-graph`
  is a benchmark, not a bank. Real deployment needs a labeled graph from a
  payment provider.
- **2-hop receptive field.** GCN/GraphSAGE with 2 layers cannot see rings
  longer than the receptive field; deeper models or ring-structure features
  (e.g., cycle counts) are next steps.
- **Cold-start accounts.** New accounts have no neighborhood — the model must
  fall back to account-level features.
- **Text/embeddings unused.** Transaction descriptions and `embedding` vectors
  are generated but not consumed; natural next feature.

## FAQ / panel answers

**"It's all synthetic — so what?"**
`gen-fraud-graph` was built for exactly this benchmarking use case, with
configurable hardness and planted ground-truth rings. We measure what matters —
recovery of whole rings on held-out rings — and we report the leak honestly.
The same code path ingests any CSV graph with the same schema.

**"You just called a library."**
The pipeline implements the hard parts: CSV→graph construction, ring-aware
splits, leak-free featurization, imbalance-aware training, and ring-recovery
evaluation — the pieces `torch_geometric` does not provide.

**"Why not XGBoost?"**
XGBoost is supported (`--model xgb`) but its OpenMP runtime conflicts with
torch's on macOS (native segfault — see git history), so the default boosted
baseline is sklearn's HistGradientBoosting, which is equivalent in practice.

## Roadmap

- [ ] Edge-level (transaction) classification + temporal features
- [ ] Realistic harder rings (higher `hardness` in the generator, overlapping rings)
- [ ] Cycle/flow features (e.g., 3-cycle counts) beyond 2-hop message passing
- [ ] Cold-start handling and drift monitoring in the API

## License

MIT. Data generator used under Apache-2.0 from Santander AI.
