# Changelog

All notable changes to Mule-Hunt are documented here, grouped by release.

## 0.6.0

### Added

- Case files: `upifraud case <account_id>` renders a complete, shareable
  Markdown investigation document (subject, ring context, top suspicious
  transactions, counterfactual probe, recommendation); served at
  `GET /api/case/{account_id}` and downloadable from the dashboard.
- Counterfactual sensitivity analysis: drop an account's top-k highest-risk
  edges and re-score the frozen model (`GET /api/counterfactual/{account_id}`,
  also reachable through the assistant: "what if acc_X ...?").
- Honest framing: counterfactuals are documented as fixed-model, feature-
  constant probes (not retrained-model experiments).

## 0.5.0

### Added

- `POST /api/ask`: the investigation assistant is now a service endpoint —
  natural-language questions about accounts, rings, the top risk list, or the
  network, answered with grounded facts.
- Dashboard investigation panel: ask the assistant directly from the browser.
- Automated PR review: `.github/workflows/pr-review.yml` runs the Codex CLI
  over every PR diff and posts a maintainer-style review comment (activates
  when a `CODEX_API_KEY` / `OPENAI_API_KEY` secret exists; otherwise skips).

## 0.4.0

### Added

- AI Fraud Investigation Assistant: `upifraud query` answers
  natural-language questions with grounded, template-generated reports
  (no LLM dependency, no hallucination).
- MCP investigation server: `upifraud mcp` exposes the risk graph to coding
  agents over stdio (`account_risk`, `explain_account`, `investigate`,
  `ring_details`, `top_risky`, `network_summary`).
- Agent and maintenance scaffolding: `AGENTS.md`, `Makefile`, `SECURITY.md`,
  issue/PR templates, README badges and a Development section.
- `__version__` now reads from installed package metadata (was stale).

## 0.3.1

### Added

- Adversarial-robustness experiment: `upifraud attack` perturbs held-out
  ring edges (inject camouflage / drop evidence) and re-scores a fixed
  model per budget.
- `count_same_ring_edges` shared by the temporal and adversarial reports.

### Changed

- Optional CSV columns: only `account_id` / `tx_id` / `src_id` / `dst_id` /
  `amount` are required; all other attributes degrade gracefully.
- Temporal (dynamic-graph) modeling: bursty ring windows in the generator,
  cumulative snapshots, per-slice forward evaluation, and staleness scoring
  (`upifraud temporal`).
- Honest full-scale results documented: temporal features are flat on the
  synthetic data; evidence-removal attacks hurt far more than camouflage.

## 0.3.0

Initial PyPI release as `mule-hunt`.

### Added

- GNN detection with GCN, GraphSAGE, and GATv2, configurable depth with
  Jumping Knowledge, and a joint transaction-level (edge) head.
- Baselines: random forest, histogram gradient boosting, XGBoost.
- Ring-aware evaluation: held-out rings, ring recovery, F1 operating point.
- Isotonic calibration, cold-start fallback for low-activity accounts, and
  PSI drift monitoring.
- FastAPI risk service with `/risk/*` and `/api/*` endpoints plus a static
  dashboard with transaction-level ring views.
- Model-grounded explanations (`/api/explain`) via GNNExplainer drivers.
- Hardness benchmark matrix (`upifraud benchmark`).
- PyPI publishing via trusted publishing on `v*` tags.
