"""FastAPI risk-scoring service over a trained GNN, with dashboard endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch_geometric.data import Data

from .models import build_model
from .train import standardize

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
OPENAI_MODEL = "gpt-5.6-luna"


class BatchRequest(BaseModel):
    account_ids: list[str]


class RiskResponse(BaseModel):
    account_id: str
    risk_score: float
    risk_band: str
    rank: int


class AccountDetail(RiskResponse):
    degree: int
    true_label: int
    ring_id: int
    neighbors: list[dict]


def band(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def create_app(
    checkpoint_dir: Path,
    dataset_path: Path,
    top_n: int = 1000,
    frontend_dir: Path | None = None,
) -> FastAPI:
    if frontend_dir is None:
        frontend_dir = FRONTEND_DIR
    candidates = sorted(p for p in checkpoint_dir.glob("*.pt") if p.stem != "graph")
    if not candidates:
        raise FileNotFoundError(f"no model checkpoint (*.pt) in {checkpoint_dir}")
    checkpoint = candidates[0]
    args = json.loads((checkpoint_dir / f"{checkpoint.stem}_args.json").read_text())
    state = torch.load(checkpoint, map_location="cpu")

    data: Data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    mean = torch.tensor(args["mean"])
    std = torch.tensor(args["std"])
    x, _, _ = standardize(data.x, mean, std)

    model = build_model(args["model"], int(args["in_dim"]), int(args["hidden"]))
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        scores = model(x, data.edge_index).sigmoid().numpy()
    order = np.argsort(-scores)
    rank_map = {int(i): r for r, i in enumerate(order)}

    id_to_idx = {aid: i for i, aid in enumerate(data.node_ids)}
    idx_to_id = {i: aid for aid, i in id_to_idx.items()}
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    n_edges = len(src)

    degree = np.bincount(np.concatenate([src, dst]), minlength=data.num_nodes)

    app = FastAPI(title="Mule-Hunt Risk API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": args["model"], "nodes": int(data.num_nodes)}

    @app.get("/risk/account/{account_id}", response_model=RiskResponse)
    def risk_account(account_id: str) -> RiskResponse:
        idx = id_to_idx.get(account_id)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
        return _risk_response(idx)

    @app.post("/risk/batch", response_model=list[RiskResponse])
    def risk_batch(req: BatchRequest) -> list[RiskResponse]:
        out = []
        for aid in req.account_ids:
            idx = id_to_idx.get(aid)
            if idx is None:
                out.append(
                    RiskResponse(account_id=aid, risk_score=0.0, risk_band="unknown", rank=-1)
                )
                continue
            out.append(_risk_response(idx))
        return out

    def _risk_response(idx: int) -> RiskResponse:
        return RiskResponse(
            account_id=idx_to_id[idx],
            risk_score=round(float(scores[idx]), 4),
            risk_band=band(float(scores[idx])),
            rank=int(rank_map[idx]),
        )

    @app.get("/api/summary")
    def summary() -> dict:
        y = data.y.numpy()
        ring_counts = {}
        for r in np.unique(data.ring_id.numpy()):
            ring_counts[int(r)] = int((data.ring_id.numpy() == r).sum())
        return {
            "n_accounts": int(data.num_nodes),
            "n_transactions": n_edges,
            "n_fraud": int(y.sum()),
            "fraud_rate": round(float(y.mean()), 5),
            "n_rings": int(data.num_rings),
            "model": args["model"],
            "explainer": "openai" if os.environ.get("OPENAI_API_KEY") else "local",
            "ring_sizes": {k: v for k, v in sorted(ring_counts.items()) if k >= 0},
        }

    @app.get("/api/top")
    def top(k: int = 50) -> list[dict]:
        k = min(max(k, 1), int(data.num_nodes))
        out = []
        for i in order[:k]:
            out.append(
                {
                    "rank": int(rank_map[i]) + 1,
                    "account_id": idx_to_id[i],
                    "risk_score": round(float(scores[i]), 4),
                    "risk_band": band(float(scores[i])),
                    "ring_id": int(data.ring_id[i]),
                    "true_label": int(data.y[i]),
                    "degree": int(degree[i]),
                }
            )
        return out

    @app.get("/api/account/{account_id}", response_model=AccountDetail)
    def account_detail(account_id: str) -> AccountDetail:
        idx = id_to_idx.get(account_id)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
        nbr = []
        for i in range(n_edges):
            if src[i] == idx:
                nbr.append(int(dst[i]))
            elif dst[i] == idx:
                nbr.append(int(src[i]))
        nbr_ids = []
        seen = set()
        for j in nbr:
            if j in seen:
                continue
            seen.add(j)
            nbr_ids.append(
                {
                    "account_id": idx_to_id[j],
                    "risk_score": round(float(scores[j]), 4),
                    "risk_band": band(float(scores[j])),
                    "rank": int(rank_map[j]) + 1,
                }
            )
        nbr_ids.sort(key=lambda r: -r["risk_score"])
        return AccountDetail(
            account_id=account_id,
            risk_score=round(float(scores[idx]), 4),
            risk_band=band(float(scores[idx])),
            rank=int(rank_map[idx]) + 1,
            degree=int(degree[idx]),
            true_label=int(data.y[idx]),
            ring_id=int(data.ring_id[idx]),
            neighbors=nbr_ids[:25],
        )

    @app.get("/api/ring/{ring_id}")
    def ring(ring_id: int) -> dict:
        ring_id = int(ring_id)
        if ring_id < 0 or ring_id >= int(data.num_rings):
            raise HTTPException(status_code=404, detail=f"unknown ring {ring_id}")
        members = [i for i in range(data.num_nodes) if int(data.ring_id[i]) == ring_id]
        member_set = set(members)
        edges = []
        ext = np.zeros(len(members), dtype=int)
        for i in range(n_edges):
            a, b = int(src[i]), int(dst[i])
            if a in member_set and b in member_set:
                edges.append([members.index(a), members.index(b)])
            elif a in member_set:
                ext[members.index(a)] += 1
            elif b in member_set:
                ext[members.index(b)] += 1
        return {
            "ring_id": ring_id,
            "size": len(members),
            "nodes": [
                {
                    "index": j,
                    "account_id": idx_to_id[m],
                    "risk_score": round(float(scores[m]), 4),
                    "risk_band": band(float(scores[m])),
                    "rank": int(rank_map[m]) + 1,
                    "true_label": int(data.y[m]),
                    "external_connections": int(ext[j]),
                }
                for j, m in enumerate(members)
            ],
            "edges": edges,
        }

    @app.get("/api/distribution")
    def distribution(bins: int = 20) -> dict:
        hist, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
        return {
            "bins": [round(float(e), 3) for e in edges.tolist()],
            "counts": [int(c) for c in hist],
        }

    def _context(idx: int) -> dict:
        nbr = []
        for i in range(n_edges):
            if src[i] == idx:
                nbr.append(int(dst[i]))
            elif dst[i] == idx:
                nbr.append(int(src[i]))
        seen = set()
        top = []
        for j in nbr:
            if j in seen:
                continue
            seen.add(j)
            top.append((float(scores[j]), idx_to_id[j]))
        top.sort(reverse=True)
        return {
            "account_id": idx_to_id[idx],
            "risk_score": round(float(scores[idx]), 4),
            "risk_band": band(float(scores[idx])),
            "risk_rank": int(rank_map[idx]) + 1,
            "degree": int(degree[idx]),
            "ring_member": int(data.ring_id[idx]) >= 0,
            "ring_id": int(data.ring_id[idx]),
            "true_label": int(data.y[idx]),
            "top_neighbors": [
                {"account_id": aid, "risk_score": round(s, 4)} for s, aid in top[:5]
            ],
        }

    def _local_explanation(ctx: dict) -> str:
        parts = []
        if ctx["ring_member"]:
            parts.append(
                f"belongs to a known fraud ring (ring {ctx['ring_id']}); ring members "
                "send money in cycles, a money-laundering pattern"
            )
        elif ctx["risk_band"] != "low":
            parts.append(
                "is not in a known ring, so its score comes from account traits and "
                "neighborhood patterns rather than ring membership"
            )
        if ctx["top_neighbors"]:
            n = ctx["top_neighbors"][0]
            parts.append(
                f"its highest-risk neighbor {n['account_id']} scores {n['risk_score']:.3f}"
            )
        if ctx["degree"] >= 10:
            parts.append(f"has high connectivity ({ctx['degree']} transactions)")
        detail = "; ".join(parts) if parts else "shows no unusual structural signals"
        return (
            f"{ctx['account_id']} shows {ctx['risk_band']} risk "
            f"(score {ctx['risk_score']:.3f}, rank {ctx['risk_rank']}). "
            f"It {detail}."
        )

    def _openai_explanation(ctx: dict) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return ""
        prompt = (
            "You are an AML risk analyst writing a short narrative for a "
            "suspicious-transaction report. Explain the risk of this payment "
            f"account in 2-3 plain sentences, non-technical: {json.dumps(ctx)}"
        )
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "Concise AML risk narratives."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 150,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    @app.get("/api/explain/{account_id}")
    def explain(account_id: str) -> dict:
        idx = id_to_idx.get(account_id)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
        ctx = _context(idx)
        try:
            text = _openai_explanation(ctx)
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            text = ""
        if text:
            return {"account_id": account_id, "source": "openai", "explanation": text}
        return {"account_id": account_id, "source": "local", "explanation": _local_explanation(ctx)}

    if frontend_dir is not None and frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="dashboard")
    return app
