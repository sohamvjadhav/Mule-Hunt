"""FastAPI risk-scoring service over a trained GNN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .models import build_model
from .train import standardize


class BatchRequest(BaseModel):
    account_ids: list[str]


class RiskResponse(BaseModel):
    account_id: str
    risk_score: float
    risk_band: str
    rank: int


def band(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def create_app(checkpoint_dir: Path, dataset_path: Path, top_n: int = 1000) -> FastAPI:
    checkpoint = next(checkpoint_dir.glob("*.pt"))
    args = json.loads((checkpoint_dir / f"{checkpoint.stem}_args.json").read_text())
    state = torch.load(checkpoint, map_location="cpu")

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
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

    app = FastAPI(title="UPI Fraud Risk API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": args["model"], "nodes": int(data.num_nodes)}

    @app.get("/risk/account/{account_id}", response_model=RiskResponse)
    def risk_account(account_id: str) -> RiskResponse:
        idx = id_to_idx.get(account_id)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"unknown account {account_id}")
        score = float(scores[idx])
        rank = rank_map[idx]
        return RiskResponse(
            account_id=account_id,
            risk_score=round(score, 4),
            risk_band=band(score),
            rank=rank,
        )

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
            score = float(scores[idx])
            out.append(
                RiskResponse(
                    account_id=aid,
                    risk_score=round(score, 4),
                    risk_band=band(score),
                    rank=rank_map[idx],
                )
            )
        return out

    return app
