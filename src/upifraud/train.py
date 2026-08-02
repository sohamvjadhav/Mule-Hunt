"""GNN training loop with class-imbalance handling and early stopping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from .models import build_model

DEVICE = "cpu"


def standardize(x: torch.Tensor, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
    if mean is None:
        mean = x.mean(dim=0)
    if std is None:
        std = x.std(dim=0).clamp_min(1e-8)
    return (x - mean) / std, mean, std


def train_gnn(
    data: Data,
    model_name: str = "sage",
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
    out_dir: Path = Path("models"),
    device: str = DEVICE,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, mean, std = standardize(data.x)
    data = data.clone()
    data.x = x.to(device)

    model = build_model(model_name, data.x.size(1), hidden)
    model.to(device)

    train_y = data.y[data.train_mask]
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)

    best_ap = -1.0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, edge_index)
        loss = loss_fn(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[data.val_mask].cpu().numpy()
            val_y = y[data.val_mask].cpu().numpy()
        val_ap = average_precision_score(val_y, val_logits)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_ap": float(val_ap)})

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    torch.save(model.state_dict(), out_dir / f"{model_name}.pt")

    args = {
        "model": model_name,
        "hidden": hidden,
        "in_dim": int(data.x.size(1)),
        "mean": mean.cpu().tolist(),
        "std": std.cpu().tolist(),
        "best_val_ap": best_ap,
        "epochs_run": len(history),
        "pos_weight": neg / max(pos, 1),
    }
    (out_dir / f"{model_name}_args.json").write_text(json.dumps(args, indent=2))

    model.eval()
    with torch.no_grad():
        scores = model(data.x, edge_index).cpu().numpy()
    return {"history": history, "args": args, "scores": scores, "model": model}


def evaluate_gnn(data: Data, scores: np.ndarray, split: str = "test") -> dict:
    mask = getattr(data, f"{split}_mask").numpy()
    y = data.y.numpy()[mask]
    s = scores[mask]
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "n_fraud": int(y.sum()),
        "n_total": len(y),
    }
