"""GNN training loop with class-imbalance handling and early stopping."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch_geometric.data import Data

from .evaluate import operating_point
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
    calibrate: bool = True,
    cold_start_threshold: int = 10,
    num_layers: int = 3,
    jk: str = "cat",
    edge_loss_weight: float = 0.5,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, mean, std = standardize(data.x)
    data = data.clone()
    data.x = x.to(device)

    edge_attr = None
    edge_mean = edge_std = None
    edge_dim = None
    if edge_loss_weight > 0 and getattr(data, "edge_attr", None) is not None:
        edge_dim = int(data.edge_attr.size(1))
        edge_attr, edge_mean, edge_std = standardize(data.edge_attr.to(torch.float32))
        edge_attr = edge_attr.to(device)

    model = build_model(
        model_name, data.x.size(1), hidden, num_layers=num_layers, jk=jk,
        edge_attr_dim=edge_dim,
    )
    model.to(device)

    train_y = data.y[data.train_mask]
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edge_index = data.edge_index.to(device)
    y = data.y.float().to(device)

    edge_target = None
    edge_pos_weight = None
    if edge_attr is not None:
        edge_y = data.edge_label.float()
        e_train_np = data.edge_train.numpy()
        e_pos = int(edge_y[e_train_np].sum())
        e_neg = int((1 - edge_y)[e_train_np].sum())
        edge_pos_weight = torch.tensor([e_neg / max(e_pos, 1)]).to(device)
        edge_target = edge_y.to(device)

    best_ap = -1.0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, edge_index)
        loss = loss_fn(logits[data.train_mask], y[data.train_mask])
        if edge_attr is not None and edge_target is not None:
            e_logits = model.edge_forward(data.x, edge_index, edge_attr)
            edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                e_logits[data.edge_train], edge_target[data.edge_train],
                pos_weight=edge_pos_weight,
            )
            loss = loss + edge_loss_weight * edge_loss
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

    cold_start_meta: dict | None = None
    if cold_start_threshold > 0 and getattr(data, "x_raw", None) is not None:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier as _HGB

        idx = getattr(data, "cold_start_indices", None)
        if idx is None:
            idx = list(range(min(3, data.x_raw.size(1))))
        cs_x = data.x_raw[:, idx].numpy()
        cs_y = data.y.numpy()
        train_mask_np = data.train_mask.numpy()
        val_mask_np = data.val_mask.numpy()
        fit_mask = train_mask_np | val_mask_np
        cs_model = _HGB(max_iter=200, random_state=seed, class_weight="balanced")
        cs_model.fit(cs_x[fit_mask], cs_y[fit_mask])
        cs_path = out_dir / f"{model_name}_coldstart.joblib"
        joblib.dump(cs_model, cs_path)
        train_probs = cs_model.predict_proba(cs_x[fit_mask])[:, 1]
        cs_auc = float(roc_auc_score(cs_y[fit_mask], train_probs))
        cold_start_meta = {
            "path": str(cs_path),
            "indices": list(idx),
            "threshold": cold_start_threshold,
            "train_auc": cs_auc,
            "feature_names": list(getattr(data, "cold_start_feature_names", [])),
        }

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(data.x, edge_index)[data.val_mask]).cpu().numpy()
        all_probs = torch.sigmoid(model(data.x, edge_index)).cpu().numpy()
    val_y_np = data.y[data.val_mask].cpu().numpy()
    brier_before = float(brier_score_loss(val_y_np, val_probs))
    cal_path = None
    if calibrate:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs, val_y_np)
        calibrated_val = iso.predict(val_probs)
        brier_after = float(brier_score_loss(val_y_np, calibrated_val))
        cal_path = out_dir / f"{model_name}_calib.pkl"
        with cal_path.open("wb") as f:
            pickle.dump(iso, f)
        all_scores = iso.predict(all_probs).astype(float)
    else:
        brier_after = brier_before
        all_scores = all_probs.astype(float)

    args = {
        "model": model_name,
        "hidden": hidden,
        "in_dim": int(data.x.size(1)),
        "mean": mean.cpu().tolist(),
        "std": std.cpu().tolist(),
        "num_layers": num_layers,
        "jk": jk,
        "edge_attr_dim": edge_dim,
        "edge_loss_weight": edge_loss_weight,
        "edge_attr_mean": edge_mean.cpu().tolist() if edge_mean is not None else None,
        "edge_attr_std": edge_std.cpu().tolist() if edge_std is not None else None,
        "best_val_ap": best_ap,
        "epochs_run": len(history),
        "pos_weight": neg / max(pos, 1),
        "calibrator": "isotonic" if calibrate else None,
        "brier_val_before": brier_before,
        "brier_val_after": brier_after,
        "cold_start": cold_start_meta,
    }
    (out_dir / f"{model_name}_args.json").write_text(json.dumps(args, indent=2))

    edge_scores = None
    edge_eval = None
    if edge_attr is not None:
        with torch.no_grad():
            edge_scores = torch.sigmoid(model.edge_forward(data.x, edge_index, edge_attr)).cpu().numpy()
        from .evaluate import evaluate_edges

        edge_eval = evaluate_edges(data, edge_scores, split="test")

    return {
        "history": history,
        "args": args,
        "scores": all_scores,
        "model": model,
        "calibrator_path": str(cal_path) if cal_path else None,
        "edge_scores": edge_scores,
        "edge_eval": edge_eval,
    }


def evaluate_gnn(data: Data, scores: np.ndarray, split: str = "test") -> dict:
    mask = getattr(data, f"{split}_mask").numpy()
    y = data.y.numpy()[mask]
    s = scores[mask]
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "n_fraud": int(y.sum()),
        "n_total": len(y),
        "operating_point": operating_point(data, scores, split),
    }
