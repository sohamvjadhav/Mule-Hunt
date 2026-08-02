"""Ring-recovery evaluation: the metric that matters for fraud-ring detection."""

from __future__ import annotations

import numpy as np
from torch_geometric.data import Data


def ring_recovery(
    data: Data,
    scores: np.ndarray,
    top_k: int | None = None,
    split: str = "test",
) -> dict:
    """For each planted ring, measure how many members appear among the top-K
    predicted-fraud accounts. High ring recall means the model catches whole
    rings, not just scattered accounts.
    """
    mask = getattr(data, f"{split}_mask").numpy()
    n_nodes = data.num_nodes
    order = np.argsort(-scores)
    ranked = order[order < n_nodes]

    if top_k is None:
        top_k = int(mask.sum())

    ring_members: dict[int, list[int]] = {}
    for i in range(n_nodes):
        r = int(data.ring_id[i])
        if r >= 0 and mask[i]:
            ring_members.setdefault(r, []).append(i)

    per_ring = []
    for r, members in ring_members.items():
        in_top = sum(1 for m in members if np.where(ranked == m)[0][0] < top_k)
        per_ring.append(
            {
                "ring_id": r,
                "size": len(members),
                "recovered": in_top,
                "recall": in_top / len(members),
            }
        )

    test_fraud = np.where(mask & (data.y.numpy() == 1))[0]
    fraud_in_top = sum(1 for m in test_fraud if np.where(ranked == m)[0][0] < top_k)

    return {
        "top_k": top_k,
        "n_rings": len(per_ring),
        "mean_ring_recall": float(np.mean([p["recall"] for p in per_ring])) if per_ring else 0.0,
        "fraud_hit_rate_at_k": fraud_in_top / max(len(test_fraud), 1),
        "per_ring": per_ring,
    }


def top_fraud_accounts(data: Data, scores: np.ndarray, k: int = 20) -> list[dict]:
    order = np.argsort(-scores)[:k]
    out = []
    for i in order:
        out.append(
            {
                "rank": len(out) + 1,
                "account_id": data.node_ids[i],
                "risk_score": round(float(scores[i]), 4),
                "ring_id": int(data.ring_id[i]),
                "true_label": int(data.y[i]),
            }
        )
    return out
