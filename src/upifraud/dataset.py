"""CSV transaction graphs -> PyTorch Geometric Data with ring-aware splits."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .features import build_edge_features, build_node_features, drop_constant_columns


def load_graph(
    data_dir: Path,
    with_amount_stats: bool = False,
    with_cycle_counts: bool = False,
    split: str = "rings",
    test_rings: int = 3,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Data:
    """Load gen-fraud-graph CSVs into a single PyG Data object.

    Node label: 1 if the account belongs to a planted fraud ring.
    Edge label: 1 if the transaction was flagged in the fraud CSV and both
    endpoints belong to the same planted ring.
    """
    rng = random.Random(seed)
    account_files = sorted((data_dir / "accounts").glob("accounts_*.csv"))
    tx_files = sorted((data_dir / "transactions").glob("transactions_*.csv"))
    if not account_files or not tx_files:
        raise FileNotFoundError(f"no graph CSVs found under {data_dir}")

    accounts = pd.concat([pd.read_csv(f) for f in account_files], ignore_index=True)
    tx = pd.concat([pd.read_csv(f) for f in tx_files], ignore_index=True)

    fraud_tx_ids: set[str] = set()
    fraud_tx_file = data_dir / "fraud" / "transactions_fraud.csv"
    if fraud_tx_file.exists():
        fraud_tx = pd.read_csv(fraud_tx_file)
        fraud_tx_ids = set(fraud_tx["tx_id"])
        tx = pd.concat([tx, fraud_tx], ignore_index=True)

    ring_rows = []
    fraud_file = data_dir / "fraud" / "fraud_cases.csv"
    if fraud_file.exists():
        ring_rows = pd.read_csv(fraud_file).to_dict("records")

    rings = [r["involved_accounts"].split("|") for r in ring_rows]

    id_to_idx = {aid: i for i, aid in enumerate(accounts["account_id"])}
    n_nodes = len(accounts)

    edges = tx[["tx_id", "src_id", "dst_id"]].drop_duplicates()
    edges = edges[edges["src_id"].isin(id_to_idx) & edges["dst_id"].isin(id_to_idx)]
    edge_index = torch.stack(
        [
            torch.tensor([id_to_idx[s] for s in edges["src_id"]]),
            torch.tensor([id_to_idx[d] for d in edges["dst_id"]]),
        ]
    )
    tx_amount_map = dict(zip(tx["tx_id"], tx["amount"]))
    edge_amounts = torch.tensor([tx_amount_map[tid] for tid in edges["tx_id"]])
    edge_labels = torch.tensor(
        [1 if tid in fraud_tx_ids else 0 for tid in edges["tx_id"]]
    )

    fraud_nodes = set()
    for ring in rings:
        fraud_nodes.update(ring)
    y = torch.zeros(n_nodes, dtype=torch.long)
    for aid in fraud_nodes:
        if aid in id_to_idx:
            y[id_to_idx[aid]] = 1

    base_names = ["balance", "risk_score", "age_days", "in_deg", "out_deg", "unique_in", "unique_out"]
    if with_amount_stats:
        base_names += ["amt_in_mean", "amt_out_mean", "amt_in_sum", "amt_out_sum"]
    if with_cycle_counts:
        base_names += ["triangle_count", "clustering_coef"]

    x_full = build_node_features(
        accounts, edge_index, edge_amounts, n_nodes, with_amount_stats, with_cycle_counts
    )
    x, keep = drop_constant_columns(x_full)
    feature_names = [base_names[i] for i in keep]

    ring_id = torch.full((n_nodes,), -1, dtype=torch.long)
    for i, ring in enumerate(rings):
        for aid in ring:
            if aid in id_to_idx:
                ring_id[id_to_idx[aid]] = i

    src_r = ring_id[edge_index[0]]
    dst_r = ring_id[edge_index[1]]
    same_ring = (src_r >= 0) & (src_r == dst_r)
    edge_labels = edge_labels & same_ring

    edge_attr, edge_feature_names = build_edge_features(tx, accounts, list(edges["tx_id"]))

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_label=edge_labels,
        ring_id=ring_id,
        num_rings=len(rings),
    )
    data.node_ids = list(accounts["account_id"])
    data.feature_names = feature_names
    data.edge_feature_names = edge_feature_names
    data.edge_amounts = edge_amounts
    data.x_raw = x_full
    data.cold_start_feature_names = [
        "balance", "risk_score", "age_days"
    ]
    data.cold_start_indices = [
        base_names.index("balance"),
        base_names.index("risk_score"),
        base_names.index("age_days"),
    ] if all(n in base_names for n in data.cold_start_feature_names) else [0, 1, 2] if len(base_names) >= 3 else None

    if split == "rings":
        _ring_aware_split(data, rng, test_rings, val_frac)
    else:
        _random_split(data, rng, val_frac)
    _edge_masks(data)
    return data


def _edge_masks(data: Data) -> None:
    """Edge-level split masks derived from the endpoints' node masks.

    An edge belongs to the training split only when both endpoints do, so
    held-out rings never contribute edges to training. Edges touching nodes
    across splits are excluded from loss and evaluation.
    """
    src, dst = data.edge_index
    data.edge_train = data.train_mask[src] & data.train_mask[dst]
    data.edge_val = data.val_mask[src] & data.val_mask[dst]
    data.edge_test = data.test_mask[src] & data.test_mask[dst]


def _ring_aware_split(data: Data, rng: random.Random, test_rings: int, val_frac: float) -> None:
    n_rings = int(data.num_rings)
    if n_rings == 0:
        _random_split(data, rng, val_frac)
        return
    ring_order = list(range(n_rings))
    rng.shuffle(ring_order)
    test = set(ring_order[: min(test_rings, n_rings)])
    remaining = [r for r in ring_order if r not in test]
    n_val = max(1, int(len(remaining) * val_frac))
    val = set(remaining[:n_val])

    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        r = int(data.ring_id[i])
        if r in test:
            test_mask[i] = True
        elif r in val:
            val_mask[i] = True
        elif r >= 0:
            train_mask[i] = True

    normal = [i for i in range(n) if int(data.ring_id[i]) == -1]
    rng.shuffle(normal)
    n_val_n = max(1, int(len(normal) * val_frac))
    n_test_n = max(1, int(len(normal) * 0.15))
    for i in normal[n_val_n + n_test_n :]:
        train_mask[i] = True
    for i in normal[:n_val_n]:
        val_mask[i] = True
    for i in normal[n_val_n : n_val_n + n_test_n]:
        test_mask[i] = True
    _finalize_masks(data, train_mask, val_mask, test_mask)


def _random_split(data: Data, rng: random.Random, val_frac: float) -> None:
    n = data.num_nodes
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * 0.15))
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for i in idx[n_val + n_test :]:
        train_mask[i] = True
    for i in idx[:n_val]:
        val_mask[i] = True
    for i in idx[n_val : n_val + n_test]:
        test_mask[i] = True
    _finalize_masks(data, train_mask, val_mask, test_mask)


def _finalize_masks(data: Data, train: torch.Tensor, val: torch.Tensor, test: torch.Tensor) -> None:
    data.train_mask = train
    data.val_mask = val
    data.test_mask = test
