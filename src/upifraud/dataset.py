"""CSV transaction graphs -> PyTorch Geometric Data with ring-aware splits."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .features import build_node_features, drop_constant_columns


def load_graph(
    data_dir: Path,
    with_amount_stats: bool = False,
    split: str = "rings",
    test_rings: int = 3,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Data:
    """Load gen-fraud-graph CSVs into a single PyG Data object.

    Node label: 1 if the account belongs to a planted fraud ring.
    Edge labels are stored for completeness (future edge-level tasks).
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

    x = drop_constant_columns(
        build_node_features(accounts, edge_index, edge_amounts, n_nodes, with_amount_stats)
    )

    ring_id = torch.full((n_nodes,), -1, dtype=torch.long)
    for i, ring in enumerate(rings):
        for aid in ring:
            if aid in id_to_idx:
                ring_id[id_to_idx[aid]] = i

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        edge_label=edge_labels,
        ring_id=ring_id,
        num_rings=len(rings),
    )
    data.node_ids = list(accounts["account_id"])

    if split == "rings":
        _ring_aware_split(data, rng, test_rings, val_frac)
    else:
        _random_split(data, rng, val_frac)
    return data


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
