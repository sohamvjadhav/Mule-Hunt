"""CSV transaction graphs -> PyTorch Geometric Data with ring-aware splits."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .features import build_edge_features, build_node_features, drop_constant_columns


def load_graph(
    data_dir: Path,
    with_amount_stats: bool = False,
    with_cycle_counts: bool = False,
    with_temporal: bool = False,
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
    edge_ts = None
    if "timestamp" in tx:
        ts_epoch = pd.to_datetime(tx["timestamp"]).astype("int64") // 10**9
        edge_ts_map = dict(zip(tx["tx_id"], ts_epoch.to_numpy()))
        edge_ts = torch.tensor([edge_ts_map[tid] for tid in edges["tx_id"]], dtype=torch.float32)
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

    x_full, full_names = build_node_features(
        accounts,
        edge_index,
        edge_amounts,
        n_nodes,
        with_amount_stats,
        with_cycle_counts,
        with_temporal=with_temporal,
        edge_ts=edge_ts,
    )
    x, keep = drop_constant_columns(x_full)
    feature_names = [full_names[i] for i in keep]

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
    data.edge_ts = edge_ts
    data.x_raw = x_full
    data.accounts = accounts
    data.feature_flags = {
        "with_amount_stats": with_amount_stats,
        "with_cycle_counts": with_cycle_counts,
        "with_temporal": with_temporal,
    }
    data.cold_start_feature_names = [
        "balance", "risk_score", "age_days"
    ]
    data.cold_start_indices = [
        full_names.index("balance"),
        full_names.index("risk_score"),
        full_names.index("age_days"),
    ] if all(n in full_names for n in data.cold_start_feature_names) else [0, 1, 2] if len(full_names) >= 3 else None

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


def count_same_ring_edges(data: Data, mask: torch.Tensor | None = None) -> int:
    """Number of edges whose endpoints belong to the same planted ring.

    ``mask`` (e.g. a node split mask) restricts the count to edges touching
    the masked nodes. This is the "revealed ring structure" measure used by
    the temporal and adversarial experiments.
    """
    src, dst = data.edge_index
    same_ring = (data.ring_id[src] >= 0) & (data.ring_id[src] == data.ring_id[dst])
    if mask is not None:
        same_ring = same_ring & mask[src]
    return int(same_ring.sum())


def build_snapshots(data: Data, k: int) -> list[Data]:
    """Slice the edge timeline into ``k`` cumulative snapshots.

    Snapshot ``s`` contains every edge whose timestamp falls at or before the
    ``(s+1)/k`` quantile of the edge timeline. Node features and edge-level
    data are recomputed from the subset that has "happened" by that time, so
    a snapshot is exactly the graph known at that point in time. If the graph
    has no timestamps (or ``k <= 1``) the original graph is returned as a
    single snapshot.
    """
    ts = getattr(data, "edge_ts", None)
    if ts is None or k <= 1 or int(ts.numel()) == 0:
        return [data]

    t = ts.numpy()
    finite = np.isfinite(t)
    sorted_t = np.sort(t[finite])
    cuts = np.quantile(sorted_t, np.arange(1, k + 1) / k)

    snaps: list[Data] = []
    for s in range(k):
        keep = np.flatnonzero(finite & (t <= cuts[s]))
        snaps.append(_rebuild_snapshot(data, keep, s, float(cuts[s])))
    return snaps


def _rebuild_snapshot(data: Data, kept_indices: np.ndarray, slice: int, boundary: float) -> Data:
    ei = data.edge_index[:, kept_indices]
    ea = data.edge_amounts[kept_indices]
    et = data.edge_ts[kept_indices]

    flags = dict(getattr(data, "feature_flags", {}))
    x_full, snap_names = build_node_features(
        data.accounts,
        ei,
        ea,
        data.num_nodes,
        flags.get("with_amount_stats", False),
        flags.get("with_cycle_counts", False),
        flags.get("with_temporal", False),
        et,
    )
    x, keep = drop_constant_columns(x_full)
    feature_names = [snap_names[i] for i in keep]

    snap = Data(
        x=x,
        edge_index=ei,
        edge_attr=data.edge_attr[kept_indices],
        y=data.y,
        edge_label=data.edge_label[kept_indices],
        ring_id=data.ring_id,
        num_rings=data.num_rings,
    )
    snap.node_ids = data.node_ids
    snap.feature_names = feature_names
    snap.edge_feature_names = data.edge_feature_names
    snap.edge_amounts = ea
    snap.edge_ts = et
    snap.x_raw = x_full
    snap.accounts = data.accounts
    snap.feature_flags = flags
    snap.cold_start_feature_names = getattr(data, "cold_start_feature_names", [])
    snap.cold_start_indices = getattr(data, "cold_start_indices", None)
    snap.slice = slice
    snap.boundary_ts = boundary

    snap.train_mask = data.train_mask
    snap.val_mask = data.val_mask
    snap.test_mask = data.test_mask
    _edge_masks(snap)
    return snap
