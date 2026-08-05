"""Node feature construction from account attributes and local graph structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def build_node_features(
    accounts_df: pd.DataFrame,
    edge_index: torch.Tensor,
    amounts: torch.Tensor,
    n_nodes: int,
    with_amount_stats: bool = False,
    with_cycle_counts: bool = False,
) -> torch.Tensor:
    """Build per-node features. Amount aggregates are optional: the fraud
    generator encodes ring edges with a distinctive amount, so including
    amount stats trivially reveals rings. The default (structural) features
    isolate the network-signal contribution to detection. Cycle counts are a
    cheap structural signal of ring structure (a 6-cycle ring shows up as
    many 3-cycles once chorded, and clustering elevates for dense mule hubs).
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    amt = amounts.numpy()

    acc = accounts_df.set_index("account_id")
    bal = np.log1p(acc["balance"].values)
    risk = acc["risk_score"].values
    creation = pd.to_datetime(acc["creation_date"])
    age_days = np.log1p((pd.Timestamp("2025-12-31") - creation).dt.days.values.astype(float))

    in_deg = np.bincount(dst, minlength=n_nodes)
    out_deg = np.bincount(src, minlength=n_nodes)

    features = [bal, risk, age_days, in_deg, out_deg]

    edges_df = pd.DataFrame({"src": src, "dst": dst})
    unique_out = edges_df.groupby("dst")["src"].nunique().reindex(range(n_nodes), fill_value=0)
    unique_in = edges_df.groupby("src")["dst"].nunique().reindex(range(n_nodes), fill_value=0)
    features += [unique_in.to_numpy(), unique_out.to_numpy()]

    if with_cycle_counts:
        tri, cc = _triangle_counts(src, dst, n_nodes)
        features += [np.log1p(tri), cc]

    if with_amount_stats:
        edges_df["amount"] = amt
        agg = edges_df.groupby("src")["amount"].agg(["sum", "mean", "count"])
        out_sum = agg["sum"].reindex(range(n_nodes), fill_value=0.0)
        out_mean = agg["mean"].reindex(range(n_nodes), fill_value=0.0)
        agg_in = edges_df.groupby("dst")["amount"].agg(["sum", "mean", "count"])
        in_sum = agg_in["sum"].reindex(range(n_nodes), fill_value=0.0)
        in_mean = agg_in["mean"].reindex(range(n_nodes), fill_value=0.0)
        features += [
            np.log1p(in_mean.to_numpy()),
            np.log1p(out_mean.to_numpy()),
            np.log1p(in_sum.to_numpy()),
            np.log1p(out_sum.to_numpy()),
        ]

    return torch.tensor(np.stack(features, axis=1), dtype=torch.float32)


def _triangle_counts(
    src: np.ndarray, dst: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node undirected 3-cycle counts and local clustering coefficient.

    tri[v] = (A^3)[v, v] / 2 where A is the undirected adjacency matrix;
    cc[v]  = 2 * tri[v] / (deg[v] * (deg[v] - 1)), 0 for deg < 2.
    Uses sparse matmul, so a 10k-node graph is computed in milliseconds.
    """
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n_nodes, n_nodes))
    A = (A + A.T) > 0
    A2 = A @ A
    tri = np.asarray(A.multiply(A2).sum(axis=1)).ravel() // 2
    deg = np.asarray(A.sum(axis=1)).ravel()
    denom = deg * (deg - 1)
    cc = np.divide(2 * tri, denom, out=np.zeros_like(tri, dtype=float), where=denom > 0)
    return tri, cc


def drop_constant_columns(x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Drop zero-variance features (e.g. constant account age in synthetic
    data). Constant columns become noise amplifiers after standardization.
    Returns the reduced tensor and the indices of the kept columns.
    """
    std = x.std(dim=0)
    keep = (std > 1e-8).nonzero().flatten().tolist()
    return x[:, keep], keep


def build_edge_features(
    tx_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    tx_ids: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Per-transaction features, aligned with the given tx_id order.

    Features: log amount, hour-of-day (sin/cos), and log hours between the
    transaction and the sender account's creation date. The amount is the
    single most predictive signal on synthetic data (the generator encodes
    ring edges with a distinctive amount), so structural signal must come
    from the node-embedding side of the edge head.
    """
    sub = tx_df.set_index("tx_id").loc[tx_ids]
    amt = np.log1p(sub["amount"].to_numpy(dtype=float))

    ts = pd.to_datetime(sub["timestamp"])
    hour = ts.dt.hour.to_numpy(dtype=float)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)

    created = pd.to_datetime(
        accounts_df.set_index("account_id").loc[sub["src_id"].to_numpy(), "creation_date"]
    ).to_numpy()
    since_hours = (ts.to_numpy() - created) / pd.Timedelta(hours=1)
    since_hours = np.nan_to_num(since_hours, nan=0.0, posinf=0.0)
    since_log = np.log1p(np.clip(since_hours, 0.0, None))

    return (
        torch.tensor(np.stack([amt, hour_sin, hour_cos, since_log], axis=1), dtype=torch.float32),
        ["amount_log", "hour_sin", "hour_cos", "since_creation_log"],
    )
