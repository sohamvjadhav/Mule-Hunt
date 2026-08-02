"""Node feature construction from account attributes and local graph structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def build_node_features(
    accounts_df: pd.DataFrame,
    edge_index: torch.Tensor,
    amounts: torch.Tensor,
    n_nodes: int,
    with_amount_stats: bool = False,
) -> torch.Tensor:
    """Build per-node features. Amount aggregates are optional: the fraud
    generator encodes ring edges with a distinctive amount, so including
    amount stats trivially reveals rings. The default (structural) features
    isolate the network-signal contribution to detection.
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


def drop_constant_columns(x: torch.Tensor) -> torch.Tensor:
    """Drop zero-variance features (e.g. constant account age in synthetic
    data). Constant columns become noise amplifiers after standardization.
    """
    std = x.std(dim=0)
    keep = (std > 1e-8).nonzero().flatten()
    return x[:, keep]
