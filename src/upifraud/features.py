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

    unique_in = np.zeros(n_nodes)
    unique_out = np.zeros(n_nodes)
    for i in range(n_nodes):
        unique_out[i] = len(np.unique(src[dst == i])) if np.any(dst == i) else 0
        unique_in[i] = len(np.unique(dst[src == i])) if np.any(src == i) else 0
    features += [unique_in, unique_out]

    if with_amount_stats:
        in_amt = np.zeros(n_nodes)
        out_amt = np.zeros(n_nodes)
        in_cnt = np.zeros(n_nodes)
        out_cnt = np.zeros(n_nodes)
        for i in range(len(amt)):
            out_amt[src[i]] += amt[i]
            in_amt[dst[i]] += amt[i]
            out_cnt[src[i]] += 1
            in_cnt[dst[i]] += 1
        features += [
            np.log1p(in_amt / np.maximum(in_cnt, 1)),
            np.log1p(out_amt / np.maximum(out_cnt, 1)),
            np.log1p(in_amt),
            np.log1p(out_amt),
        ]

    return torch.tensor(np.stack(features, axis=1), dtype=torch.float32)


def drop_constant_columns(x: torch.Tensor) -> torch.Tensor:
    """Drop zero-variance features (e.g. constant account age in synthetic
    data). Constant columns become noise amplifiers after standardization.
    """
    std = x.std(dim=0)
    keep = (std > 1e-8).nonzero().flatten()
    return x[:, keep]
