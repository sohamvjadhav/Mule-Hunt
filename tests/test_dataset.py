import numpy as np
import pandas as pd
import torch

from upifraud.dataset import build_snapshots, count_same_ring_edges, load_graph
from upifraud.evaluate import operating_point
from upifraud.features import _triangle_counts
from upifraud.generate import generate_toy


def test_toy_graph_roundtrip(tmp_path):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=7)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=7)

    assert data.num_nodes == 200
    assert data.edge_index.shape[0] == 2
    assert data.x.shape[0] == 200
    assert int(data.y.sum()) > 0
    assert int(data.edge_label.sum()) > 0
    assert data.num_rings == 4


def test_ring_aware_split_holds_out_whole_rings(tmp_path):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=7)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=7)

    train = data.train_mask
    val = data.val_mask
    test = data.test_mask
    assert torch.all(train | val | test)
    assert not torch.any(train & test)
    assert not torch.any(train & val)
    assert not torch.any(val & test)

    normal_test = sum(1 for i in range(data.num_nodes) if int(data.ring_id[i]) == -1 and test[i])
    assert normal_test > 0


def test_ring_nodes_never_split_across_train_and_test(tmp_path):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=7)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=7)

    for r in range(int(data.num_rings)):
        members = [i for i in range(data.num_nodes) if int(data.ring_id[i]) == r]
        split_sets = {int(data.train_mask[i]) * 2 + int(data.test_mask[i]) for i in members}
        assert not (1 in split_sets and 2 in split_sets)

    assert sum(1 for i in range(data.num_nodes) if data.test_mask[i] and data.y[i]) >= 2


def test_feature_leak_flag_changes_dimension(tmp_path):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=7)
    base = load_graph(tmp_path, with_amount_stats=False)
    rich = load_graph(tmp_path, with_amount_stats=True)
    cycle = load_graph(tmp_path, with_cycle_counts=True)
    assert base.x.shape[1] == 7
    assert rich.x.shape[1] == 11
    assert cycle.x.shape[1] == 9


def test_operating_point_picks_best_f1_threshold(tmp_path):
    generate_toy(tmp_path, n_accounts=60, n_tx=300, n_rings=2, seed=7)
    data = load_graph(tmp_path, split="rings", test_rings=1, seed=7)
    mask = data.test_mask.numpy()
    s = data.y.numpy().astype(float)  # perfect full-length scores
    op = operating_point(data, s, split="test")
    assert op["f1"] == 1.0
    assert op["precision"] == 1.0
    assert op["recall"] == 1.0
    assert op["n_fraud_detected"] == int(data.y.numpy()[mask].sum())


def test_triangle_counts_on_planted_structure():
    src = np.array([0, 1, 0, 3, 4])  # triangle 0-1-2, chain 3-4-5
    dst = np.array([1, 2, 2, 4, 5])
    tri, cc = _triangle_counts(src, dst, n_nodes=6)
    assert tri[0] == 1 and tri[1] == 1 and tri[2] == 1
    assert tri[3] == 0 and tri[4] == 0 and tri[5] == 0
    assert cc[0] == 1.0 and cc[1] == 1.0 and cc[2] == 1.0
    assert cc[4] == 0.0


def test_generator_ring_activity_is_bursty(tmp_path):
    generate_toy(tmp_path, n_accounts=300, n_tx=2500, n_rings=5, burst_days=2, seed=5)
    tx = pd.read_csv(tmp_path / "transactions" / "transactions_0_0.csv")
    fraud_tx = pd.read_csv(tmp_path / "fraud" / "transactions_fraud.csv")
    cases = pd.read_csv(tmp_path / "fraud" / "fraud_cases.csv")
    fraud_tx["ts"] = pd.to_datetime(fraud_tx["timestamp"])
    tx["ts"] = pd.to_datetime(tx["timestamp"])

    all_members = set(cases["involved_accounts"].str.split("|").sum())
    for members in cases["involved_accounts"].str.split("|"):
        members = set(members)
        ring_tx = fraud_tx[
            fraud_tx["src_id"].isin(members) & fraud_tx["dst_id"].isin(members)
        ]
        assert len(ring_tx) > 0
        span = (ring_tx["ts"].max() - ring_tx["ts"].min()).total_seconds() / 86400
        assert span <= 2.5  # burst_days=2 plus a rounding day

    ring_noise = tx[tx["src_id"].isin(all_members)]
    assert len(ring_noise) > 0

    background = tx[~tx["src_id"].isin(all_members) & ~tx["dst_id"].isin(all_members)]
    assert (background["ts"].max() - background["ts"].min()).days > 300


def test_build_snapshots_cumulative(tmp_path):
    generate_toy(tmp_path, n_accounts=300, n_tx=2500, n_rings=5, burst_days=2, seed=5)
    data = load_graph(tmp_path, split="random", seed=1)
    snaps = build_snapshots(data, 4)
    assert len(snaps) == 4
    counts = [int(s.edge_index.size(1)) for s in snaps]
    assert counts == sorted(counts)
    assert counts[-1] == data.edge_index.size(1)
    assert counts[0] < counts[-1]
    for s in snaps:
        assert s.x.shape[0] == data.num_nodes
        assert s.x.shape[1] == len(s.feature_names)
        assert float(s.edge_ts.max()) <= float(s.boundary_ts) + 1e-6
        assert int((s.train_mask | s.val_mask | s.test_mask).sum()) == data.num_nodes
        assert s.edge_train.dtype == torch.bool


def test_temporal_features_flag_changes_dimension(tmp_path):
    generate_toy(tmp_path, n_accounts=300, n_tx=2500, n_rings=5, burst_days=2, seed=5)
    base = load_graph(tmp_path, with_temporal=False)
    temp = load_graph(tmp_path, with_temporal=True)
    assert base.x.shape[1] + 2 == temp.x.shape[1]
    assert not {"burst_recent_frac", "activity_span_log"} <= set(base.feature_names)
    assert {"burst_recent_frac", "activity_span_log"} <= set(temp.feature_names)


def test_snapshot_ring_edge_reveal_non_decreasing(tmp_path):
    generate_toy(tmp_path, n_accounts=300, n_tx=2500, n_rings=5, burst_days=2, seed=5)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=5)
    snaps = build_snapshots(data, 4)
    reveals = [count_same_ring_edges(s, s.test_mask) for s in snaps]
    assert reveals == sorted(reveals)
    assert reveals[-1] == count_same_ring_edges(data, data.test_mask)


def test_count_same_ring_edges_matches_edges(tmp_path):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=7)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=7)
    src, dst = data.edge_index
    expected = int(
        ((data.ring_id[src] >= 0) & (data.ring_id[src] == data.ring_id[dst])).sum()
    )
    assert count_same_ring_edges(data) == expected
    assert count_same_ring_edges(data, data.test_mask) <= expected
