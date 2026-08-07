import random

import pandas as pd

from upifraud.cli import _parse_budgets, _perturb
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy


def _graph_context(data_dir, seed=5):
    data = load_graph(data_dir, split="rings", test_rings=2, seed=seed)
    node_ids = data.node_ids
    ring_ids = [int(r) for r in data.ring_id]
    test_ring_ids = {int(r) for r in data.ring_id[data.test_mask & (data.ring_id >= 0)]}
    ring_accounts = {node_ids[i] for i in range(len(node_ids)) if ring_ids[i] in test_ring_ids}
    all_ring = {node_ids[i] for i in range(len(node_ids)) if ring_ids[i] >= 0}
    normal_accounts = sorted(set(node_ids) - all_ring)
    src, dst = data.edge_index
    same_ring_test = (
        (data.ring_id[src] >= 0)
        & (data.ring_id[src] == data.ring_id[dst])
        & data.test_mask[src]
    )
    ring_pairs = {
        (node_ids[src[i]], node_ids[dst[i]])
        for i in same_ring_test.nonzero().flatten().tolist()
    }
    return data, ring_accounts, ring_pairs, normal_accounts


def test_parse_budgets():
    assert _parse_budgets("0.25,0.5,1.0") == [0.25, 0.5, 1.0]
    assert _parse_budgets("1") == [1.0]
    assert _parse_budgets("0.5,,") == [0.5]


def test_perturb_inject_adds_camouflage_edges(tmp_path):
    data_dir = tmp_path / "data"
    generate_toy(data_dir, n_accounts=300, n_tx=2500, n_rings=5, seed=5)
    data, ring_accounts, ring_pairs, normal_accounts = _graph_context(data_dir)

    pert_dir = _perturb(
        data_dir, tmp_path / "out", "inject", 1.0, len(ring_pairs),
        ring_accounts, ring_pairs, normal_accounts, random.Random(5),
    )
    tx = pd.read_csv(pert_dir / "transactions" / "transactions_0_0.csv")
    attacks = tx[tx["tx_id"].str.startswith("attack_")]
    assert len(attacks) == len(ring_pairs)
    assert (attacks["amount"] <= 500).all()
    assert set(attacks["src_id"]) <= ring_accounts
    assert set(attacks["dst_id"]) <= set(normal_accounts)

    g = load_graph(pert_dir, split="rings", test_rings=2, seed=5)
    assert g.node_ids == data.node_ids
    assert int(g.edge_index.size(1)) == int(data.edge_index.size(1)) + len(ring_pairs)
    assert int(g.edge_label.sum()) == int(data.edge_label.sum())
    assert int(g.train_mask.sum()) == int(data.train_mask.sum())


def test_perturb_drop_removes_ring_evidence(tmp_path):
    data_dir = tmp_path / "data"
    generate_toy(data_dir, n_accounts=300, n_tx=2500, n_rings=5, seed=5)
    data, ring_accounts, ring_pairs, normal_accounts = _graph_context(data_dir)
    n_ring = len(ring_pairs)
    n_fraud_before = len(pd.read_csv(data_dir / "fraud" / "transactions_fraud.csv"))

    pert_dir = _perturb(
        data_dir, tmp_path / "out", "drop", 1.0, n_ring,
        ring_accounts, ring_pairs, normal_accounts, random.Random(5),
    )
    tx = pd.read_csv(pert_dir / "transactions" / "transactions_0_0.csv")
    fraud = pd.read_csv(pert_dir / "fraud" / "transactions_fraud.csv")
    n_tx_before = len(pd.read_csv(data_dir / "transactions" / "transactions_0_0.csv"))
    n_dropped_fraud = n_fraud_before - len(fraud)
    assert n_dropped_fraud > 0
    assert len(tx) + len(fraud) == n_tx_before + n_fraud_before - n_ring

    g = load_graph(pert_dir, split="rings", test_rings=2, seed=5)
    assert g.node_ids == data.node_ids
    assert int(g.edge_index.size(1)) == int(data.edge_index.size(1)) - n_ring
    assert int(g.edge_label.sum()) == int(data.edge_label.sum()) - n_dropped_fraud