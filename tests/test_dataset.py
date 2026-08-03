import numpy as np
import torch

from upifraud.dataset import load_graph
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
