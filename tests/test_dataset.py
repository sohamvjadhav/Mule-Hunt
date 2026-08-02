import torch

from upifraud.dataset import load_graph
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
    assert base.x.shape[1] == 7
    assert rich.x.shape[1] == 11
