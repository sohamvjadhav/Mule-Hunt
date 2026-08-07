
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy
from upifraud.models import GCN, GraphSAGE, build_model
from upifraud.train import train_gnn


def _toy_data(tmp_path):
    generate_toy(tmp_path, n_accounts=120, n_tx=500, n_rings=3, seed=3)
    return load_graph(tmp_path, split="rings", test_rings=1, seed=3)


def test_models_forward_shape(tmp_path):
    data = _toy_data(tmp_path)
    for model in (GCN(data.x.size(1)), GraphSAGE(data.x.size(1))):
        out = model(data.x, data.edge_index)
        assert out.shape == (data.num_nodes,)


def test_unknown_model_raises(tmp_path):
    data = _toy_data(tmp_path)
    try:
        build_model("nope", data.x.size(1))
        assert False
    except ValueError:
        pass


def test_train_gnn_loss_decreases(tmp_path):
    data = _toy_data(tmp_path)
    result = train_gnn(data, model_name="sage", hidden=16, epochs=15, patience=5, seed=1)
    first_loss = result["history"][0]["loss"]
    min_loss = min(h["loss"] for h in result["history"])
    assert min_loss < first_loss
    assert result["args"]["best_val_ap"] > 0.0
