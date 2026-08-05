import pickle

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from upifraud.api import create_app
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy
from upifraud.models import MODELS, GATv2, build_model
from upifraud.train import train_gnn


def _toy(tmp_path, n_accounts=120, n_tx=500, n_rings=3):
    generate_toy(tmp_path / "data", n_accounts=n_accounts, n_tx=n_tx, n_rings=n_rings, seed=3)
    return load_graph(tmp_path / "data", split="rings", test_rings=1, seed=3)


def test_gat_arch_registered_and_forward():
    assert "gat" in MODELS
    model = build_model("gat", in_dim=7, hidden=16)
    assert isinstance(model, GATv2)
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    out = model(x, edge_index)
    assert out.shape == (10,)
    assert torch.isfinite(out).all()


def test_gat_trains_end_to_end(tmp_path):
    data = _toy(tmp_path)
    result = train_gnn(data, model_name="gat", hidden=16, epochs=8, patience=3, seed=1, out_dir=tmp_path)
    assert result["args"]["model"] == "gat"
    assert len(result["scores"]) == data.num_nodes
    assert (tmp_path / "gat.pt").exists()
    assert 0.0 <= float(np.nanmin(result["scores"])) <= 1.0


def test_calibration_saves_artifacts_and_improves_brier(tmp_path):
    data = _toy(tmp_path)
    cal = train_gnn(data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1, out_dir=tmp_path)
    assert (tmp_path / "sage_calib.pkl").exists()
    assert cal["args"]["calibrator"] == "isotonic"
    assert cal["args"]["brier_val_after"] <= cal["args"]["brier_val_before"] + 1e-9

    with (tmp_path / "sage_calib.pkl").open("rb") as f:
        iso = pickle.load(f)
    assert 0.0 <= float(iso.predict([0.5])[0]) <= 1.0


def test_calibration_can_be_disabled(tmp_path):
    data = _toy(tmp_path)
    result = train_gnn(
        data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1,
        out_dir=tmp_path, calibrate=False,
    )
    assert result["args"]["calibrator"] is None
    assert not (tmp_path / "sage_calib.pkl").exists()


def test_cold_start_saved_with_meta(tmp_path):
    data = _toy(tmp_path)
    result = train_gnn(
        data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1,
        out_dir=tmp_path, cold_start_threshold=10,
    )
    cs = result["args"]["cold_start"]
    assert cs is not None
    assert cs["threshold"] == 10
    assert (tmp_path / "sage_coldstart.joblib").exists()
    assert cs["feature_names"] == ["balance", "risk_score", "age_days"]
    assert joblib.load(tmp_path / "sage_coldstart.joblib").classes_.shape[0] == 2


def _client(tmp_path, cold_start_threshold=10):
    data = _toy(tmp_path)
    train_gnn(
        data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1,
        out_dir=tmp_path, cold_start_threshold=cold_start_threshold,
    )
    torch.save(data, tmp_path / "graph.pt")
    return TestClient(create_app(tmp_path, tmp_path / "graph.pt")), data


def test_api_uses_cold_start_for_low_degree_accounts(tmp_path):
    client, data = _client(tmp_path)
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    degree = np.bincount(np.concatenate([src, dst]), minlength=data.num_nodes)
    low = np.where(degree < 10)[0]
    assert len(low) > 0

    cold_start = joblib.load(tmp_path / "sage_coldstart.joblib")
    cs_idx = [0, 1, 2]
    for i in low.tolist()[:3]:
        aid = data.node_ids[i]
        body = client.get(f"/risk/account/{aid}").json()
        expected = float(cold_start.predict_proba(data.x_raw[i, cs_idx].numpy().reshape(1, -1))[0, 1])
        assert body["risk_score"] == pytest.approx(expected, abs=1e-4)


def test_api_drift_endpoint(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/drift").json()
    assert body["metric"] == "PSI"
    assert body["verdict"] in ("stable", "minor_drift", "major_drift")
    assert body["bins"] == 10
    assert body["train_size"] > 0 and body["test_size"] > 0
    assert body["calibrator"] is True
    assert client.get("/api/drift?bins=2").status_code == 400
    assert client.get("/api/drift?bins=60").status_code == 400
    assert client.get("/api/drift?bins=20").status_code == 200


def test_summary_exposes_calibration_fields(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/summary").json()
    assert body["calibrator"] == "isotonic"
    assert 0.0 <= body["brier_val_after"] <= 1.0
    assert body["cold_start_threshold"] == 10


def test_depth_and_jk_shapes():
    for name in ("gcn", "sage", "gat"):
        for layers in (1, 2, 4):
            m = build_model(name, in_dim=7, hidden=8, num_layers=layers)
            x = torch.randn(10, 7)
            e = torch.tensor([[0, 1, 2], [1, 2, 0]])
            assert m(x, e).shape == (10,)
            emb = m.embed(x, e)
            expected = 8 * layers if layers > 1 else 8
            assert emb.shape == (10, expected), (name, layers)


def test_edge_head_shapes_and_optionality():
    m = build_model("sage", in_dim=7, hidden=8, edge_attr_dim=4)
    assert m.edge_head is not None
    x = torch.randn(10, 7)
    e = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ea = torch.randn(3, 4)
    assert m.edge_forward(x, e, ea).shape == (3,)
    assert torch.isfinite(m.edge_forward(x, e, ea)).all()

    plain = build_model("sage", in_dim=7, hidden=8)
    assert plain.edge_head is None


def test_dataset_edge_features_and_masks(tmp_path):
    data = _toy(tmp_path)
    assert data.edge_attr.shape[1] == 4
    assert data.edge_feature_names == ["amount_log", "hour_sin", "hour_cos", "since_creation_log"]
    assert data.edge_train.dtype == torch.bool
    assert data.edge_train.any() and data.edge_val.any() and data.edge_test.any()
    e_pos = int(data.edge_label.sum())
    assert e_pos > 0
    same_ring = (data.ring_id[data.edge_index[0]] >= 0) & (
        data.ring_id[data.edge_index[0]] == data.ring_id[data.edge_index[1]]
    )
    assert not (data.edge_label.bool() & ~same_ring).any()
    assert int((data.edge_label.bool() & same_ring).sum()) == e_pos
    assert int(data.edge_label[data.edge_test].sum()) > 0


def test_edge_head_trains_and_reports_metrics(tmp_path):
    data = _toy(tmp_path)
    result = train_gnn(
        data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1,
        out_dir=tmp_path, edge_loss_weight=1.0,
    )
    assert result["edge_scores"] is not None
    assert len(result["edge_scores"]) == data.edge_index.size(1)
    ev = result["edge_eval"]
    assert ev is not None
    assert ev["n_pos"] > 0 and ev["n_total"] > ev["n_pos"]
    assert ev["auc"] is not None and 0.0 <= ev["auc"] <= 1.0
    assert (tmp_path / "sage_args.json").exists()
    args = result["args"]
    assert args["num_layers"] == 3 and args["edge_attr_dim"] == 4
    assert args["edge_attr_mean"] and args["edge_attr_std"]


def test_edge_loss_can_be_disabled(tmp_path):
    data = _toy(tmp_path)
    result = train_gnn(
        data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1,
        out_dir=tmp_path, edge_loss_weight=0.0,
    )
    assert result["edge_scores"] is None
    assert result["edge_eval"] is None
    assert result["args"]["edge_attr_dim"] is None


def test_ring_endpoint_returns_transactions(tmp_path):
    client, _ = _client(tmp_path)
    ring = client.get("/api/ring/0").json()
    assert len(ring["transactions"]) > 0
    t = ring["transactions"][0]
    assert set(t) >= {"src", "dst", "amount", "risk_score", "label"}
    assert 0.0 <= t["amount"]
    assert t["risk_score"] is None or 0.0 <= t["risk_score"] <= 1.0
    assert t["label"] in (0, 1)
    internal = [e for e in ring["edges"]]
    assert (len(internal)) == len(ring["transactions"])
    assert client.get("/api/ring/99").status_code == 404
