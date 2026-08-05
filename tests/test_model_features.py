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
