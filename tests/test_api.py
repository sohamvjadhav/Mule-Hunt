import torch
from fastapi.testclient import TestClient

from upifraud.api import create_app
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy
from upifraud.train import train_gnn


def test_api_end_to_end(tmp_path):
    generate_toy(tmp_path / "data", n_accounts=120, n_tx=500, n_rings=3, seed=3)
    data = load_graph(tmp_path / "data", split="rings", test_rings=1, seed=3)
    train_gnn(data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1, out_dir=tmp_path)
    torch.save(data, tmp_path / "graph.pt")

    client = TestClient(create_app(tmp_path, tmp_path / "graph.pt"))

    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["nodes"] == 120

    node_id = data.node_ids[5]
    resp = client.get(f"/risk/account/{node_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["account_id"] == node_id
    assert body["risk_band"] in ("low", "medium", "high")

    assert client.get("/risk/account/does_not_exist").status_code == 404

    batch = client.post("/risk/batch", json={"account_ids": [node_id, "nope"]})
    assert batch.status_code == 200
    assert len(batch.json()) == 2
