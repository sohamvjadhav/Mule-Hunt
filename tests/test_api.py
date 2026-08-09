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


def _make_client(tmp_path):

    from upifraud.api import create_app

    generate_toy(tmp_path / "data", n_accounts=120, n_tx=500, n_rings=3, seed=3)
    data = load_graph(tmp_path / "data", split="rings", test_rings=1, seed=3)
    train_gnn(data, model_name="sage", hidden=16, epochs=8, patience=3, seed=1, out_dir=tmp_path)
    torch.save(data, tmp_path / "graph.pt")
    return TestClient(create_app(tmp_path, tmp_path / "graph.pt"))


def test_dashboard_endpoints(tmp_path):
    client = _make_client(tmp_path)

    summary = client.get("/api/summary").json()
    assert summary["n_accounts"] == 120
    assert summary["n_rings"] == 3
    assert summary["n_fraud"] > 0
    assert summary["n_transactions"] > 0

    top = client.get("/api/top?k=10").json()
    assert len(top) == 10
    assert top[0]["rank"] == 1
    assert 0.0 <= top[0]["risk_score"] <= 1.0
    assert "ring_id" in top[0] and "degree" in top[0]

    detail = client.get(f"/api/account/{top[0]['account_id']}").json()
    assert detail["degree"] >= 0
    assert isinstance(detail["neighbors"], list)
    assert detail["true_label"] in (0, 1)

    ring = client.get("/api/ring/0").json()
    assert ring["size"] > 0
    assert len(ring["nodes"]) == ring["size"]
    assert isinstance(ring["edges"], list)
    assert client.get("/api/ring/99").status_code == 404

    dist = client.get("/api/distribution").json()
    assert len(dist["bins"]) == len(dist["counts"]) + 1
    assert sum(dist["counts"]) == 120


def test_frontend_served(tmp_path):
    client = _make_client(tmp_path)
    html = client.get("/").text
    assert "Mule-Hunt" in html
    assert "Investigation assistant" in html and 'id="ask-form"' in html
    assert client.get("/styles.css").status_code == 200
    assert client.get("/app.js").status_code == 200


def test_explain_fallback_local(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _make_client(tmp_path)
    summary = client.get("/api/summary").json()
    assert summary["explainer"] == "local"

    top = client.get("/api/top?k=1").json()
    resp = client.get(f"/api/explain/{top[0]['account_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "local"
    assert len(body["explanation"]) > 20
    assert isinstance(body["model_evidence"], dict)

    assert client.get("/api/explain/nope").status_code == 404


def test_ask_endpoint_routes_questions(tmp_path):
    client = _make_client(tmp_path)
    top = client.get("/api/top?k=1").json()
    aid = top[0]["account_id"]

    for q, kind, key in (
        (f"why is {aid} risky?", "account", "report"),
        ("describe ring 0", "ring", "report"),
        ("show me the top risky accounts", "top", "accounts"),
        ("give me a network summary", "summary", "facts"),
    ):
        resp = client.post("/api/ask", json={"question": q})
        assert resp.status_code == 200, q
        body = resp.json()
        assert body["kind"] == kind, q
        assert key in body and body["report"]

    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    unknown = client.post("/api/ask", json={"question": "what time is it"}).json()
    assert unknown["kind"] == "unknown" and unknown["report"]
