import numpy as np
import torch

from upifraud.assistant import (
    DeployedModel,
    account_facts,
    account_report,
    answer,
    case_document,
    counterfactual,
    load_deployed,
    ring_report,
    top_accounts,
)
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy
from upifraud.train import train_gnn


def _dm(tmp_path, seed=5):
    generate_toy(tmp_path, n_accounts=200, n_tx=800, n_rings=4, seed=seed)
    data = load_graph(tmp_path, split="rings", test_rings=2, seed=seed)
    scores = data.y.numpy().astype(float)
    order = np.argsort(-scores)
    rank_map = {int(i): r for r, i in enumerate(order)}
    id_to_idx = {aid: i for i, aid in enumerate(data.node_ids)}
    return DeployedModel(
        data, None, scores, None, {"model": "test"}, rank_map, id_to_idx
    )


def _ring_member(dm):
    for i, r in enumerate(dm.data.ring_id.tolist()):
        if int(r) >= 0:
            return dm.data.node_ids[i], int(r)
    raise AssertionError("no ring members")


def test_account_facts_are_graph_grounded(tmp_path):
    dm = _dm(tmp_path)
    aid, _ = _ring_member(dm)
    f = account_facts(dm, aid)
    assert f["risk_score"] == 1.0
    assert f["risk_band"] == "high"
    assert f["label"] == 1
    assert f["ring_id"] >= 0
    assert len(f["ring_members"]) >= 4
    assert f["ring_edges"] > 0
    assert f["ring_amount"] > 0
    i = dm.id_to_idx[aid]
    assert f["degree"] == f["n_out_edges"] + f["n_in_edges"]
    assert f["n_out_edges"] + f["n_in_edges"] == int(
        (dm.data.edge_index[0].numpy() == i).sum()
        + (dm.data.edge_index[1].numpy() == i).sum()
    )


def test_account_report_reads_like_an_investigation(tmp_path):
    dm = _dm(tmp_path)
    aid, _ = _ring_member(dm)
    report = account_report(dm, aid)
    assert aid in report
    assert "HIGH" in report
    assert "Ring member" in report


def test_ring_report_lists_members_and_amount(tmp_path):
    dm = _dm(tmp_path)
    _, ring = _ring_member(dm)
    report = ring_report(dm, ring)
    assert f"Ring {ring}" in report
    assert "₹" in report


def test_answer_routes_intents(tmp_path):
    dm = _dm(tmp_path)
    aid, ring = _ring_member(dm)

    out = answer(dm, f"why is {aid} risky?")
    assert out["kind"] == "account"
    assert out["account_id"] == aid
    assert out["facts"]["risk_band"] == "high"

    out = answer(dm, f"describe ring {ring}")
    assert out["kind"] == "ring"
    assert out["ring_id"] == ring

    out = answer(dm, "top accounts")
    assert out["kind"] == "top"
    assert len(out["accounts"]) == 10

    out = answer(dm, "give me a summary")
    assert out["kind"] == "summary"
    assert out["facts"]["nodes"] == 200

    out = answer(dm, "what is the meaning of life?")
    assert out["kind"] == "unknown"


def test_top_accounts_sorted_desc(tmp_path):
    dm = _dm(tmp_path)
    rows = top_accounts(dm, 5)
    assert len(rows) == 5
    assert [r["risk_score"] for r in rows] == sorted(
        (r["risk_score"] for r in rows), reverse=True
    )


def _dm_trained(tmp_path, seed=7):
    data_dir = tmp_path / "data"
    generate_toy(data_dir, n_accounts=150, n_tx=600, n_rings=3, seed=seed)
    data = load_graph(data_dir, split="rings", test_rings=1, seed=seed)
    train_gnn(data, model_name="sage", hidden=16, epochs=6, patience=2,
              seed=seed, out_dir=tmp_path)
    torch.save(data, tmp_path / "graph.pt")
    return load_deployed(tmp_path)


def _ring_member_id(dm):
    aid, _ = _ring_member(dm)
    return aid


def test_counterfactual_drops_incident_edges_and_rescores(tmp_path):
    dm = _dm_trained(tmp_path)
    aid = _ring_member_id(dm)
    i = dm.id_to_idx[aid]
    src, dst = dm.src, dm.dst

    cf = counterfactual(dm, aid, k=2)
    assert len(cf["dropped_edges"]) == 2
    for e in cf["dropped_edges"]:
        idx = np.flatnonzero((src == dm.id_to_idx[e["src"]]) & (dst == dm.id_to_idx[e["dst"]]))
        assert len(idx) == 1 and int(idx[0]) in np.flatnonzero((src == i) | (dst == i))
    assert 0.0 <= cf["score_served"] <= 1.0
    assert cf["band_served"] in ("low", "medium", "high")
    assert abs(cf["delta_model_score"]) < 1.0
    assert "fixed-model sensitivity" in cf["caveat"]

    try:
        counterfactual(dm, "does_not_exist")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_case_document_is_a_complete_markdown_file(tmp_path):
    dm = _dm_trained(tmp_path)
    aid = _ring_member_id(dm)
    doc = case_document(dm, aid, k=2, top_tx=3)
    assert f"# Case file: {aid}" in doc
    for section in (
        "## 1. Subject",
        "## 2. Ring context",
        "## 3. Top suspicious transactions",
        "## 4. Counterfactual sensitivity",
        "## 5. Recommendation",
    ):
        assert section in doc
    assert "mule-hunt" in doc and "no LLM" in doc
    assert "Recommendation" in doc and doc.index("## 5. Recommendation") < len(doc)


def test_answer_routes_case_and_counterfactual_intents(tmp_path):
    dm = _dm_trained(tmp_path)
    aid = _ring_member_id(dm)

    out = answer(dm, f"write a case file for {aid}")
    assert out["kind"] == "case"
    assert out["account_id"] == aid
    assert "# Case file:" in out["document"]

    out = answer(dm, f"what if {aid} had no ring ties?")
    assert out["kind"] == "counterfactual"
    assert out["facts"]["account_id"] == aid
    assert "Counterfactual for" in out["report"]
