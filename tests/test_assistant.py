import numpy as np

from upifraud.assistant import (
    DeployedModel,
    account_facts,
    account_report,
    answer,
    ring_report,
    top_accounts,
)
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy


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
