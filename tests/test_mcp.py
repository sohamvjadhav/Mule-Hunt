import asyncio
import json

import torch

from upifraud.assistant import load_deployed
from upifraud.dataset import load_graph
from upifraud.generate import generate_toy
from upifraud.mcp_server import build_tools
from upifraud.train import train_gnn


def _dm_trained(tmp_path, seed=7):
    data_dir = tmp_path / "data"
    generate_toy(data_dir, n_accounts=150, n_tx=600, n_rings=3, seed=seed)
    data = load_graph(data_dir, split="rings", test_rings=1, seed=seed)
    train_gnn(data, model_name="sage", hidden=16, epochs=6, patience=2,
              seed=seed, out_dir=tmp_path)
    torch.save(data, tmp_path / "graph.pt")
    return load_deployed(tmp_path)


def _ring_member(dm):
    for i, r in enumerate(dm.data.ring_id.tolist()):
        if int(r) >= 0:
            return dm.data.node_ids[i]
    raise AssertionError("no ring members")


def test_mcp_tools_registered(tmp_path):
    dm = _dm_trained(tmp_path)
    mcp = build_tools(dm)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {
        "network_summary_tool",
        "account_risk",
        "explain_account",
        "investigate",
        "ring_details",
        "top_risky",
        "counterfactual",
        "case_file",
    }


def test_mcp_counterfactual_and_case_file(tmp_path):
    dm = _dm_trained(tmp_path)
    mcp = build_tools(dm)
    aid = _ring_member(dm)

    def unpack(result):
        payload = result[0]
        while isinstance(payload, list):
            payload = payload[0]
        try:
            return json.loads(payload.text)
        except json.JSONDecodeError:
            return payload.text

    cf = unpack(asyncio.run(mcp.call_tool("counterfactual", {"account_id": aid, "k": 2})))
    assert cf["account_id"] == aid
    assert len(cf["dropped_edges"]) == 2
    assert cf["caveat"]

    doc = unpack(asyncio.run(mcp.call_tool("case_file", {"account_id": aid})))
    assert f"# Case file: {aid}" in doc
    assert "## 5. Recommendation" in doc

    summary = unpack(asyncio.run(mcp.call_tool("network_summary_tool", {})))
    assert summary["nodes"] == 150
