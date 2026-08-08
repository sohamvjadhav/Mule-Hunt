"""Grounded natural-language investigation reports over a trained risk graph.

The assistant answers questions like "why is account acc_42 risky?" with
template-generated prose built entirely from graph facts (cycle membership,
amounts, timing, neighbor structure, and the model's own explanation
drivers). There is no LLM and nothing is hallucinated: every sentence is a
rendered fact. This makes the output safe for a human investigator and
directly consumable by a coding agent (via ``upifraud query`` and the MCP
server).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .api import band
from .models import build_model
from .train import standardize

RING_BANDS = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 1.0),
}


class DeployedModel:
    """Loaded checkpoint + graph, mirroring what the risk API serves."""

    def __init__(
        self,
        data: Data,
        model,
        scores: np.ndarray,
        edge_scores: np.ndarray | None,
        args: dict,
        rank_map: dict[int, int],
        id_to_idx: dict[str, int],
    ):
        self.data = data
        self.model = model
        self.scores = scores
        self.edge_scores = edge_scores
        self.args = args
        self.rank_map = rank_map
        self.id_to_idx = id_to_idx
        self.src = data.edge_index[0].numpy()
        self.dst = data.edge_index[1].numpy()
        self.degree = np.bincount(
            np.concatenate([self.src, self.dst]), minlength=data.num_nodes
        )

    def account(self, account_id: str) -> int:
        try:
            return self.id_to_idx[account_id]
        except KeyError:
            raise ValueError(f"unknown account {account_id!r}") from None


def load_deployed(checkpoint_dir: Path, dataset_path: Path | None = None) -> DeployedModel:
    """Load a model + graph exactly as the API does (scores calibrated)."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates = sorted(p for p in checkpoint_dir.glob("*.pt") if p.stem != "graph")
    if not candidates:
        raise FileNotFoundError(f"no model checkpoint (*.pt) in {checkpoint_dir}")
    checkpoint = candidates[0]
    args = json.loads((checkpoint_dir / f"{checkpoint.stem}_args.json").read_text())
    state = torch.load(checkpoint, map_location="cpu")
    if dataset_path is None:
        dataset_path = checkpoint_dir / "graph.pt"
    data: Data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    mean = torch.tensor(args["mean"])
    std = torch.tensor(args["std"])
    x, _, _ = standardize(data.x, mean, std)
    model = build_model(
        args["model"], int(args["in_dim"]), int(args["hidden"]),
        num_layers=int(args.get("num_layers", 2)),
        jk=args.get("jk", "cat"),
        edge_attr_dim=args.get("edge_attr_dim"),
    )
    model.load_state_dict(state)
    model.eval()

    edge_scores = None
    if getattr(model, "edge_head", None) is not None and getattr(data, "edge_attr", None) is not None:
        em = torch.tensor(args.get("edge_attr_mean"))
        es = torch.tensor(args.get("edge_attr_std"))
        eattr, _, _ = standardize(data.edge_attr.to(torch.float32), em, es)
        with torch.no_grad():
            edge_scores = model.edge_forward(x, data.edge_index, eattr).sigmoid().numpy().astype(float)

    calibrator = None
    calib_path = checkpoint_dir / f"{checkpoint.stem}_calib.pkl"
    if calib_path.exists():
        with calib_path.open("rb") as f:
            calibrator = pickle.load(f)

    with torch.no_grad():
        raw = model(x, data.edge_index).sigmoid().numpy()
    scores = calibrator.predict(raw).astype(float) if calibrator is not None else raw.astype(float)
    order = np.argsort(-scores)
    rank_map = {int(i): r for r, i in enumerate(order)}
    id_to_idx = {aid: i for i, aid in enumerate(data.node_ids)}
    return DeployedModel(data, model, scores, edge_scores, args, rank_map, id_to_idx)


def _money(amounts: np.ndarray) -> str:
    total = float(amounts.sum())
    if total >= 1e7:
        return f"₹{total / 1e7:.2f} crore"
    if total >= 1e5:
        return f"₹{total / 1e5:.2f} lakh"
    return f"₹{total:,.0f}"


def account_facts(dm: DeployedModel, account_id: str) -> dict:
    """Structured facts about one account, all graph-grounded."""
    i = dm.account(account_id)
    d = dm.data
    y = d.y.numpy()
    ring_id = d.ring_id.numpy()
    src, dst = dm.src, dm.dst
    outs = np.flatnonzero(src == i)
    ins = np.flatnonzero(dst == i)
    edge_ts = getattr(d, "edge_ts", None)
    ts = edge_ts.numpy() if edge_ts is not None else None

    involved = np.unique(np.concatenate([dst[outs], src[ins]]))
    amount = d.edge_amounts.numpy()
    times = []
    if ts is not None and len(involved) > 0:
        et = np.concatenate([ts[outs], ts[ins]])
        if len(et):
            times = [float(et.min()), float(et.max()), len(et)]

    ring_members = []
    ring_amt = 0.0
    ring_n = 0
    ring_days = 0.0
    r = int(ring_id[i])
    if r >= 0:
        members = np.flatnonzero(ring_id == r)
        ring_members = [d.node_ids[m] for m in members]
        internal = (ring_id[src] == r) & (ring_id[dst] == r)
        ring_amt = float(amount[internal].sum())
        ring_n = int(internal.sum())
        if ts is not None and ring_n:
            ring_days = float((ts[internal].max() - ts[internal].min()) / 86400.0)

    edge_scores = dm.edge_scores
    hot_edges = []
    if edge_scores is not None:
        cand = np.concatenate([outs, ins])
        if len(cand):
            es = edge_scores[cand]
            top = np.argsort(-es)[:5]
            for k in top:
                e = int(cand[int(k)])
                hot_edges.append(
                    {
                        "src": d.node_ids[int(src[e])],
                        "dst": d.node_ids[int(dst[e])],
                        "amount": float(amount[e]),
                        "risk": float(es[int(k)]),
                    }
                )

    neighbors = []
    for nb in involved[:8]:
        neighbors.append(
            {
                "account": d.node_ids[int(nb)],
                "risk_score": float(dm.scores[nb]),
                "label": int(y[nb]),
            }
        )

    return {
        "account_id": account_id,
        "risk_score": float(dm.scores[i]),
        "risk_band": band(float(dm.scores[i])),
        "rank": int(dm.rank_map.get(i, -1)) + 1,
        "label": int(y[i]),
        "ring_id": r,
        "degree": int(dm.degree[i]),
        "n_out_edges": len(outs),
        "n_in_edges": len(ins),
        "ring_members": ring_members,
        "ring_edges": ring_n,
        "ring_amount": ring_amt,
        "ring_days": ring_days,
        "activity_start": times[0] if times else None,
        "activity_end": times[1] if times else None,
        "activity_edges": times[2] if times else 0,
        "neighbors": neighbors,
        "hot_edges": hot_edges,
    }


def account_report(dm: DeployedModel, account_id: str) -> str:
    """A rendered investigation report for one account."""
    f = account_facts(dm, account_id)
    lines = [
        f"Investigation: {f['account_id']}",
        (
            f"Risk: {f['risk_score']:.3f} ({f['risk_band'].upper()}) "
            f"— ranked {f['rank']:,} of {dm.data.num_nodes:,} accounts."
        ),
    ]
    if f["ring_id"] >= 0:
        lines.append(
            f"Ring member: {f['account_id']} sits in a planted ring of "
            f"{len(f['ring_members'])} accounts ({f['ring_edges']} internal transfers, "
            f"{_money(np.array([f['ring_amount']]))} moved "
            f"across {max(1, round(f['ring_days']))} day(s))."
        )
    lines.append(f"Activity: {f['n_out_edges']} outgoing and {f['n_in_edges']} incoming "
                 f"transfers (degree {f['degree']}).")
    if f["activity_edges"]:
        lines.append(
            f"Timing: first edge at {_ts(f['activity_start'])}, last at "
            f"{_ts(f['activity_end'])} — {f['activity_edges']} timestamped edges."
        )
    if f["neighbors"]:
        top = sorted(f["neighbors"], key=lambda n: -n["risk_score"])[:3]
        lines.append("Highest-risk counterparties: "
                     + ", ".join(f"{n['account']} ({n['risk_score']:.3f})" for n in top) + ".")
    if f["hot_edges"]:
        he = f["hot_edges"][0]
        lines.append(
            f"Top suspicious transfer: {he['src']} → {he['dst']} "
            f"({_money(np.array([he['amount']]))}) with transaction risk "
            f"{he['risk']:.3f}."
        )
    lines.append(f"Model label: {'fraud (ring)' if f['label'] else 'normal'}.")
    return "\n".join(lines)


def _ts(epoch: float) -> str:
    return str(epoch) if epoch is None else f"{epoch:.0f} (epoch s)"


def ring_facts(dm: DeployedModel, ring_id: int) -> dict:
    d = dm.data
    ring = d.ring_id.numpy()
    members = np.flatnonzero(ring == ring_id)
    if not len(members):
        raise ValueError(f"unknown ring {ring_id}")
    internal = (ring[d.edge_index[0].numpy()] == ring_id) & (ring[d.edge_index[1].numpy()] == ring_id)
    amt = d.edge_amounts.numpy()[internal]
    ts = getattr(d, "edge_ts", None)
    span = None
    if ts is not None and internal.any():
        t = ts.numpy()[internal]
        span = float((t.max() - t.min()) / 86400.0)
    return {
        "ring_id": ring_id,
        "members": [d.node_ids[m] for m in members],
        "n_internal_edges": int(internal.sum()),
        "total_amount": float(amt.sum()),
        "span_days": span,
        "member_scores": [
            {"account": d.node_ids[m], "risk_score": float(dm.scores[m])} for m in members
        ],
    }


def ring_report(dm: DeployedModel, ring_id: int) -> str:
    f = ring_facts(dm, ring_id)
    lines = [f"Ring {f['ring_id']}",
             f"{len(f['members'])} members: {', '.join(f['members'])}.",
             f"{f['n_internal_edges']} internal transfers totaling "
             f"{_money(np.array([f['total_amount']]))}"
             + (f" over {max(1, round(f['span_days']))} day(s)." if f["span_days"] else ".")]
    for m in sorted(f["member_scores"], key=lambda x: -x["risk_score"]):
        lines.append(f"  {m['account']}: risk {m['risk_score']:.3f}")
    return "\n".join(lines)


def top_accounts(dm: DeployedModel, k: int = 10) -> list[dict]:
    order = np.argsort(-dm.scores)
    return [
        {
            "account_id": dm.data.node_ids[i],
            "risk_score": float(dm.scores[i]),
            "band": band(float(dm.scores[i])),
            "label": int(dm.data.y.numpy()[i]),
            "ring_id": int(dm.data.ring_id.numpy()[i]),
        }
        for i in order[:k]
    ]


def network_summary(dm: DeployedModel) -> dict:
    d = dm.data
    y = d.y.numpy()
    s = dm.scores
    frac = float(np.mean(s[d.test_mask.numpy()] >= 0.7))
    return {
        "model": dm.args["model"],
        "nodes": int(d.num_nodes),
        "edges": int(d.edge_index.size(1)),
        "rings": int(d.num_rings),
        "fraud_nodes": int(y.sum()),
        "risk_frac_high": round(frac, 4),
        "test_auc_approx": "see results/ for benchmark metrics",
        "cold_start": dm.args.get("cold_start", None) is not None,
    }


def answer(dm: DeployedModel, question: str, k: int = 10) -> dict:
    """Route a natural-language question and return a structured answer."""
    q = question.lower()

    account = _extract_account(q, dm)
    ring = _extract_ring(q)
    if ring is not None and any(w in q for w in ("ring", "cycle", "group")):
        return {"kind": "ring", "question": question, "ring_id": ring,
                "facts": ring_facts(dm, ring), "report": ring_report(dm, ring)}
    if any(w in q for w in ("top", "highest risk", "riskiest", "worst")):
        rows = top_accounts(dm, k)
        text = "Top risky accounts:\n" + "\n".join(
            f"  {r['account_id']}: {r['risk_score']:.3f} ({r['band']})" for r in rows
        )
        return {"kind": "top", "question": question, "accounts": rows, "report": text}
    if any(w in q for w in ("summary", "overview", "how many", "statistics")):
        s = network_summary(dm)
        text = (
            f"Network: {s['nodes']:,} accounts, {s['edges']:,} transfers, "
            f"{s['rings']} rings, {s['fraud_nodes']} fraud-labeled accounts "
            f"(model {s['model']})."
        )
        return {"kind": "summary", "question": question, "facts": s, "report": text}
    if account is not None:
        if any(w in q for w in ("why", "explain", "risk", "flag", "suspicious", "investigate")):
            return {"kind": "account", "question": question, "account_id": account,
                    "facts": account_facts(dm, account), "report": account_report(dm, account)}
        return {"kind": "account", "question": question, "account_id": account,
                "facts": account_facts(dm, account), "report": account_report(dm, account)}
    return {"kind": "unknown", "question": question,
            "report": "Ask about an account (\"why is acc_1 risky?\"), a ring "
                      "(\"describe ring 2\"), the top accounts, or a network summary."}


def _extract_account(q: str, dm: DeployedModel) -> str | None:
    matches = [aid for aid in dm.id_to_idx if aid in q]
    if not matches:
        return None
    return max(matches, key=len)


def _extract_ring(q: str) -> int | None:
    import re

    m = re.search(r"(?:ring|cycle)\s+(\d+)", q)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"ring\s+\d+", q.strip()):
        return int(q.strip().split()[-1])
    return None
