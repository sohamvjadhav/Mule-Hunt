"""MCP (stdio) server exposing the deployed risk graph to coding agents.

Any MCP-capable agent (Claude Code, Codex, opencode, ...) can attach to
``upifraud mcp`` and investigate the fraud graph with grounded tools:
account risk, explanations, full investigation reports, ring details, top
accounts, and network summaries. Everything is deterministic — no external
model calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from .assistant import DeployedModel

from .assistant import (
    account_facts,
    account_report,
    network_summary,
    top_accounts,
)


def build_tools(dm: DeployedModel) -> FastMCP:
    """Register all investigation tools on a FastMCP server (testable in-process)."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mule-hunt", instructions=(
        "Fraud-investigation tools over a trained GNN risk graph. Use "
        "network_summary first, then account_risk / explain_account / "
        "investigate for specific accounts. All outputs are grounded in the "
        "graph; there is no LLM."
    ))

    @mcp.tool()
    def network_summary_tool() -> dict:
        """High-level dataset/model summary (nodes, edges, rings, fraud count)."""
        return network_summary(dm)

    @mcp.tool()
    def account_risk(account_id: str) -> dict:
        """Risk score, band, rank, degree, and label for one account."""
        return account_facts(dm, account_id)

    @mcp.tool()
    def explain_account(account_id: str) -> str:
        """Why one account is (or is not) risky, as prose."""
        return account_report(dm, account_id)

    @mcp.tool()
    def investigate(account_id: str) -> dict:
        """Full investigation: facts plus a rendered report for one account."""
        return {
            "facts": account_facts(dm, account_id),
            "report": account_report(dm, account_id),
        }

    @mcp.tool()
    def ring_details(ring_id: int) -> dict:
        """Members, internal transfers, amounts, and timing for one ring."""
        from .assistant import ring_facts

        return ring_facts(dm, ring_id)

    @mcp.tool()
    def top_risky(k: int = 10) -> list[dict]:
        """The k highest-risk accounts."""
        return top_accounts(dm, max(1, min(int(k), 500)))

    @mcp.tool()
    def counterfactual(account_id: str, k: int = 3) -> dict:
        """Fixed-model sensitivity: what if the account's k highest-risk
        transfers were gone? Re-scores the frozen model and reports the
        delta, with an honest fixed-model caveat."""
        from .assistant import counterfactual as cf_probe

        return cf_probe(dm, account_id, k=max(1, min(int(k), 10)))

    @mcp.tool()
    def case_file(account_id: str, k: int = 3) -> str:
        """A complete, shareable Markdown investigation document for one
        account (subject, ring context, top suspicious transactions,
        counterfactual probe, recommendation)."""
        from .assistant import case_document

        return case_document(dm, account_id, k=max(1, min(int(k), 10)))

    return mcp


def run_mcp(checkpoint_dir: Path) -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "the MCP server needs the 'mcp' package; install it with: "
            "pip install 'mcp>=1.0'"
        ) from e

    from .assistant import load_deployed

    dm = load_deployed(checkpoint_dir)
    build_tools(dm).run()
