"""MCP (stdio) server exposing the deployed risk graph to coding agents.

Any MCP-capable agent (Claude Code, Codex, opencode, ...) can attach to
``upifraud mcp`` and investigate the fraud graph with grounded tools:
account risk, explanations, full investigation reports, ring details, top
accounts, and network summaries. Everything is deterministic — no external
model calls.
"""

from __future__ import annotations

from pathlib import Path

from .assistant import (
    account_facts,
    account_report,
    network_summary,
    top_accounts,
)


def run_mcp(checkpoint_dir: Path) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "the MCP server needs the 'mcp' package; install it with: "
            "pip install 'mcp>=1.0'"
        ) from e

    from .assistant import DeployedModel, load_deployed

    dm: DeployedModel = load_deployed(checkpoint_dir)
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

    mcp.run()
