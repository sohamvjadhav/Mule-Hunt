"""GNN architectures for node- and transaction-level fraud classification."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, JumpingKnowledge, SAGEConv


class _LayerStack(nn.Module):
    """Stack of message-passing layers with Jumping Knowledge aggregation.

    JK concatenates every layer's output before the classifier, so a node's
    prediction sees 1..num_layers hops instead of only the deepest layer
    (which over-smooths on deep stacks). out_dim = hidden * num_layers.
    """

    def __init__(
        self,
        conv_factory,
        in_dim: int,
        hidden: int,
        num_layers: int,
        dropout: float = 0.5,
        act: nn.Module | None = None,
        jk: str = "cat",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList(
            conv_factory(in_dim if i == 0 else hidden, hidden, i) for i in range(num_layers)
        )
        self.jk = JumpingKnowledge(jk, hidden, num_layers) if num_layers > 1 else None
        self.act = act if act is not None else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * num_layers if (jk == "cat" and num_layers > 1) else hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        xs = []
        for i, conv in enumerate(self.convs):
            x = self.act(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = self.drop(x)
            xs.append(x)
        if self.jk is not None:
            return self.jk(xs)
        return xs[-1]


class _GATBlock(nn.Module):
    """One GATv2 layer that always emits `hidden` channels, so multi-head
    attention on the first layer can coexist with uniform JK aggregation."""

    def __init__(self, in_dim: int, hidden: int, heads: int, dropout: float = 0.5):
        super().__init__()
        self.conv = GATv2Conv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.proj = nn.Linear(hidden * heads, hidden) if heads > 1 else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, edge_index)
        if self.proj is not None:
            x = self.proj(x)
        return x


class _EdgeHead(nn.Module):
    """Transaction-level head on shared node embeddings: an MLP over
    concat(embedding[src], embedding[dst], edge features)."""

    def __init__(self, node_dim: int, hidden: int, edge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = torch.cat([emb[edge_index[0]], emb[edge_index[1]], edge_attr], dim=-1)
        return self.mlp(h).squeeze(-1)


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.5,
        num_layers: int = 2,
        jk: str = "cat",
        edge_attr_dim: int | None = None,
    ):
        super().__init__()
        self.layers = _LayerStack(
            lambda i, o, _: GCNConv(i, o),
            in_dim, hidden, num_layers, dropout, nn.ReLU(), jk,
        )
        self.cls = nn.Linear(self.layers.out_dim, 1)
        self.edge_head = _EdgeHead(self.layers.out_dim, hidden, edge_attr_dim) if edge_attr_dim else None

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.layers(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.cls(self.embed(x, edge_index)).squeeze(-1)

    def edge_forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_head(self.embed(x, edge_index), edge_index, edge_attr)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.5,
        num_layers: int = 2,
        jk: str = "cat",
        edge_attr_dim: int | None = None,
    ):
        super().__init__()
        self.layers = _LayerStack(
            lambda i, o, _: SAGEConv(i, o, aggr="mean"),
            in_dim, hidden, num_layers, dropout, nn.ReLU(), jk,
        )
        self.cls = nn.Linear(self.layers.out_dim, 1)
        self.edge_head = _EdgeHead(self.layers.out_dim, hidden, edge_attr_dim) if edge_attr_dim else None

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.layers(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.cls(self.embed(x, edge_index)).squeeze(-1)

    def edge_forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_head(self.embed(x, edge_index), edge_index, edge_attr)


class GATv2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.5,
        num_layers: int = 2,
        jk: str = "cat",
        edge_attr_dim: int | None = None,
        heads: int = 4,
    ):
        super().__init__()
        self.layers = _LayerStack(
            lambda i, o, li: _GATBlock(i, o, heads if li == 0 else 1, dropout),
            in_dim, hidden, num_layers, dropout, nn.ELU(), jk,
        )
        self.cls = nn.Linear(self.layers.out_dim, 1)
        self.edge_head = _EdgeHead(self.layers.out_dim, hidden, edge_attr_dim) if edge_attr_dim else None

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.layers(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.cls(self.embed(x, edge_index)).squeeze(-1)

    def edge_forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.edge_head(self.embed(x, edge_index), edge_index, edge_attr)


MODELS = {"gcn": GCN, "sage": GraphSAGE, "gat": GATv2}


def build_model(
    name: str,
    in_dim: int,
    hidden: int = 64,
    dropout: float = 0.5,
    num_layers: int = 2,
    jk: str = "cat",
    edge_attr_dim: int | None = None,
) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](in_dim, hidden, dropout, num_layers, jk, edge_attr_dim)
