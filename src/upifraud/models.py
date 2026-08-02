"""GNN architectures for node-level fraud classification."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.cls = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        return self.cls(x).squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.cls = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        return self.cls(x).squeeze(-1)


MODELS = {"gcn": GCN, "sage": GraphSAGE}


def build_model(name: str, in_dim: int, hidden: int = 64, dropout: float = 0.5) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](in_dim, hidden, dropout)
