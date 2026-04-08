"""
model/gnn.py — PassLocationGNN architecture.

Predicts pass end location as (x_norm, y_norm) in [0, 1]^2.
Multiply output by (PITCH_LEN, PITCH_WID) for pitch coordinates.

Architecture
------------
  input_proj      NODE_DIM=6  →  hidden=64
  player_embed    vocab_size  →  embed_dim=16  →  hidden  (added to actor node)
  GATv2Conv × 3   hidden, heads=4, edge_dim=5   (residual + LayerNorm)
  GlobalMeanPool  →  (B, hidden)
  global_proj     GLOBAL_DIM=10  →  hidden//2
  concat          →  (B, hidden + hidden//2)
  MLP             →  64  →  32  →  2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

from config import (
    NODE_DIM, EDGE_DIM, GLOBAL_DIM, OUT_DIM,
    EMBED_DIM, HIDDEN, HEADS, DROPOUT,
)


class PassLocationGNN(nn.Module):
    """
    GNN that predicts pass end location from a freeze-frame graph.

    Parameters
    ----------
    node_dim, edge_dim, global_dim, out_dim : feature dimensions
    vocab_size  : number of known players + 1 (index 0 = <UNK>)
    embed_dim   : player identity embedding size
    hidden      : internal hidden width (must be divisible by heads)
    heads       : number of GATv2 attention heads
    dropout     : dropout rate applied after each GATv2 block
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        global_dim: int = GLOBAL_DIM,
        out_dim: int = OUT_DIM,
        vocab_size: int = 1,
        embed_dim: int = EMBED_DIM,
        hidden: int = HIDDEN,
        heads: int = HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden = hidden

        # Player identity embedding
        self.player_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_proj = nn.Linear(embed_dim, hidden, bias=False)

        # Node encoder
        self.input_proj = nn.Linear(node_dim, hidden)

        self.conv1 = GATv2Conv(
            hidden, hidden // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )
        self.conv2 = GATv2Conv(
            hidden, hidden // heads, heads=heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )
        self.conv3 = GATv2Conv(
            hidden, hidden, heads=1,
            edge_dim=edge_dim, dropout=dropout, concat=False,
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Global context projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden // 2),
            nn.ReLU(),
        )

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden + hidden // 2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Data / Batch with fields x, edge_index, edge_attr, u,
               actor_idx, batch

        Returns
        -------
        (B, 2) tensor  [x_norm, y_norm] — no final activation
        """
        h, h_graph = self._encode(
            data.x, data.edge_index, data.edge_attr,
            data.batch, data.actor_idx,
        )
        u = self.global_proj(data.u.squeeze(1))               # (B, hidden//2)
        return self.regressor(torch.cat([h_graph, u], dim=-1)) # (B, 2)

    # ── Encoding (exposed for reuse in analysis code) ──────────────────────────

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        actor_idx: torch.Tensor,
    ):
        """
        Project node features, inject player embedding, run GATv2 stack.

        Returns
        -------
        h        : (N_total, hidden)  — per-node hidden states
        h_graph  : (B, hidden)        — batch-pooled graph embeddings
        """
        h = F.relu(self.input_proj(x))

        # Add player embedding to the actor node in each graph
        actor_mask = x[:, 2] > 0.5
        if actor_mask.any():
            actor_node_ids = actor_mask.nonzero(as_tuple=True)[0]
            graph_ids = batch[actor_node_ids]
            actor_vocab_idxs = actor_idx.view(-1)[graph_ids]
            embed = self.embed_proj(self.player_embed(actor_vocab_idxs))
            delta = torch.zeros_like(h).index_add(0, actor_node_ids, embed)
            h = h + delta

        # GATv2 stack with residual connections
        h = self.norm1(F.relu(self.conv1(h, edge_index, edge_attr)) + h)
        h = self.dropout(h)
        h = self.norm2(F.relu(self.conv2(h, edge_index, edge_attr)) + h)
        h = self.dropout(h)
        h = self.norm3(F.relu(self.conv3(h, edge_index, edge_attr)) + h)

        h_graph = global_mean_pool(h, batch)   # (B, hidden)
        return h, h_graph


def build_model(vocab_size: int, **kwargs) -> PassLocationGNN:
    """Convenience factory that forwards all config defaults."""
    return PassLocationGNN(vocab_size=vocab_size, **kwargs)


def count_parameters(model: PassLocationGNN) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.player_embed.parameters())
    return {"total": total, "embedding": embed, "non_embedding": total - embed}
