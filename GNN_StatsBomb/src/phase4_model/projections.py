"""
Feature projection modules for event and player nodes.

Before feeding into the GNN, the raw features of each node type must be
projected into a common latent space of dimension *d*.

- **EventProjection**: 122-D sparse/mixed vector → d via MLP + LayerNorm.
- **PlayerProjection**: (position_idx, team_flag, dx, dy) → d via
  learned position embedding + MLP + LayerNorm.
"""

import torch
import torch.nn as nn

from ..config import ModelConfig


class EventProjection(nn.Module):
    """Project 122-D Phase 1 event features to latent dim *d*."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.latent_dim
        self.net = nn.Sequential(
            nn.Linear(122, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, 122) → (*, d)"""
        return self.net(x)


class PlayerProjection(nn.Module):
    """
    Project raw player-node features to latent dim *d*.

    Raw features per player node (from Phase 3 graph builder):
      [position_idx (float), is_possession_team (0/1), dx, dy]

    The integer position_idx is used as a lookup into a learned embedding
    table; the result is concatenated with the continuous features and
    passed through a small MLP.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.latent_dim
        d_pos = config.position_embed_dim
        n_pos = config.n_positions

        self.pos_embed = nn.Embedding(n_pos, d_pos)

        # continuous features: is_possession_team (1) + dx (1) + dy (1)
        self.net = nn.Sequential(
            nn.Linear(d_pos + 3, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 4) with columns [position_idx, team_flag, dx, dy].
        Returns (N, d).
        """
        pos_idx = x[:, 0].long().clamp(0, self.pos_embed.num_embeddings - 1)
        pos_emb = self.pos_embed(pos_idx)            # (N, d_pos)
        cont = x[:, 1:]                               # (N, 3)
        h = torch.cat([pos_emb, cont], dim=-1)        # (N, d_pos + 3)
        return self.net(h)
