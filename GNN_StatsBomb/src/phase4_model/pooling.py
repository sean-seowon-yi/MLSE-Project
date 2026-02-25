"""
Attention-weighted pooling for aggregating per-possession player
embeddings into a single global player trait embedding ``z_p``.

Instead of a simple mean (which over-emphasises routine recycling),
a learned scoring function lets the model up-weight high-leverage
possessions (shots, line-breaking plays) when computing the global
player embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class AttentionPooling(nn.Module):
    """
    Learns ``score(h) → α`` then computes ``z = Σ α_i h_i``.

    Parameters
    ----------
    d : int
        Embedding dimension.
    """

    def __init__(self, d: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (total, d)
            Concatenated per-possession embeddings for *one* player,
            stacked across all possessions they appear in.
        lengths : (n_players,)
            Number of possession embeddings per player so we can
            split ``embeddings`` into per-player groups.

        Returns
        -------
        z : (n_players, d)
            One global trait embedding per player.
        """
        scores = self.score_net(embeddings).squeeze(-1)  # (total,)

        splits = torch.split(embeddings, lengths.tolist(), dim=0)
        score_splits = torch.split(scores, lengths.tolist(), dim=0)

        z_list: List[torch.Tensor] = []
        for emb_i, sc_i in zip(splits, score_splits):
            alpha = F.softmax(sc_i, dim=0).unsqueeze(-1)  # (k, 1)
            z_i = (alpha * emb_i).sum(dim=0)               # (d,)
            z_list.append(z_i)

        return torch.stack(z_list, dim=0)  # (n_players, d)

    def pool_single(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pool a single player's embeddings (K, d) → (d,)."""
        scores = self.score_net(embeddings).squeeze(-1)    # (K,)
        alpha = F.softmax(scores, dim=0).unsqueeze(-1)     # (K, 1)
        return (alpha * embeddings).sum(dim=0)              # (d,)
