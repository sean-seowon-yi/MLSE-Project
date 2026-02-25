"""
Phase 6-B: Player similarity search.

Given the global trait embeddings ``z_p`` from EmbeddingGenerator, this
module provides:
  - Cosine or Euclidean similarity matrix computation.
  - Top-k nearest-neighbour retrieval with optional filters
    (position group, minimum possessions, league, etc.).
  - Inference-time mirroring for cross-sided queries.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..config import InferenceConfig, POSITION_GROUPS


# Reverse lookup: position_name → group
_POS_TO_GROUP: Dict[str, str] = {}
for group, positions in POSITION_GROUPS.items():
    for pos in positions:
        _POS_TO_GROUP[pos] = group


class SimilaritySearcher:
    """
    Finds the most similar players given trait embeddings.

    Parameters
    ----------
    config : InferenceConfig
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

    def compute_similarity_matrix(
        self,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Pairwise similarity.

        Parameters
        ----------
        Z : (n_players, d)

        Returns
        -------
        sim : (n_players, n_players)
        """
        if self.config.similarity_metric == "cosine":
            return cosine_similarity(Z)
        elif self.config.similarity_metric == "euclidean":
            from scipy.spatial.distance import cdist
            dist = cdist(Z, Z, metric="euclidean")
            return 1.0 / (1.0 + dist)
        else:
            return cosine_similarity(Z)

    def find_similar_players(
        self,
        query_player_id: int,
        Z: np.ndarray,
        player_info: pd.DataFrame,
        similarity_matrix: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        position_group: Optional[str] = None,
        min_possessions: Optional[int] = None,
        exclude_same_team: bool = False,
    ) -> pd.DataFrame:
        """
        Return top-k players most similar to *query_player_id*.

        Parameters
        ----------
        query_player_id : int
        Z : (n_players, d)
        player_info : DataFrame with columns player_id, player_name,
                      position_name, n_possessions.
        similarity_matrix : pre-computed (optional; computed if None).
        top_k : override config.top_k.
        position_group : filter to a specific group ("Defender", etc.).
        min_possessions : override config.min_samples_per_player.
        exclude_same_team : if True exclude same-team players.

        Returns
        -------
        DataFrame with columns: player_id, player_name, position_name,
                                similarity, n_possessions.
        """
        if top_k is None:
            top_k = self.config.top_k

        pids = player_info["player_id"].values
        target_idx = np.where(pids == query_player_id)[0]
        if len(target_idx) == 0:
            raise ValueError(f"Player {query_player_id} not in embedding index.")
        target_idx = target_idx[0]

        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix(Z)

        sims = similarity_matrix[target_idx].copy()
        sims[target_idx] = -np.inf  # exclude self

        # Apply filters
        mask = np.ones(len(pids), dtype=bool)
        mask[target_idx] = False

        if position_group:
            pos_names = player_info["position_name"].values
            mask &= np.array([
                _POS_TO_GROUP.get(p, "") == position_group for p in pos_names
            ])

        if min_possessions is not None:
            mask &= player_info["n_possessions"].values >= min_possessions

        if exclude_same_team and "team_id" in player_info.columns:
            target_team = player_info.iloc[target_idx].get("team_id")
            if target_team is not None:
                mask &= player_info["team_id"].values != target_team

        valid = np.where(mask)[0]
        if len(valid) == 0:
            return pd.DataFrame()

        valid_sims = sims[valid]
        top_indices = valid[np.argsort(valid_sims)[::-1][:top_k]]

        results = []
        for idx in top_indices:
            row = player_info.iloc[idx]
            results.append({
                "player_id": int(row["player_id"]),
                "player_name": row.get("player_name", ""),
                "position_name": row.get("position_name", ""),
                "similarity": float(sims[idx]),
                "n_possessions": int(row.get("n_possessions", 0)),
            })

        return pd.DataFrame(results)

    @staticmethod
    def mirror_query_embedding(z: np.ndarray) -> np.ndarray:
        """
        Flip a query player's embedding for cross-sided similarity.

        This is a placeholder — the exact mirroring operation depends on
        how the embedding space encodes sidedness.  A practical approach
        is to re-run the model with y-flipped event coordinates and
        left/right-swapped position labels, then pool to z_p.
        """
        return z.copy()
