"""
Full Player Similarity Model.

Wraps the feature projections, heterogeneous GNN encoder, attention
pooling, and prediction heads into a single ``nn.Module``.

Forward pass (per batch of possession graphs):
  1. Project event features (126-D → d) and player features → d.
  2. Run heterogeneous GATv2 encoder → h_event, h_player.
  3. Pool actor embeddings **by real player_id** within the batch via
     learned AttentionPooling → z_p (one vector per unique player).
  4. For each event, use FiLM conditioning: z_p produces scale + shift
     that modulate h_event, forcing multiplicative dependence on player
     traits for action prediction.
  5. For each possession, predict outcome labels (shot / goal / xG).

The pooling is trained end-to-end: gradients from the action heads flow
through the pooled z_p into the attention scoring network and back into
the GNN.  At inference, the same pooling is used to aggregate
per-possession embeddings into a global player trait.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from typing import Dict, List, Optional, Tuple

from ..config import ModelConfig
from .projections import EventProjection, PlayerProjection
from .gnn_encoder import PossessionGNNEncoder
from .pooling import AttentionPooling


class PlayerSimilarityModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.latent_dim

        self.event_proj = EventProjection(config)
        self.player_proj = PlayerProjection(config)
        self.gnn = PossessionGNNEncoder(config)
        self.pooling = AttentionPooling(d)

        # FiLM conditioning: z_p modulates h_event via scale (gamma) and
        # shift (beta).  This forces the prediction to be multiplicatively
        # dependent on z_p — identical z_p ⇒ identical predictions.
        self.film_gamma = nn.Linear(d, d)
        self.film_beta = nn.Linear(d, d)

        # Action prediction heads now take FiLM-conditioned h_event (dim d)
        self.action_type_head = nn.Linear(d, config.n_action_types)
        self.angle_bin_head = nn.Linear(d, config.n_angle_bins)
        self.length_bin_head = nn.Linear(d, config.n_length_bins)

        # Outcome prediction head (state-only, no player trait needed)
        self.outcome_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 2),
        )

    def film_condition(
        self,
        h_event: torch.Tensor,
        z_p: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM conditioning: (1 + gamma) * h_event + beta.

        The ``1 +`` centres the scale factor at identity so that event
        information flows from the first training step (gamma ≈ 0 at init).
        """
        gamma = self.film_gamma(z_p)
        beta = self.film_beta(z_p)
        return (1 + gamma) * h_event + beta

    def forward(
        self,
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          "h_event"           : (E, d)
          "h_player"          : (P, d)
          "action_type"       : (E, n_action_types)
          "angle_bin"         : (E, n_angle_bins)
          "length_bin"        : (E, n_length_bins)
          "outcome"           : (E, 2)
          "pooled_z_p"        : (n_unique_players, d) — for contrastive loss
          "pooled_z_p_pids"   : (n_unique_players,)   — player IDs
        """
        x_event = self.event_proj(data["event"].x)
        x_player = self.player_proj(data["player"].x)

        x_dict = {"event": x_event, "player": x_player}
        edge_index_dict = {
            et: data[et].edge_index for et in data.edge_types
        }
        edge_attr_dict = self._collect_edge_attrs(data)
        out_dict = self.gnn(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        h_event = out_dict["event"]    # (E, d)
        h_player = out_dict["player"]  # (P, d)

        z_p, unique_pooled, unique_pids = self._gather_and_pool_actor_embeddings(
            data, h_player
        )

        h_conditioned = self.film_condition(h_event, z_p)
        action_type_logits = self.action_type_head(h_conditioned)
        angle_bin_logits = self.angle_bin_head(h_conditioned)
        length_bin_logits = self.length_bin_head(h_conditioned)

        outcome_logits = self.outcome_head(h_event)

        return {
            "h_event": h_event,
            "h_player": h_player,
            "action_type": action_type_logits,
            "angle_bin": angle_bin_logits,
            "length_bin": length_bin_logits,
            "outcome": outcome_logits,
            "pooled_z_p": unique_pooled,
            "pooled_z_p_pids": unique_pids,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _collect_edge_attrs(
        data: HeteroData,
    ) -> Optional[Dict[Tuple[str, str, str], torch.Tensor]]:
        """Extract per-edge-type attributes (e.g. time deltas) from HeteroData."""
        ea: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in data.edge_types:
            store = data[et]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                ea[et] = store.edge_attr
        return ea if ea else None

    # ------------------------------------------------------------------

    def _gather_and_pool_actor_embeddings(
        self,
        data: HeteroData,
        h_player: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully-vectorised per-player attention pooling.

        Returns
        -------
        z_p : (E, d)
            Pooled z_p broadcast to every event.
        unique_pooled : (n_unique, d)
            One z_p per unique player (for contrastive loss).
        unique_pids : (n_unique,)
            Corresponding player IDs.
        """
        E = data["event"].x.shape[0]
        d = h_player.shape[1]
        device = h_player.device
        z_p = torch.zeros(E, d, device=device)
        empty = torch.zeros(0, d, device=device), torch.zeros(0, dtype=torch.long, device=device)

        edge_key = ("player", "acts_in", "event")
        if edge_key not in data.edge_types:
            return z_p, *empty

        src, dst = data[edge_key].edge_index

        if not hasattr(data, "event_player_ids"):
            z_p[dst] = h_player[src]
            return z_p, *empty

        event_pids = data.event_player_ids  # (E,)

        # Deduplicate to one embedding per unique player node in the batch.
        # Within a single graph all acts_in edges from the same player share
        # the same player node, so collapsing by src gives one h_player per
        # possession per player — matching inference behaviour and avoiding
        # event-count bias in the attention weights.
        unique_src, inv_edge = torch.unique(src, return_inverse=True)
        dedup_embs = h_player[unique_src]                 # (n_unique_nodes, d)
        dedup_pids = data["player"].player_node_pids[unique_src]  # (n_unique_nodes,)

        # Group unique nodes by player_id for attention pooling
        unique_pids_tensor, group_idx = torch.unique(dedup_pids, return_inverse=True)

        # Attention scores from the pooling scoring network
        scores = self.pooling.score_net(dedup_embs).squeeze(-1)

        # Numerically-stable scatter-softmax per player group
        max_per_group = scatter(scores, group_idx, dim=0, reduce="max")
        scores_stable = scores - max_per_group[group_idx]
        exp_scores = torch.exp(scores_stable)
        sum_exp = scatter(exp_scores, group_idx, dim=0, reduce="sum")
        alpha = exp_scores / sum_exp[group_idx]

        # Attention-weighted sum per player group
        weighted = alpha.unsqueeze(-1) * dedup_embs
        pooled = scatter(weighted, group_idx, dim=0, reduce="sum")  # (n_unique, d)

        # Broadcast back to event dimension via edge → dedup node → player group
        dedup_group = group_idx[inv_edge]                # (n_edges,)
        z_p[dst] = pooled[dedup_group]

        return z_p, pooled, unique_pids_tensor

    # ------------------------------------------------------------------

    def encode_possession(
        self,
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """Run projection + GNN only (no heads). Works for single or batched graphs."""
        x_event = self.event_proj(data["event"].x)
        x_player = self.player_proj(data["player"].x)
        x_dict = {"event": x_event, "player": x_player}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        edge_attr_dict = self._collect_edge_attrs(data)
        return self.gnn(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
