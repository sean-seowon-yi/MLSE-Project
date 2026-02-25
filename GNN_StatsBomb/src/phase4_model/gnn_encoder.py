"""
Heterogeneous GNN encoder for possession graphs.

Uses ``torch_geometric.nn.HeteroConv`` to wrap per-relation ``GATv2Conv``
layers, giving each edge type its own learned attention weights.

Temporal edge types (``next``, ``prev``) accept a 1-D time-delta edge
attribute via ``edge_dim=1`` in GATv2Conv, allowing the attention
mechanism to condition on the elapsed time between consecutive events.

Architecture (per layer):
  HeteroConv({
      ("event",  "next",         "event"):  GATv2Conv(edge_dim=1),
      ("event",  "prev",         "event"):  GATv2Conv(edge_dim=1),
      ("player", "acts_in",      "event"):  GATv2Conv,
      ("event",  "performed_by", "player"): GATv2Conv,
      ("player", "context_for",  "event"):  GATv2Conv,
  })
  → per-node-type ELU + LayerNorm + Dropout

Skip connections (input projected to d) are added after the final layer
to prevent over-smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

from ..config import ModelConfig


# All relation triplets supported by this encoder.
ALL_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("event",  "next",         "event"),
    ("event",  "prev",         "event"),
    ("player", "acts_in",      "event"),
    ("event",  "performed_by", "player"),
    ("player", "context_for",  "event"),
]


class PossessionGNNEncoder(nn.Module):
    """
    2-layer heterogeneous GATv2 encoder with skip connections.

    After encoding, both ``"event"`` and ``"player"`` node types live
    in the same *d*-dimensional latent space.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.latent_dim
        h = config.hidden_dim
        heads = config.num_heads
        dropout = config.dropout

        # Skip projections (identity shortcut through the GNN)
        self.skip_event = nn.Linear(d, d)
        self.skip_player = nn.Linear(d, d)

        _TEMPORAL_RELS = {"next", "prev"}

        # Layer 1: d → hidden_dim * heads
        conv1_dict = {}
        for et in ALL_EDGE_TYPES:
            conv1_dict[et] = GATv2Conv(
                in_channels=d,
                out_channels=h,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if et[1] in _TEMPORAL_RELS else None,
            )
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")

        self.norm1_event = nn.LayerNorm(h * heads)
        self.norm1_player = nn.LayerNorm(h * heads)

        # Layer 2: hidden_dim * heads → d (single head, no concat)
        conv2_dict = {}
        for et in ALL_EDGE_TYPES:
            conv2_dict[et] = GATv2Conv(
                in_channels=h * heads,
                out_channels=d,
                heads=1,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=1 if et[1] in _TEMPORAL_RELS else None,
            )
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x_dict : {"event": (E, d), "player": (P, d)}
            Already-projected node features.
        edge_index_dict : {relation_triplet: (2, num_edges)}
        edge_attr_dict : optional
            Per-edge-type attributes, e.g. time deltas for temporal edges.

        Returns
        -------
        dict  {"event": (E, d), "player": (P, d)}
        """
        # Skip connections
        skip = {
            "event": self.skip_event(x_dict["event"]),
            "player": self.skip_player(x_dict["player"]),
        }

        # Filter to only the edge types present in this graph
        active_ei = {k: v for k, v in edge_index_dict.items() if k in ALL_EDGE_TYPES}

        ea_kw: Dict = {}
        if edge_attr_dict:
            ea_kw["edge_attr_dict"] = {
                k: v for k, v in edge_attr_dict.items() if k in active_ei
            }

        # Layer 1
        h_dict = self.conv1(x_dict, active_ei, **ea_kw)
        if "event" in h_dict:
            h_dict["event"] = self.dropout(self.norm1_event(F.elu(h_dict["event"])))
        if "player" in h_dict:
            h_dict["player"] = self.dropout(self.norm1_player(F.elu(h_dict["player"])))

        # Layer 2
        out_dict = self.conv2(h_dict, active_ei, **ea_kw)

        # Add skip
        for ntype in ("event", "player"):
            if ntype in out_dict:
                out_dict[ntype] = out_dict[ntype] + skip[ntype]

        return out_dict
