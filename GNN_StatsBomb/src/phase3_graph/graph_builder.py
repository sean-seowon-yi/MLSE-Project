"""
Phase 3: Per-possession heterogeneous graph construction.

For each possession we build a ``torch_geometric.data.HeteroData`` with:

Node types
──────────
  ``"event"``  — one node per on-ball event in the possession.
                 Features: 126-D Phase 1 vector (Spatial_360 zeroed).
  ``"player"`` — one node per distinct player (actors + off-ball from 360).
                 Features: [position_idx, is_possession_team, dx, dy].

Edge types (relation triplets)
──────────────────────────────
  ("event",  "next",         "event")       — temporal: e_t → e_{t+1}
  ("event",  "prev",         "event")       — reverse temporal (optional)
  ("player", "acts_in",      "event")       — actor → event
  ("event",  "performed_by", "player")      — event → actor (reverse)

  When ``split_context_edges=False`` (default):
  ("player", "context_for",  "event")       — all off-ball 360 players → event

  When ``split_context_edges=True`` (via --split_context_edges):
  ("player", "context_for_tm",  "event")    — teammate 360 player → event
  ("player", "context_for_opp", "event")    — opponent 360 player → event
"""

import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from ..config import GraphConfig, POSITIONS, PITCH_LENGTH, PITCH_WIDTH, get_config
from ..phase2_possession.possession_builder import Possession, PossessionBuilder
from .masking import mask_spatial_360


logger = logging.getLogger(__name__)

_POS_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}
_UNKNOWN_POS_IDX = _POS_TO_IDX.get("Unknown", len(POSITIONS) - 1)


class PossessionGraphBuilder:
    """
    Converts possessions + freeze frames into heterogeneous PyG graphs.

    Parameters
    ----------
    config : GraphConfig
    """

    def __init__(self, config: GraphConfig):
        self.config = config

    # ── public API ───────────────────────────────────────────────────

    def build_graphs(
        self,
        possessions: List[Possession],
        event_features: np.ndarray,
        freeze_frames: List[Optional[List[Dict]]],
    ) -> List[HeteroData]:
        """
        Build one HeteroData per possession.

        Parameters
        ----------
        possessions : list[Possession]
        event_features : ndarray (n_events, 126)
        freeze_frames : list
            Length n_events.  Each entry is either a list of freeze-frame
            dicts or None.
        """
        graphs: List[HeteroData] = []

        for poss in tqdm(possessions, desc="Building graphs"):
            try:
                g = self._build_one(poss, event_features, freeze_frames)
                if g is not None:
                    graphs.append(g)
            except Exception as exc:
                logger.warning("Skipping possession %s: %s", poss.pos_key, exc)
                continue

        return graphs

    # ── internals ────────────────────────────────────────────────────

    def _build_one(
        self,
        poss: Possession,
        event_features: np.ndarray,
        freeze_frames: List[Optional[List[Dict]]],
    ) -> Optional[HeteroData]:
        T = len(poss.event_indices)
        if T == 0:
            return None

        # ── Event node features (mask Spatial_360) ───────────────────
        raw_feats = event_features[poss.event_indices]  # (T, 126)
        event_x = mask_spatial_360(raw_feats)

        # ── Collect player nodes ─────────────────────────────────────
        #  actor players: identified by player_id from metadata
        #  off-ball players: from freeze frames, keyed by (teammate, slot)
        player_key_to_idx: Dict = {}   # key → player-node index
        player_features: List[List[float]] = []  # [pos_idx, team_flag, dx, dy]
        player_node_pids: List[int] = []  # per-node player_id (-1 for off-ball)

        # Actor edges
        actor_src: List[int] = []  # player indices
        actor_dst: List[int] = []  # event indices (local 0..T-1)

        # Context edges — split or unified depending on config
        ctx_src: List[int] = []
        ctx_dst: List[int] = []
        ctx_tm_src: List[int] = []
        ctx_tm_dst: List[int] = []
        ctx_opp_src: List[int] = []
        ctx_opp_dst: List[int] = []

        split_ctx = self.config.split_context_edges

        for local_ev_idx in range(T):
            global_idx = poss.event_indices[local_ev_idx]
            pid = poss.player_ids[local_ev_idx]
            pos_name = poss.position_names[local_ev_idx]

            ball_x = float(raw_feats[local_ev_idx, 14]) * PITCH_LENGTH
            ball_y = float(raw_feats[local_ev_idx, 15]) * PITCH_WIDTH

            # --- actor player node ---
            actor_key = ("actor", pid)
            if actor_key not in player_key_to_idx:
                p_idx = len(player_features)
                player_key_to_idx[actor_key] = p_idx
                pos_idx = _POS_TO_IDX.get(pos_name, _UNKNOWN_POS_IDX)
                is_poss_team = 1.0 if poss.team_ids[local_ev_idx] == poss.possession_team_id else 0.0
                player_features.append([float(pos_idx), is_poss_team, 0.0, 0.0])
                player_node_pids.append(int(pid))
            p_node = player_key_to_idx[actor_key]

            # Update relative geometry to latest event's ball position
            if ball_x >= 0 and ball_y >= 0:
                player_features[p_node][2] = 0.0  # actor is at the ball
                player_features[p_node][3] = 0.0

            actor_src.append(p_node)
            actor_dst.append(local_ev_idx)

            # --- off-ball players from freeze frame ---
            actor_on_poss_team = (poss.team_ids[local_ev_idx] == poss.possession_team_id)
            ff = freeze_frames[global_idx] if global_idx < len(freeze_frames) else None
            if ff and isinstance(ff, list):
                self._process_freeze_frame(
                    ff, local_ev_idx, ball_x, ball_y,
                    poss.possession_team_id,
                    actor_on_poss_team,
                    player_key_to_idx, player_features,
                    player_node_pids,
                    ctx_src, ctx_dst,
                    ctx_tm_src, ctx_tm_dst,
                    ctx_opp_src, ctx_opp_dst,
                    split_ctx,
                )

        n_players = len(player_features)
        if n_players == 0:
            return None

        # ── Temporal edges ───────────────────────────────────────────
        next_src = list(range(T - 1))
        next_dst = list(range(1, T))

        # Time-delta edge attributes for temporal edges.
        # Normalized as min(delta / 30, 1) so that 0-30 seconds maps to [0, 1].
        timestamps = getattr(poss, "timestamps_sec", [])
        if len(timestamps) == T and T > 1:
            deltas = []
            for i in range(T - 1):
                dt = max(0.0, timestamps[i + 1] - timestamps[i])
                deltas.append(min(dt / 30.0, 1.0))
            time_delta_attr = torch.tensor(deltas, dtype=torch.float32).unsqueeze(-1)
        elif T > 1:
            time_delta_attr = torch.zeros(T - 1, 1)
        else:
            time_delta_attr = None

        # ── Assemble HeteroData ──────────────────────────────────────
        data = HeteroData()

        data["event"].x = torch.tensor(event_x, dtype=torch.float32)
        data["player"].x = torch.tensor(player_features, dtype=torch.float32)

        data["event", "next", "event"].edge_index = self._to_edge_index(next_src, next_dst)
        if time_delta_attr is not None:
            data["event", "next", "event"].edge_attr = time_delta_attr

        if self.config.use_reverse_temporal and T > 1:
            data["event", "prev", "event"].edge_index = self._to_edge_index(next_dst, next_src)
            if time_delta_attr is not None:
                data["event", "prev", "event"].edge_attr = time_delta_attr

        if actor_src:
            data["player", "acts_in", "event"].edge_index = self._to_edge_index(actor_src, actor_dst)
            data["event", "performed_by", "player"].edge_index = self._to_edge_index(actor_dst, actor_src)

        if split_ctx:
            if ctx_tm_src:
                data["player", "context_for_tm", "event"].edge_index = self._to_edge_index(ctx_tm_src, ctx_tm_dst)
            if ctx_opp_src:
                data["player", "context_for_opp", "event"].edge_index = self._to_edge_index(ctx_opp_src, ctx_opp_dst)
        else:
            if ctx_src:
                data["player", "context_for", "event"].edge_index = self._to_edge_index(ctx_src, ctx_dst)

        # ── Metadata for downstream use ──────────────────────────────
        data.pos_key = poss.pos_key
        data.match_id = poss.match_id
        data.ends_in_shot = poss.ends_in_shot
        data.ends_in_goal = poss.ends_in_goal
        data.total_xg = poss.total_xg

        data.event_player_ids = torch.tensor(poss.player_ids, dtype=torch.long)
        data.event_types = poss.event_types
        data.event_indices_global = torch.tensor(poss.event_indices, dtype=torch.long)

        # Per-player-node IDs (actors get real pid, off-ball get -1)
        data["player"].player_node_pids = torch.tensor(player_node_pids, dtype=torch.long)

        return data

    def _process_freeze_frame(
        self,
        ff: List[Dict],
        local_ev_idx: int,
        ball_x: float,
        ball_y: float,
        possession_team_id: int,
        actor_on_poss_team: bool,
        player_key_to_idx: Dict,
        player_features: List[List[float]],
        player_node_pids: List[int],
        ctx_src: List[int],
        ctx_dst: List[int],
        ctx_tm_src: List[int],
        ctx_tm_dst: List[int],
        ctx_opp_src: List[int],
        ctx_opp_dst: List[int],
        split_ctx: bool,
    ) -> None:
        """Add off-ball player nodes and context edges from a single freeze frame."""
        teammate_counter = 0
        opponent_counter = 0

        for entry in ff:
            if not isinstance(entry, dict):
                continue
            if entry.get("actor", False):
                continue

            loc = entry.get("location")
            if not loc or len(loc) < 2:
                continue

            px = max(0.0, min(float(loc[0]), PITCH_LENGTH))
            py = max(0.0, min(float(loc[1]), PITCH_WIDTH))

            is_teammate = entry.get("teammate", False)

            # Derive team_flag as "is_possession_team" (consistent with
            # actor nodes).  StatsBomb 360's ``teammate`` field is relative
            # to the event's actor, so we must XOR with whether the actor
            # is on the possession team to get the absolute flag.
            if actor_on_poss_team:
                team_flag = 1.0 if is_teammate else 0.0
            else:
                team_flag = 0.0 if is_teammate else 1.0

            # Resolve whether this context player is on the possession team
            is_on_poss_team = (team_flag == 1.0)

            if is_teammate:
                slot_key = ("tm", local_ev_idx, teammate_counter)
                teammate_counter += 1
            else:
                slot_key = ("opp", local_ev_idx, opponent_counter)
                opponent_counter += 1

            p_idx = len(player_features)
            player_key_to_idx[slot_key] = p_idx

            dx = (px - ball_x) / PITCH_LENGTH if PITCH_LENGTH > 0 else 0.0
            dy = (py - ball_y) / PITCH_WIDTH if PITCH_WIDTH > 0 else 0.0

            player_features.append([float(_UNKNOWN_POS_IDX), team_flag, dx, dy])
            player_node_pids.append(-1)

            if split_ctx:
                if is_on_poss_team:
                    ctx_tm_src.append(p_idx)
                    ctx_tm_dst.append(local_ev_idx)
                else:
                    ctx_opp_src.append(p_idx)
                    ctx_opp_dst.append(local_ev_idx)
            else:
                ctx_src.append(p_idx)
                ctx_dst.append(local_ev_idx)

    @staticmethod
    def _to_edge_index(src: List[int], dst: List[int]) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

    # ── Persistence ──────────────────────────────────────────────────

    @staticmethod
    def save(graphs: List[HeteroData], path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> List[HeteroData]:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    config = get_config()
    out_dir = Path(config.data.output_dir)

    features = np.load(out_dir / "event_features.npy")
    possessions = PossessionBuilder.load(str(out_dir / "possessions.pkl"))

    with open(out_dir / "freeze_frames.pkl", "rb") as f:
        freeze_frames = pickle.load(f)

    builder = PossessionGraphBuilder(config.graph)
    graphs = builder.build_graphs(possessions[:100], features, freeze_frames)

    print(f"Built {len(graphs)} graphs from first 100 possessions")
    if graphs:
        g = graphs[0]
        print(f"  Event nodes: {g['event'].x.shape}")
        print(f"  Player nodes: {g['player'].x.shape}")
        for edge_type in g.edge_types:
            ei = g[edge_type].edge_index
            print(f"  Edge {edge_type}: {ei.shape[1]} edges")
