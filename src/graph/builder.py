"""
graph/builder.py — Builds PyG Data objects from StatsBomb freeze-frames.

Each completed pass with a 360 frame becomes one PyG Data object.
Camera-boundary nodes (from the visible_area polygon) are appended after
player nodes so GATv2 attention can incorporate spatial camera coverage.

Graph layout
------------
Nodes       : [player_0 ... player_N | cam_0 ... cam_7]
Edges       : fully-connected (no self-loops), directed
Node features (data.x)   : (N+8, 6)
Edge features (data.edge_attr) : (E, 5)
Global (data.u)          : (1, 10)
Label (data.y)           : (2,)  [x_norm, y_norm]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import src.graph.visible_area as va
from config import (
    N_CAM_PTS,
    NODE_DIM, EDGE_DIM, GLOBAL_DIM, OUT_DIM,
    PITCH_LEN, PITCH_WID, PITCH_DIAG,
    NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, LABEL_NAMES,
)


# ── Pre-grouping ───────────────────────────────────────────────────────────────

def group_frames(frames: pd.DataFrame) -> Dict:
    """
    Pre-group the frames table by event_id for O(1) lookup.

    Returns
    -------
    dict  event_id -> {'arr': (N, 5) ndarray, 'vis': visible_area | None}
        arr columns: player_x, player_y, actor, teammate, keeper
    """
    t0 = time.time()
    groups: Dict = {}
    for eid, grp in frames.groupby("id"):
        vis = (
            grp["visible_area"].dropna().iloc[0]
            if "visible_area" in grp.columns and grp["visible_area"].notna().any()
            else None
        )
        groups[eid] = {
            "arr": grp[["player_x", "player_y", "actor", "teammate", "keeper"]].values,
            "vis": vis,
        }
    print(f"Pre-grouped {len(groups):,} frame groups in {time.time() - t0:.2f}s")
    return groups


# ── Dataset construction ───────────────────────────────────────────────────────

def build_dataset(
    pass_events: pd.DataFrame,
    frame_groups: Dict,
    player_vocab: Dict,
    score_lookup: Dict,
    match_meta: Dict,
) -> List[Data]:
    """
    Build a PyG Data object for every valid pass event.

    Returns a list of Data objects (skips events that fail validation).
    """
    need_cols = [
        "id", "match_id", "player_id", "team",
        "event_x", "event_y", "pass_end_x", "pass_end_y",
        "period", "minute", "second",
    ]
    need_cols = [c for c in need_cols if c in pass_events.columns]
    records = pass_events[need_cols].to_dict("records")

    t0 = time.time()
    graphs = []
    for record in records:
        g = _process_record(record, frame_groups, player_vocab, score_lookup, match_meta)
        if g is not None:
            graphs.append(g)

    elapsed = time.time() - t0
    print(
        f"Built {len(graphs):,} graphs in {elapsed:.2f}s "
        f"({len(graphs) / (elapsed + 1e-6):.0f} graphs/sec)"
    )
    return graphs


# ── Single-graph builder ───────────────────────────────────────────────────────

def build_pass_graph(
    event_row: dict,
    pos: np.ndarray,
    is_actor: np.ndarray,
    is_teammate: np.ndarray,
    is_keeper: np.ndarray,
    visible_area=None,
    score_state: tuple = (0, 0),
    actor_idx: int = 0,
    match_meta: Dict = {},
    label: bool = True,
) -> Optional[Data]:
    """
    Construct a single PyG Data object for one freeze-frame pass event.

    Parameters
    ----------
    event_row    : dict-like row from the events DataFrame
    pos          : (N, 2) player positions in yards
    is_actor     : (N,) bool — True for the passer
    is_teammate  : (N,) bool — True for same-team players
    is_keeper    : (N,) bool — True for goalkeepers
    visible_area : StatsBomb visible_area polygon (list of [x, y] pairs)
    score_state  : (home_goals, away_goals) before this event
    actor_idx    : passer's embedding index (0 = UNK)
    match_meta   : dict match_id -> {home_team, away_team}
    label        : if False, skip the y / end_loc_raw fields

    Returns None if the graph would be degenerate (too few players, no
    teammates, missing end location).
    """
    N = len(pos)
    if N < 2:
        return None
    if label and is_teammate.sum() == 0:
        return None

    bx = float(event_row.get("event_x") or PITCH_LEN / 2)
    by = float(event_row.get("event_y") or PITCH_WID / 2)

    if label:
        end_x = event_row.get("pass_end_x")
        end_y = event_row.get("pass_end_y")
        if pd.isna(end_x) or pd.isna(end_y):
            return None

    # ── Camera-boundary nodes ──────────────────────────────────────────────────
    cam_pts = va.resample_boundary_nodes(visible_area)
    if cam_pts is None:
        cam_pts = va.fallback_boundary_nodes()
    N_cam = len(cam_pts)
    N_all = N + N_cam
    pos_all = np.vstack([pos, cam_pts]).astype(np.float32)

    # ── Node features ──────────────────────────────────────────────────────────
    player_feats = np.stack([
        pos[:, 0] / PITCH_LEN,
        pos[:, 1] / PITCH_WID,
        is_actor.astype(np.float32),
        is_teammate.astype(np.float32),
        is_keeper.astype(np.float32),
        np.zeros(N, dtype=np.float32),
    ], axis=1)
    cam_feats = np.stack([
        cam_pts[:, 0] / PITCH_LEN,
        cam_pts[:, 1] / PITCH_WID,
        np.zeros(N_cam, dtype=np.float32),
        np.zeros(N_cam, dtype=np.float32),
        np.zeros(N_cam, dtype=np.float32),
        np.ones(N_cam, dtype=np.float32),
    ], axis=1)
    node_feats = np.vstack([player_feats, cam_feats]).astype(np.float32)

    # ── Fully-connected edges (no self-loops) ──────────────────────────────────
    ii, jj = np.meshgrid(np.arange(N_all), np.arange(N_all), indexing="ij")
    mask = ~np.eye(N_all, dtype=bool)
    src, dst = ii[mask], jj[mask]
    edge_index = np.stack([src, dst], axis=0)

    dx_e = pos_all[dst, 0] - pos_all[src, 0]
    dy_e = pos_all[dst, 1] - pos_all[src, 1]
    dist = np.sqrt(dx_e ** 2 + dy_e ** 2) + 1e-6
    team_flag = np.concatenate([
        is_teammate.astype(np.float32),
        np.zeros(N_cam, dtype=np.float32),
    ])
    same_team = (team_flag[src] == team_flag[dst]).astype(np.float32)
    edge_feats = np.stack([
        dist / PITCH_DIAG,
        dx_e / PITCH_LEN,
        dy_e / PITCH_WID,
        same_team,
        np.arctan2(dy_e, dx_e) / np.pi,
    ], axis=1).astype(np.float32)

    # ── Global features ────────────────────────────────────────────────────────
    global_feats = _build_global_features(
        event_row, visible_area, score_state, match_meta
    )

    # ── Assemble Data object ───────────────────────────────────────────────────
    kwargs = dict(
        x=torch.from_numpy(node_feats),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_feats),
        pos=torch.tensor(pos_all, dtype=torch.float),
        u=torch.from_numpy(global_feats).unsqueeze(0),
        passer_pos=torch.tensor([bx, by], dtype=torch.float),
        actor_idx=torch.tensor([actor_idx], dtype=torch.long),
    )
    if label:
        kwargs["y"] = torch.tensor(
            [float(end_x) / PITCH_LEN, float(end_y) / PITCH_WID],
            dtype=torch.float,
        )
        kwargs["end_loc_raw"] = torch.tensor(
            [float(end_x), float(end_y)], dtype=torch.float
        )
    return Data(**kwargs)


# ── Private helpers ────────────────────────────────────────────────────────────

def _process_record(record, frame_groups, player_vocab, score_lookup, match_meta):
    """Process one pass record dict -> Data | None."""
    eid = record["id"]
    fg = frame_groups.get(eid)
    if fg is None or len(fg["arr"]) < 2:
        return None

    arr = fg["arr"]
    pos = arr[:, :2].astype(np.float32)
    is_actor = arr[:, 2].astype(bool)
    is_teammate = arr[:, 3].astype(bool)
    is_keeper = arr[:, 4].astype(bool)

    if is_teammate.sum() == 0:
        return None

    actor_idx = player_vocab.get(record.get("player_id"), 0)
    state = score_lookup.get(eid, (0, 0))

    return build_pass_graph(
        record, pos, is_actor, is_teammate, is_keeper,
        visible_area=fg["vis"],
        score_state=state,
        actor_idx=actor_idx,
        match_meta=match_meta,
        label=True,
    )


def _build_global_features(event_row, visible_area, score_state, match_meta) -> np.ndarray:
    """Assemble the 10-d global feature vector."""
    vis = va.global_features(visible_area)

    period = float(event_row.get("period") or 1)
    minute = float(event_row.get("minute") or 0)
    second = float(event_row.get("second") or 0)
    period_n = period / 5.0
    time_n = float(np.clip((minute * 60 + second) / 5400.0, 0.0, 1.0))

    home_g, away_g = score_state
    meta = match_meta.get(event_row.get("match_id"), {})
    passer_team = event_row.get("team")
    passer_score = home_g if passer_team == meta.get("home_team") else away_g
    opp_score = away_g if passer_team == meta.get("home_team") else home_g
    diff = passer_score - opp_score

    return np.array([
        vis[0], vis[1], vis[2], vis[3],
        period_n, time_n,
        float(np.clip(diff, -3, 3) / 3.0),
        float(diff == 0),
        float(diff > 0),
        float(diff < 0),
    ], dtype=np.float32)
