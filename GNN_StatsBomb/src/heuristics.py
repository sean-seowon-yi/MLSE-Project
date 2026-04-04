"""
Simple heuristic methods for player similarity — no GNN, no training.

Each heuristic produces the same output format as Phase 6 inference:
  - ``player_embeddings.npy``  (n_players, D)
  - ``player_info.parquet``    (player_id, player_name, position_name, n_possessions, team_id, team_name)

Three heuristics are implemented:

1. **mean_features** — Average future-masked 126-D event features per player.
2. **action_profile** — Per-player histograms of action type, angle, length,
   pitch zone, and body part distributions.
3. **fifa_attributes** — FIFA sub-attribute vectors for players with a FIFA match.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    EVENT_TYPES,
    InferenceConfig,
)
from .phase3_graph.masking import mask_future_info


def _clean_name(name: str) -> str:
    out = []
    for ch in name:
        cp = ord(ch)
        if (cp <= 0x024F
                or 0x1E00 <= cp <= 0x1EFF
                or 0x2000 <= cp <= 0x206F
                or 0x0300 <= cp <= 0x036F):
            out.append(ch)
    return "".join(out).strip()


def _build_player_info(
    player_ids: List[int],
    metadata: pd.DataFrame,
    n_events_per_player: Dict[int, int],
) -> pd.DataFrame:
    rows = []
    for pid in player_ids:
        name = ""
        pos = "Unknown"
        team_id = None
        team_name = ""
        if "player_id" in metadata.columns:
            prows = metadata[metadata["player_id"] == pid]
            if len(prows) > 0:
                name = _clean_name(str(prows.iloc[0].get("player_name", "")))
                pos_mode = prows["position_name"].mode()
                pos = str(pos_mode.iloc[0]) if len(pos_mode) > 0 else "Unknown"
                if "team_id" in prows.columns:
                    tm = prows["team_id"].mode()
                    if len(tm) > 0 and pd.notna(tm.iloc[0]):
                        team_id = int(tm.iloc[0])
                if "team_name" in prows.columns:
                    tn = prows["team_name"].mode()
                    if len(tn) > 0 and pd.notna(tn.iloc[0]):
                        team_name = str(tn.iloc[0])
        rows.append({
            "player_id": pid,
            "player_name": name,
            "position_name": pos,
            "n_possessions": n_events_per_player.get(pid, 0),
            "team_id": team_id,
            "team_name": team_name,
        })
    return pd.DataFrame(rows)


def _save_heuristic(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "player_embeddings.npy"), Z)
    player_info.to_parquet(str(output_dir / "player_info.parquet"), index=False)
    print(f"  Saved {Z.shape[0]} player embeddings ({Z.shape[1]}-D) to {output_dir}/")


# ── Split-half embed_fn adapters ─────────────────────────────────────────
# Used by evaluate_split_half_stability.  Each returns a 1-D embedding.

def embed_fn_mean_features(feature_rows: np.ndarray, event_types: List[str]) -> np.ndarray:
    """Average future-masked features."""
    masked = mask_future_info(feature_rows)
    return masked.mean(axis=0)


def embed_fn_action_profile(feature_rows: np.ndarray, event_types: List[str]) -> np.ndarray:
    """Action distribution profile."""
    return _compute_action_profile(feature_rows, event_types)


# ── Heuristic 1: Mean event features ────────────────────────────────────

def generate_mean_features(
    event_features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    min_events: int = 50,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Average future-masked 126-D event features per player."""
    print("\n  [mean_features] Masking future info ...")
    masked = mask_future_info(event_features)

    print("  [mean_features] Grouping by player ...")
    pid_col = metadata["player_id"].values

    accum: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(pid_col):
        if pid >= 0:
            accum[int(pid)].append(i)

    player_ids = []
    embeddings = []
    n_events: Dict[int, int] = {}
    for pid, indices in accum.items():
        if len(indices) < min_events:
            continue
        player_ids.append(pid)
        embeddings.append(masked[indices].mean(axis=0))
        n_events[pid] = len(indices)

    Z = np.stack(embeddings, axis=0)  # (n_players, 126)
    player_info = _build_player_info(player_ids, metadata, n_events)

    print(f"  [mean_features] {Z.shape[0]} players, {Z.shape[1]}-D")
    _save_heuristic(Z, player_info, output_dir)
    return Z, player_info


# ── Heuristic 2: Action profile ─────────────────────────────────────────

_EVENT_TYPE_TO_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
_N_ACTION_TYPES = len(EVENT_TYPES)  # 14
_N_ANGLE_BINS = 9   # 8 sectors + no-angle
_N_LENGTH_BINS = 5
_N_PITCH_ZONES = 9  # 3x3
_N_BODY_PARTS = 7

_LENGTH_THRESHOLDS = [0.05, 0.15, 0.30, 0.50]

_BODY_PART_START = 58
_BODY_PART_END = 65
_ZONE_START = 104
_ZONE_END = 113


def _compute_action_profile(feat_rows: np.ndarray, event_types: List[str]) -> np.ndarray:
    """Build a histogram profile vector for one player's events."""
    n = len(event_types)
    action_hist = np.zeros(_N_ACTION_TYPES, dtype=np.float64)
    angle_hist = np.zeros(_N_ANGLE_BINS, dtype=np.float64)
    length_hist = np.zeros(_N_LENGTH_BINS, dtype=np.float64)
    zone_hist = np.zeros(_N_PITCH_ZONES, dtype=np.float64)
    body_hist = np.zeros(_N_BODY_PARTS, dtype=np.float64)

    for i in range(n):
        etype = event_types[i]
        idx = _EVENT_TYPE_TO_IDX.get(etype, 0)
        action_hist[idx] += 1

        # Angle bin from delta (indices 19, 20) and has_end flag (index 18)
        dx = float(feat_rows[i, 19])
        dy = float(feat_rows[i, 20])
        has_end = float(feat_rows[i, 18])

        if has_end > 0.5 and (abs(dx) > 1e-6 or abs(dy) > 1e-6):
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            b = min(int(angle / (2 * math.pi) * 8), 7)
            angle_hist[b] += 1

            dist = math.sqrt(dx * dx + dy * dy)
            lb = len(_LENGTH_THRESHOLDS)
            for j, thresh in enumerate(_LENGTH_THRESHOLDS):
                if dist <= thresh:
                    lb = j
                    break
            length_hist[lb] += 1
        else:
            angle_hist[8] += 1
            length_hist[0] += 1

        # Pitch zone: argmax of one-hot at indices 104-112
        zone_vec = feat_rows[i, _ZONE_START:_ZONE_END]
        zone_hist[int(np.argmax(zone_vec))] += 1

        # Body part: argmax of one-hot at indices 58-64
        bp_vec = feat_rows[i, _BODY_PART_START:_BODY_PART_END]
        if bp_vec.sum() > 0:
            body_hist[int(np.argmax(bp_vec))] += 1

    # Normalize each sub-histogram to sum to 1
    for h in [action_hist, angle_hist, length_hist, zone_hist, body_hist]:
        s = h.sum()
        if s > 0:
            h /= s

    return np.concatenate([action_hist, angle_hist, length_hist, zone_hist, body_hist]).astype(np.float32)


def generate_action_profile(
    event_features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    min_events: int = 50,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Per-player behavioral distribution vectors."""
    print("\n  [action_profile] Building per-player profiles ...")
    pid_col = metadata["player_id"].values
    etype_col = metadata["event_type"].values if "event_type" in metadata.columns else None

    accum: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(pid_col):
        if pid >= 0:
            accum[int(pid)].append(i)

    player_ids = []
    embeddings = []
    n_events: Dict[int, int] = {}
    for pid, indices in accum.items():
        if len(indices) < min_events:
            continue
        etypes = [str(etype_col[i]) for i in indices] if etype_col is not None else ["Unknown"] * len(indices)
        profile = _compute_action_profile(event_features[indices], etypes)
        player_ids.append(pid)
        embeddings.append(profile)
        n_events[pid] = len(indices)

    Z = np.stack(embeddings, axis=0)
    player_info = _build_player_info(player_ids, metadata, n_events)

    print(f"  [action_profile] {Z.shape[0]} players, {Z.shape[1]}-D")
    _save_heuristic(Z, player_info, output_dir)
    return Z, player_info


# ── Heuristic 3: FIFA attributes ────────────────────────────────────────

_FIFA_STATS = [
    "fifa_attacking_crossing", "fifa_attacking_finishing",
    "fifa_attacking_heading_accuracy", "fifa_attacking_short_passing",
    "fifa_attacking_volleys",
    "fifa_skill_dribbling", "fifa_skill_curve", "fifa_skill_fk_accuracy",
    "fifa_skill_long_passing", "fifa_skill_ball_control",
    "fifa_movement_acceleration", "fifa_movement_sprint_speed",
    "fifa_movement_agility", "fifa_movement_reactions",
    "fifa_movement_balance",
    "fifa_power_shot_power", "fifa_power_jumping", "fifa_power_stamina",
    "fifa_power_strength", "fifa_power_long_shots",
    "fifa_mentality_aggression", "fifa_mentality_interceptions",
    "fifa_mentality_positioning", "fifa_mentality_vision",
    "fifa_mentality_penalties", "fifa_mentality_composure",
    "fifa_defending_marking_awareness", "fifa_defending_standing_tackle",
    "fifa_defending_sliding_tackle",
    "fifa_goalkeeping_diving", "fifa_goalkeeping_handling",
    "fifa_goalkeeping_kicking", "fifa_goalkeeping_positioning",
    "fifa_goalkeeping_reflexes",
]


def generate_fifa_heuristic(
    metadata: pd.DataFrame,
    male_fifa_csv: Path,
    female_fifa_csv: Path,
    output_dir: Path,
    min_events: int = 50,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Use FIFA sub-attribute vectors as player embeddings.

    Only players present in both the StatsBomb corpus (with sufficient events)
    and the FIFA CSVs are included.
    """
    print("\n  [fifa_heuristic] Loading FIFA data ...")
    dfs = []
    for csv_path in [male_fifa_csv, female_fifa_csv]:
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
            print(f"    Loaded {csv_path.name}: {len(dfs[-1])} rows")
    if not dfs:
        print("    No FIFA CSVs found — cannot build FIFA heuristic.")
        return np.empty((0, 0)), pd.DataFrame()

    fifa = pd.concat(dfs, ignore_index=True)

    # Deduplicate: keep first match per sb_player_id
    fifa = fifa.drop_duplicates(subset=["sb_player_id"], keep="first")

    # Count events per player from metadata
    pid_counts: Dict[int, int] = {}
    if "player_id" in metadata.columns:
        for pid, cnt in metadata["player_id"].value_counts().items():
            if int(pid) >= 0:
                pid_counts[int(pid)] = int(cnt)

    player_ids = []
    embeddings = []
    n_events: Dict[int, int] = {}

    for _, row in fifa.iterrows():
        pid = int(row["sb_player_id"])
        if pid_counts.get(pid, 0) < min_events:
            continue

        vec = []
        valid = True
        for stat in _FIFA_STATS:
            val = row.get(stat)
            if pd.isna(val):
                valid = False
                break
            vec.append(float(val))

        if not valid:
            continue

        player_ids.append(pid)
        embeddings.append(np.array(vec, dtype=np.float32))
        n_events[pid] = pid_counts.get(pid, 0)

    if not embeddings:
        print("    No players matched — cannot build FIFA heuristic.")
        return np.empty((0, 0)), pd.DataFrame()

    Z = np.stack(embeddings, axis=0)  # (n_players, 34)
    # Normalize to [0, 1] per attribute (FIFA stats are 1-99 scale)
    col_min = Z.min(axis=0, keepdims=True)
    col_max = Z.max(axis=0, keepdims=True)
    denom = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    Z = (Z - col_min) / denom

    player_info = _build_player_info(player_ids, metadata, n_events)

    print(f"  [fifa_heuristic] {Z.shape[0]} players, {Z.shape[1]}-D")
    _save_heuristic(Z, player_info, output_dir)
    return Z, player_info
