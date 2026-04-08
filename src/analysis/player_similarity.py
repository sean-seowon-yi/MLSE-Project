"""
analysis/player_similarity.py — Three methods for measuring player passing similarity.

Section 16 — Actual pass profile vectors (feature-based cosine similarity)
Section 17 — Cross-predicted pass profiles (model predictions, controlled situations)
Section 18 — Direct distribution comparison (Energy Distance + MMD)

All three methods produce a (P, P) similarity/dissimilarity matrix over the
players in the dataset.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import ot 
import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy as sp_entropy
from tqdm.auto import tqdm
from torch_geometric.data import Data

from config import (
    PITCH_LEN, PITCH_WID, PITCH_DIAG,
    N_SAMPLE, SAMPLE_SEED, DEVICE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Section 16 — Actual pass profile vectors
# ══════════════════════════════════════════════════════════════════════════════

def compute_pass_profiles(
    pass_events: pd.DataFrame,
    min_passes: int = 5,
) -> pd.DataFrame:
    """
    Build a 10-d statistical profile per player from their actual passes.

    Features
    --------
    end_x_mean/std     pass destination x, normalised
    end_y_mean/std     pass destination y, normalised
    dist_mean/std      pass length / pitch diagonal
    dir_entropy        directional variety in [0, 1]
    zone_def/mid/att   fraction of passes from each pitch third

    Returns DataFrame indexed by player_id.
    """
    rows = []
    for pid, grp in pass_events.groupby("player_id"):
        if len(grp) < min_passes:
            continue

        ex = grp["event_x"].values.astype(float)
        ey = grp["event_y"].values.astype(float)
        px = grp["pass_end_x"].values.astype(float)
        py = grp["pass_end_y"].values.astype(float)

        valid = ~(np.isnan(ex) | np.isnan(px))
        if valid.sum() < min_passes:
            continue
        ex, ey, px, py = ex[valid], ey[valid], px[valid], py[valid]

        dist = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        angle = np.arctan2(py - ey, px - ex)
        dir_ent = _direction_entropy(angle)
        zone_frc = _zone_fractions(ex)

        rows.append({
            "player_id":   pid,
            "end_x_mean":  px.mean() / PITCH_LEN,
            "end_x_std":   px.std()  / PITCH_LEN,
            "end_y_mean":  py.mean() / PITCH_WID,
            "end_y_std":   py.std()  / PITCH_WID,
            "dist_mean":   dist.mean() / PITCH_DIAG,
            "dist_std":    dist.std()  / PITCH_DIAG,
            "dir_entropy": dir_ent,
            "zone_def":    zone_frc[0],
            "zone_mid":    zone_frc[1],
            "zone_att":    zone_frc[2],
        })

    return pd.DataFrame(rows).set_index("player_id")


def cosine_similarity_matrix(profile_df: pd.DataFrame) -> np.ndarray:
    """
    Compute a (P, P) cosine similarity matrix from a profile DataFrame.
    Columns are z-scored before computing cosine similarity.
    """
    mat = profile_df.values.astype(np.float32)
    z = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
    normed = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return normed @ normed.T


# ══════════════════════════════════════════════════════════════════════════════
# Section 17 — Cross-predicted pass profiles
# ══════════════════════════════════════════════════════════════════════════════

def build_style_sample(
    graphs: List[Data],
    n_total: int = N_SAMPLE,
    seed: int = SAMPLE_SEED,
) -> List[Data]:
    """
    Return ~n_total graphs stratified evenly across:
      - 3 pitch zones  (passer origin: def / mid / att third)
      - 3 game states  (drawing / winning / losing)

    Stratification removes situational confounding from the cross-prediction.
    """
    rng = np.random.default_rng(seed)
    strata: Dict[Tuple, List] = defaultdict(list)

    for g in graphs:
        px = g.passer_pos[0].item()
        zone = 0 if px < 40 else (1 if px < 80 else 2)
        u = g.u[0].cpu().numpy()
        gs = int(np.argmax(u[7:10]))  # 0=drawing, 1=winning, 2=losing
        strata[(zone, gs)].append(g)

    n_per = max(1, n_total // 9)
    sample: List[Data] = []
    for key, bucket in strata.items():
        chosen = rng.choice(len(bucket), size=min(n_per, len(bucket)), replace=False)
        sample.extend([bucket[i] for i in chosen])

    zone_names = ["Def", "Mid", "Att"]
    gs_names = ["Drawing", "Winning", "Losing"]
    print(f"Style sample: {len(sample)} graphs across {len(strata)} strata")
    for (z, gs), bucket in sorted(strata.items()):
        sampled = min(n_per, len(bucket))
        print(f"  {zone_names[z]} / {gs_names[gs]}: {len(bucket)} avail -> {sampled} sampled")
    return sample


@torch.no_grad()
def run_cross_prediction(
    sample_graphs: List[Data],
    model: torch.nn.Module,
    player_vocab: Dict,
    device=DEVICE,
    batch_size: int = 64,
) -> Dict[int, List[Tuple]]:
    """
    For each graph in sample_graphs, run inference once per known player
    (swapping actor_idx). Returns:

        buckets[player_id] = [(pred_x, pred_y, passer_x, passer_y), ...]

    Every player gets exactly len(sample_graphs) predictions on the same
    tactical situations, removing situational confounding.
    """
    model.eval()
    all_pids = sorted(player_vocab.keys())
    all_idxs = torch.tensor(
        [player_vocab[p] for p in all_pids], dtype=torch.long, device=device
    )
    P = len(all_pids)
    buckets: Dict[int, List] = defaultdict(list)

    for g in tqdm(sample_graphs, desc="Cross-predicting"):
        passer_x = g.passer_pos[0].item()
        passer_y = g.passer_pos[1].item()

        for start in range(0, P, batch_size):
            batch_pids = all_pids[start : start + batch_size]
            batch_idxs = all_idxs[start : start + batch_size]
            B = len(batch_pids)

            x_rep = g.x.unsqueeze(0).expand(B, -1, -1).reshape(B * g.num_nodes, -1).to(device)
            ei_rep = torch.cat(
                [g.edge_index + i * g.num_nodes for i in range(B)], dim=1
            ).to(device)
            ea_rep = g.edge_attr.unsqueeze(0).expand(B, -1, -1).reshape(B * g.num_edges, -1).to(device)
            u_rep = g.u.expand(B, -1).to(device)
            batch_vec = torch.arange(B, device=device).repeat_interleave(g.num_nodes)

            from torch_geometric.data import Data as PyGData
            d = PyGData(
                x=x_rep, edge_index=ei_rep, edge_attr=ea_rep,
                u=u_rep, actor_idx=batch_idxs, batch=batch_vec,
            )
            preds = model(d).cpu().numpy()  # (B, 2) normalised

            for pid, pred_norm in zip(batch_pids, preds):
                buckets[pid].append((
                    pred_norm[0] * PITCH_LEN,
                    pred_norm[1] * PITCH_WID,
                    passer_x, passer_y,
                ))
    return buckets


def compute_predicted_profiles(
    buckets: Dict[int, List],
    min_passes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build centroid and 10-d distribution profile per player from cross_buckets.

    Returns
    -------
    centroids  : DataFrame  player_id -> mean_pred_x, mean_pred_y, n_passes
    profiles   : DataFrame  player_id -> 10 distribution features
    """
    centroid_rows, profile_rows = [], []

    for pid, entries in buckets.items():
        if len(entries) < min_passes:
            continue
        arr = np.array(entries, dtype=np.float32)  # (M, 4)
        px, py = arr[:, 0], arr[:, 1]
        passer_x, passer_y = arr[:, 2], arr[:, 3]

        centroid_rows.append({
            "player_id":   pid,
            "mean_pred_x": float(px.mean()),
            "mean_pred_y": float(py.mean()),
            "n_passes":    len(entries),
        })

        dx, dy = px - passer_x, py - passer_y
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        angle = np.arctan2(dy, dx)
        dir_ent = _direction_entropy(angle)
        zone_frc = _zone_fractions(px)

        profile_rows.append({
            "player_id":        pid,
            "pred_x_mean":      px.mean()    / PITCH_LEN,
            "pred_x_std":       px.std()     / PITCH_LEN,
            "pred_y_mean":      py.mean()    / PITCH_WID,
            "pred_y_std":       py.std()     / PITCH_WID,
            "pred_dist_mean":   dist.mean()  / PITCH_DIAG,
            "pred_dist_std":    dist.std()   / PITCH_DIAG,
            "pred_dir_entropy": dir_ent,
            "pred_zone_def":    zone_frc[0],
            "pred_zone_mid":    zone_frc[1],
            "pred_zone_att":    zone_frc[2],
        })

    centroids = pd.DataFrame(centroid_rows).set_index("player_id")
    profiles = pd.DataFrame(profile_rows).set_index("player_id")
    return centroids, profiles


# ══════════════════════════════════════════════════════════════════════════════
# Section 18 — Direct distribution comparison (Energy Distance + MMD)
# ══════════════════════════════════════════════════════════════════════════════

def earth_mover_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Earth mover distance (Wasserstein-1) between two (N, 2) point clouds.
    Returns a non-negative scalar in yards.
    """
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    a = torch.ones(len(A)) / len(A)
    b = torch.ones(len(B)) / len(B)

    # cost matrix (pairwise distances)
    C = torch.cdist(A, B)

    # EMD
    emd = ot.emd2(a.numpy(), b.numpy(), C.numpy())
    return emd

def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Unbiased energy distance between two (N, 2) point clouds.
    ED(A,B) = 2*E[||a-b||] - E[||a-a'||] - E[||b-b'||]
    Returns a non-negative scalar in yards.
    """
    cross = _mean_pairwise_dist(A, B)
    within_a = _mean_pairwise_dist(A)
    within_b = _mean_pairwise_dist(B)
    return float(max(0.0, 2 * cross - within_a - within_b))


def mmd_rbf(A: np.ndarray, B: np.ndarray, sigma2: float) -> float:
    """
    Unbiased MMD² estimate with RBF kernel k(x,y) = exp(-||x-y||²/(2σ²)).
    Non-negative (clamped at 0 for numerical safety).
    """
    kAA = _rbf_mean_within(A, sigma2)
    kBB = _rbf_mean_within(B, sigma2)
    kAB = _rbf_mean(A, B, sigma2)
    return float(max(0.0, kAA + kBB - 2 * kAB))


def compute_emd_matrix(
    point_clouds: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """Compute the full (P, P) Earth Mover Distance matrix."""
    pids = sorted(point_clouds.keys())
    P = len(pids)
    mat = np.zeros((P, P), dtype=np.float32)
    for i, j in tqdm(list(itertools.combinations(range(P), 2)), desc="Earth Mover Distance"):
        d = earth_mover_distance(point_clouds[pids[i]], point_clouds[pids[j]])
        mat[i, j] = mat[j, i] = d
    return mat, pids


def compute_ed_matrix(
    point_clouds: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """Compute the full (P, P) Energy Distance matrix."""
    pids = sorted(point_clouds.keys())
    P = len(pids)
    mat = np.zeros((P, P), dtype=np.float32)
    for i, j in tqdm(list(itertools.combinations(range(P), 2)), desc="Energy Distance"):
        d = energy_distance(point_clouds[pids[i]], point_clouds[pids[j]])
        mat[i, j] = mat[j, i] = d
    return mat, pids


def compute_mmd_matrix(
    point_clouds: Dict[int, np.ndarray],
    sigma2: Optional[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int], float]:
    """
    Compute the full (P, P) MMD matrix.
    sigma2 is estimated via the median heuristic if not provided.
    """
    pids = sorted(point_clouds.keys())
    P = len(pids)

    if sigma2 is None:
        all_pts = np.vstack(list(point_clouds.values()))
        rng = np.random.default_rng(seed)
        sub = all_pts[rng.choice(len(all_pts), size=min(2000, len(all_pts)), replace=False)]
        sq = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        sigma2 = float(np.median(sq[sq > 0]))
        print(f"MMD bandwidth σ² = {sigma2:.2f} (median heuristic)")

    mat = np.zeros((P, P), dtype=np.float32)
    for i, j in tqdm(list(itertools.combinations(range(P), 2)), desc="MMD"):
        m = mmd_rbf(point_clouds[pids[i]], point_clouds[pids[j]], sigma2)
        mat[i, j] = mat[j, i] = m
    return mat, pids, sigma2


def dissim_to_sim(D: np.ndarray) -> np.ndarray:
    """Map a non-negative dissimilarity matrix to similarity in [-1, 1]."""
    d_max = D.max()
    if d_max < 1e-8:
        return np.ones_like(D)
    return 1.0 - 2.0 * D / d_max


def nearest_players(
    query_pid: int,
    centroids_df: pd.DataFrame,
    pid_to_label: Dict[int, str],
    n: int = 5,
) -> None:
    """Print the n players whose predicted centroid is closest to query_pid."""
    if query_pid not in centroids_df.index:
        print(f"player_id {query_pid} not in centroid table")
        return
    qxy = centroids_df.loc[query_pid, ["mean_pred_x", "mean_pred_y"]].values
    other = centroids_df.drop(index=query_pid)
    dists = np.linalg.norm(
        other[["mean_pred_x", "mean_pred_y"]].values - qxy, axis=1
    )
    top = np.argsort(dists)[:n]
    print(f"Nearest centroids to {pid_to_label.get(query_pid, str(query_pid))}:")
    for i in top:
        pid = other.index[i]
        print(f"  {dists[i]:5.1f} yds  {pid_to_label.get(pid, str(pid))}")


def extract_point_clouds(
    buckets: Dict[int, List],
) -> Dict[int, np.ndarray]:
    """
    Convert cross_buckets to a dict of (N, 2) numpy arrays (pred_x, pred_y only).
    Convenience wrapper so notebooks don't need to repeat this slicing.
    """
    return {
        pid: np.array(entries, dtype=np.float32)[:, :2]
        for pid, entries in buckets.items()
    }


def most_divergent_pair(
    point_clouds: Dict[int, np.ndarray],
    pred_centroids: pd.DataFrame,
) -> Tuple[int, int]:
    """Return the two player_ids whose predicted centroids are furthest apart."""
    pids = pred_centroids.index.tolist()
    cx = pred_centroids[["mean_pred_x", "mean_pred_y"]].values
    dist_mat = np.linalg.norm(cx[:, None] - cx[None, :], axis=2)
    i_max, j_max = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return pids[i_max], pids[j_max]


def cross_method_correlation(
    matrices: List[np.ndarray],
    n_players: int,
) -> np.ndarray:
    """
    Compute a (M, M) Pearson correlation matrix across M similarity matrices,
    using upper-triangle pairs of the (n_players, n_players) sub-matrices.
    """
    tri = np.triu_indices(n_players, k=1)
    vecs = [mat[tri] for mat in matrices]
    return np.array([
        [float(np.corrcoef(vecs[i], vecs[j])[0, 1]) for j in range(len(vecs))]
        for i in range(len(vecs))
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Name-based similarity query — the main user-facing entry point
# ══════════════════════════════════════════════════════════════════════════════

def find_similar_players(
    name: str,
    pid_to_label: Dict[int, str],
    n: int = 10,
    *,
    # Section 16 — actual profile cosine similarity
    profile_ids: Optional[List[int]] = None,
    profile_cos: Optional[np.ndarray] = None,
    # Section 17 — cross-predicted profile cosine similarity
    pred_profile_ids: Optional[List[int]] = None,
    pred_cos: Optional[np.ndarray] = None,
    # Section 18 — energy distance (lower = more similar)
    cloud_pids: Optional[List[int]] = None,
    ed_mat: Optional[np.ndarray] = None,
    # Section 18 — MMD (lower = more similar)
    mmd_mat: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Look up a player by name and return their top-N most similar players
    across whichever similarity methods you have computed.

    Parameters
    ----------
    name            : partial or full player name, case-insensitive
                      e.g. "messi", "Pedri", "de Bruyne"
    pid_to_label    : dict  player_id -> "First Last (Position)"
    n               : number of similar players to return (default 10)

    Similarity sources (pass whichever you have — all are optional):
    profile_ids     : list of player_ids for the S16 cosine matrix rows/cols
    profile_cos     : (P16, P16) cosine similarity array from S16
    pred_profile_ids: list of player_ids for the S17 cosine matrix rows/cols
    pred_cos        : (P17, P17) cosine similarity array from S17
    cloud_pids      : list of player_ids for the S18 ED/MMD matrix rows/cols
    ed_mat          : (P18, P18) energy distance matrix (yards, lower = closer)
    mmd_mat         : (P18, P18) MMD² matrix (lower = closer)

    Returns
    -------
    DataFrame with columns:
        player_id, name, s16_cosine, s17_cosine, ed_yards, mmd,
        mean_rank  (average rank across available methods, ascending)
    Sorted by mean_rank ascending (most similar first).

    Also prints a formatted table to stdout.

    Raises
    ------
    ValueError  if no matching player is found or no similarity source is given.
    """
    # ── 1. Resolve name → player_id ───────────────────────────────────────────
    query_pid, matched_label = _resolve_player_name(name, pid_to_label)

    # ── 2. Collect candidate player_ids across all available matrices ─────────
    candidate_pids: set = set()
    if profile_ids is not None and profile_cos is not None:
        candidate_pids.update(profile_ids)
    if pred_profile_ids is not None and pred_cos is not None:
        candidate_pids.update(pred_profile_ids)
    if cloud_pids is not None and ed_mat is not None:
        candidate_pids.update(cloud_pids)
    candidate_pids.discard(query_pid)

    if not candidate_pids:
        raise ValueError(
            "No similarity data provided. Pass at least one of: "
            "(profile_ids + profile_cos), (pred_profile_ids + pred_cos), "
            "(cloud_pids + ed_mat)."
        )

    # ── 3. Build score table — one row per candidate ──────────────────────────
    rows = []
    for pid in sorted(candidate_pids):
        row: Dict = {"player_id": pid, "name": pid_to_label.get(pid, str(pid))}

        # S16 cosine — higher = more similar
        if (profile_ids is not None and profile_cos is not None
                and query_pid in profile_ids and pid in profile_ids):
            qi = profile_ids.index(query_pid)
            ci = profile_ids.index(pid)
            row["s16_cosine"] = float(profile_cos[qi, ci])
        else:
            row["s16_cosine"] = np.nan

        # S17 cosine — higher = more similar
        if (pred_profile_ids is not None and pred_cos is not None
                and query_pid in pred_profile_ids and pid in pred_profile_ids):
            qi = pred_profile_ids.index(query_pid)
            ci = pred_profile_ids.index(pid)
            row["s17_cosine"] = float(pred_cos[qi, ci])
        else:
            row["s17_cosine"] = np.nan

        # Energy distance — lower = more similar
        if (cloud_pids is not None and ed_mat is not None
                and query_pid in cloud_pids and pid in cloud_pids):
            qi = cloud_pids.index(query_pid)
            ci = cloud_pids.index(pid)
            row["ed_yards"] = float(ed_mat[qi, ci])
        else:
            row["ed_yards"] = np.nan

        # MMD — lower = more similar
        if (cloud_pids is not None and mmd_mat is not None
                and query_pid in cloud_pids and pid in cloud_pids):
            qi = cloud_pids.index(query_pid)
            ci = cloud_pids.index(pid)
            row["mmd"] = float(mmd_mat[qi, ci])
        else:
            row["mmd"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── 4. Compute per-method ranks then average them ─────────────────────────
    # For cosine: rank descending (higher = better).  For ED/MMD: ascending.
    rank_cols = []
    if df["s16_cosine"].notna().any():
        df["_r_s16"] = df["s16_cosine"].rank(ascending=False, na_option="bottom")
        rank_cols.append("_r_s16")
    if df["s17_cosine"].notna().any():
        df["_r_s17"] = df["s17_cosine"].rank(ascending=False, na_option="bottom")
        rank_cols.append("_r_s17")
    if df["ed_yards"].notna().any():
        df["_r_ed"] = df["ed_yards"].rank(ascending=True, na_option="bottom")
        rank_cols.append("_r_ed")
    if df["mmd"].notna().any():
        df["_r_mmd"] = df["mmd"].rank(ascending=True, na_option="bottom")
        rank_cols.append("_r_mmd")

    df["mean_rank"] = df[rank_cols].mean(axis=1)
    df = (df
          .drop(columns=rank_cols)
          .sort_values("mean_rank")
          .head(n)
          .reset_index(drop=True))
    df.index += 1  # 1-based rank

    # ── 5. Pretty-print ───────────────────────────────────────────────────────
    _print_similar_players(matched_label, df)

    # Drop internal rank col before returning
    return df.drop(columns=["mean_rank"])


def _resolve_player_name(
    name: str,
    pid_to_label: Dict[int, str],
) -> Tuple[int, str]:
    """
    Match a name string to a player_id using case-insensitive substring search.

    Matching priority
    -----------------
    1. Exact match on full label (case-insensitive)
    2. Exact match on name-only part (stripping the position suffix)
    3. All candidates whose name contains the query as a substring
       — if exactly one, use it; if multiple, pick the one with the most
         characters in common (longest common subsequence length) and warn
         about the ambiguity.

    Raises ValueError if no match is found.
    """
    query = name.strip().lower()

    # Build lookup structures
    label_to_pid = {label.lower(): pid for pid, label in pid_to_label.items()}
    # Name-only (strip position suffix " (Midfielder)" etc.)
    name_to_pid: Dict[str, int] = {}
    for pid, label in pid_to_label.items():
        name_part = label.split(" (")[0].lower()
        name_to_pid[name_part] = pid

    # Priority 1: exact full label match
    if query in label_to_pid:
        pid = label_to_pid[query]
        return pid, pid_to_label[pid]

    # Priority 2: exact name-only match
    if query in name_to_pid:
        pid = name_to_pid[query]
        return pid, pid_to_label[pid]

    # Priority 3: substring search on both label and name
    matches = {
        pid: label
        for pid, label in pid_to_label.items()
        if query in label.lower() or query in label.split(" (")[0].lower()
    }

    if not matches:
        # Last resort: try each word in the query individually
        words = query.split()
        for word in words:
            if len(word) < 3:
                continue
            matches = {
                pid: label
                for pid, label in pid_to_label.items()
                if word in label.lower()
            }
            if matches:
                break

    if not matches:
        close = _fuzzy_suggestions(query, pid_to_label, k=5)
        suggestion_str = "\n  ".join(close) if close else "(none found)"
        raise ValueError(
            f"No player matching '{name}' found.\n"
            f"Did you mean one of:\n  {suggestion_str}"
        )

    if len(matches) == 1:
        pid, label = next(iter(matches.items()))
        return pid, label

    # Multiple matches — rank by overlap and pick best, but warn
    def _overlap(label: str) -> int:
        return sum(c in label.lower() for c in query)

    best_pid = max(matches, key=lambda p: _overlap(matches[p]))
    best_label = matches[best_pid]
    others = [label for pid, label in matches.items() if pid != best_pid]
    print(
        f"Ambiguous name '{name}' — matched {len(matches)} players. "
        f"Using: '{best_label}'.\n"
        f"Other matches: {', '.join(others[:5])}"
        + (" ..." if len(others) > 5 else "")
    )
    return best_pid, best_label


def _fuzzy_suggestions(query: str, pid_to_label: Dict[int, str], k: int = 5) -> List[str]:
    """Return up to k player labels ranked by character overlap with query."""
    def score(label: str) -> int:
        label_l = label.lower()
        return sum(c in label_l for c in query)

    ranked = sorted(pid_to_label.values(), key=score, reverse=True)
    return ranked[:k]


def _print_similar_players(query_label: str, df: pd.DataFrame) -> None:
    """Print a formatted similarity table to stdout."""
    col_map = {
        "s16_cosine": "S16 cosine",
        "s17_cosine": "S17 cosine",
        "ed_yards":   "ED (yds)",
        "mmd":        "MMD²",
    }
    present = [c for c in col_map if c in df.columns and df[c].notna().any()]

    header_parts = ["Rank", f"{'Player':<40}"]
    for c in present:
        header_parts.append(f"{col_map[c]:>12}")
    header = "  ".join(header_parts)

    sep = "─" * len(header)
    print(f"\nTop {len(df)} most similar players to: {query_label}")
    print(sep)
    print(header)
    print(sep)

    for rank, row in df.iterrows():
        parts = [f"{rank:>4}", f"{row['name']:<40}"]
        for c in present:
            val = row[c]
            if np.isnan(val):
                parts.append(f"{'—':>12}")
            elif c in ("s16_cosine", "s17_cosine"):
                parts.append(f"{val:>12.4f}")
            elif c == "ed_yards":
                parts.append(f"{val:>11.1f}y")
            else:
                parts.append(f"{val:>12.5f}")
        print("  ".join(parts))
    print(sep)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _direction_entropy(angles: np.ndarray) -> float:
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist + 1e-6
    return float(sp_entropy(hist / hist.sum())) / np.log(8)


def _zone_fractions(x: np.ndarray) -> np.ndarray:
    zones = np.clip(np.digitize(x, [0, 40, 80, 120]) - 1, 0, 2)
    return np.bincount(zones, minlength=3) / len(x)


def _mean_pairwise_dist(X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
    if Y is None:
        n = len(X)
        if n < 2:
            return 0.0
        diff = X[:, None, :] - X[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        return d[np.triu_indices(n, k=1)].mean()
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).mean()


def _rbf_mean(X: np.ndarray, Y: np.ndarray, sigma2: float) -> float:
    diff = X[:, None, :] - Y[None, :, :]
    return float(np.exp(-(diff ** 2).sum(-1) / (2 * sigma2)).mean())


def _rbf_mean_within(X: np.ndarray, sigma2: float) -> float:
    n = len(X)
    if n < 2:
        return 1.0
    diff = X[:, None, :] - X[None, :, :]
    sq = (diff ** 2).sum(-1)
    mask = ~np.eye(n, dtype=bool)
    return float(np.exp(-sq[mask] / (2 * sigma2)).mean())
