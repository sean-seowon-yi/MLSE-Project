"""
Embedding-only evaluations that work for both GNN models and simple heuristics.

All evaluations require only:
  - ``player_embeddings.npy``  (n_players, D)
  - ``player_info.parquet``    (player_id, player_name, position_name, ...)

No trained model, graphs, or checkpoints are needed.

Evaluations
───────────
1. **Position-group retrieval precision** — fraction of top-K neighbors
   sharing the query player's coarse position group (GK/Def/Mid/Fwd).
2. **Split-half embedding stability** — split each player's events into
   two random halves, recompute embeddings on each, and check if the
   player retrieves themselves as the nearest neighbor.
3. **Qualitative nearest-neighbor table** — for well-known players, show
   top-5 neighbors with name, position, and similarity score.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import POSITION_GROUPS

_POS_TO_GROUP: Dict[str, str] = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _p in _positions:
        _POS_TO_GROUP[_p] = _grp


# ── 1. Position-group retrieval precision ────────────────────────────────

def evaluate_position_retrieval(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Optional[Path] = None,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
    gender_map: Optional[Dict[int, str]] = None,
) -> dict:
    """Measure what fraction of top-K neighbors share the query's position group.

    Returns per-group and overall precision@K, plus a bar chart visualization.
    """
    pids = player_info["player_id"].values
    positions = player_info["position_name"].values
    groups = np.array([_POS_TO_GROUP.get(str(p), "Unknown") for p in positions])
    n = Z.shape[0]

    sim = cosine_similarity(Z)
    np.fill_diagonal(sim, -np.inf)

    # Optional gender filtering
    gender_arr = None
    if gender_map:
        gender_arr = np.array([gender_map.get(int(pid), "unknown") for pid in pids])

    results_per_k: Dict[str, Dict[str, float]] = {}
    group_names = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

    for k in ks:
        group_precisions: Dict[str, List[float]] = {g: [] for g in group_names}
        all_precisions: List[float] = []

        for i in range(n):
            query_group = groups[i]
            if query_group == "Unknown":
                continue

            scores = sim[i].copy()
            if gender_arr is not None:
                mask = gender_arr != gender_arr[i]
                scores[mask] = -np.inf

            top_k_idx = np.argsort(scores)[-k:][::-1]
            neighbor_groups = groups[top_k_idx]
            prec = float(np.mean(neighbor_groups == query_group))
            group_precisions[query_group].append(prec)
            all_precisions.append(prec)

        per_group = {}
        for g in group_names:
            vals = group_precisions[g]
            per_group[g] = {
                "mean_precision": float(np.mean(vals)) if vals else 0.0,
                "n_players": len(vals),
            }

        results_per_k[str(k)] = {
            "overall_precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
            "n_players": len(all_precisions),
            "per_group": per_group,
        }

    result = {"position_retrieval": results_per_k}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "position_retrieval_results.json", "w") as f:
            json.dump(result, f, indent=2)

        _plot_position_retrieval(results_per_k, ks, group_names, output_dir)

        lines = ["Position-Group Retrieval Precision", "=" * 40, ""]
        for k in ks:
            rk = results_per_k[str(k)]
            lines.append(f"Precision@{k}: {rk['overall_precision']:.3f}  (n={rk['n_players']})")
            for g in group_names:
                gd = rk["per_group"][g]
                lines.append(f"  {g:12s}: {gd['mean_precision']:.3f}  (n={gd['n_players']})")
            lines.append("")
        report_path = output_dir / "position_retrieval_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return result


def _plot_position_retrieval(
    results_per_k: dict,
    ks: Tuple[int, ...],
    group_names: List[str],
    output_dir: Path,
) -> None:
    colors = {"Goalkeeper": "#e41a1c", "Defender": "#377eb8",
              "Midfielder": "#4daf4a", "Forward": "#ff7f00"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ks))
    width = 0.18

    for j, g in enumerate(group_names):
        vals = [results_per_k[str(k)]["per_group"][g]["mean_precision"] for k in ks]
        ax.bar(x + j * width, vals, width, label=g, color=colors.get(g, "#999999"))

    overall_vals = [results_per_k[str(k)]["overall_precision"] for k in ks]
    ax.plot(x + 1.5 * width, overall_vals, "ko-", label="Overall", linewidth=2, markersize=6)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"@{k}" for k in ks])
    ax.set_ylabel("Precision (same position group)")
    ax.set_title("Position-Group Retrieval Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "position_retrieval.png", dpi=150)
    plt.close(fig)


# ── 2. Split-half embedding stability ───────────────────────────────────

def evaluate_split_half_stability(
    event_features: np.ndarray,
    metadata: pd.DataFrame,
    embed_fn,
    output_dir: Optional[Path] = None,
    min_events: int = 100,
    seed: int = 42,
    gender_map: Optional[Dict[int, str]] = None,
) -> dict:
    """Split each player's events into two halves and check self-retrieval.

    Parameters
    ----------
    event_features : (n_events, D) array (raw, unmasked features).
    metadata : DataFrame with player_id, position_name columns.
    embed_fn : callable(feature_rows: ndarray, event_types: list[str]) -> ndarray (d,)
        Produces one 1-D embedding vector from a set of event features / types.
    min_events : Minimum events per player to include.
    seed : Random seed for the split.
    gender_map : Optional gender filtering.
    """
    rng = random.Random(seed)
    pid_col = metadata["player_id"].values
    etype_col = metadata["event_type"].values if "event_type" in metadata.columns else None

    player_events: Dict[int, List[int]] = defaultdict(list)
    for i, pid in enumerate(pid_col):
        if pid >= 0:
            player_events[int(pid)].append(i)

    testable_pids = [pid for pid, idxs in player_events.items()
                     if len(idxs) >= min_events]

    if len(testable_pids) < 5:
        result = {"split_half": {"status": "insufficient_players",
                                 "n_testable": len(testable_pids)}}
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "split_half_results.json", "w") as f:
                json.dump(result, f, indent=2)
        return result

    embeddings_a: Dict[int, np.ndarray] = {}
    embeddings_b: Dict[int, np.ndarray] = {}
    player_positions: Dict[int, str] = {}

    for pid in testable_pids:
        idxs = list(player_events[pid])
        rng.shuffle(idxs)
        mid = len(idxs) // 2
        half_a, half_b = idxs[:mid], idxs[mid:]

        etypes_a = [str(etype_col[i]) for i in half_a] if etype_col is not None else ["Unknown"] * len(half_a)
        etypes_b = [str(etype_col[i]) for i in half_b] if etype_col is not None else ["Unknown"] * len(half_b)

        embeddings_a[pid] = embed_fn(event_features[half_a], etypes_a)
        embeddings_b[pid] = embed_fn(event_features[half_b], etypes_b)

        rows = metadata[metadata["player_id"] == pid]
        pos_mode = rows["position_name"].mode()
        player_positions[pid] = str(pos_mode.iloc[0]) if len(pos_mode) > 0 else "Unknown"

    ordered_pids = list(testable_pids)
    Za = np.stack([embeddings_a[p] for p in ordered_pids])
    Zb = np.stack([embeddings_b[p] for p in ordered_pids])

    # Gallery = all B-halves; queries = A-halves
    sim = cosine_similarity(Za, Zb)  # (n, n)
    n = len(ordered_pids)

    gender_arr = None
    if gender_map:
        gender_arr = np.array([gender_map.get(pid, "unknown") for pid in ordered_pids])

    ranks = []
    self_sims = []
    per_player = []

    for i in range(n):
        scores = sim[i].copy()
        if gender_arr is not None:
            mask = gender_arr != gender_arr[i]
            scores[mask] = -np.inf

        sorted_idx = np.argsort(scores)[::-1]
        rank = int(np.where(sorted_idx == i)[0][0]) + 1
        ranks.append(rank)
        self_sims.append(float(sim[i, i]))

        grp = _POS_TO_GROUP.get(player_positions[ordered_pids[i]], "Unknown")
        per_player.append({
            "player_id": ordered_pids[i],
            "position_group": grp,
            "self_rank": rank,
            "self_cosine": float(sim[i, i]),
        })

    ranks_arr = np.array(ranks)
    summary = {
        "n_players": n,
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": int(np.median(ranks_arr)),
        "mean_self_cosine": float(np.mean(self_sims)),
        "hit_at_1": float(np.mean(ranks_arr <= 1)),
        "hit_at_5": float(np.mean(ranks_arr <= 5)),
        "hit_at_10": float(np.mean(ranks_arr <= 10)),
        "hit_at_20": float(np.mean(ranks_arr <= 20)),
    }

    # Per-group breakdown
    by_group: Dict[str, Dict] = {}
    for grp in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        grp_players = [p for p in per_player if p["position_group"] == grp]
        if grp_players:
            grp_ranks = np.array([p["self_rank"] for p in grp_players])
            by_group[grp] = {
                "n": len(grp_players),
                "mean_rank": float(np.mean(grp_ranks)),
                "hit_at_1": float(np.mean(grp_ranks <= 1)),
                "mean_self_cosine": float(np.mean([p["self_cosine"] for p in grp_players])),
            }

    result = {"split_half": {"summary": summary, "by_group": by_group, "per_player": per_player}}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "split_half_results.json", "w") as f:
            json.dump(result, f, indent=2)

        _plot_split_half(summary, by_group, ranks_arr, output_dir)

        lines = ["Split-Half Embedding Stability", "=" * 40, ""]
        lines.append(f"Players tested : {n}")
        lines.append(f"Mean self-rank : {summary['mean_rank']:.1f} / {n}")
        lines.append(f"Median rank    : {summary['median_rank']} / {n}")
        lines.append(f"Mean self-cos  : {summary['mean_self_cosine']:.4f}")
        lines.append(f"Hit@1          : {summary['hit_at_1']:.1%}")
        lines.append(f"Hit@5          : {summary['hit_at_5']:.1%}")
        lines.append(f"Hit@10         : {summary['hit_at_10']:.1%}")
        lines.append("")
        lines.append("Per position group:")
        for grp, gd in by_group.items():
            lines.append(f"  {grp:12s}: mean_rank={gd['mean_rank']:.1f}  "
                         f"hit@1={gd['hit_at_1']:.1%}  cos={gd['mean_self_cosine']:.4f}  (n={gd['n']})")
        report_path = output_dir / "split_half_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return result


def _plot_split_half(summary: dict, by_group: dict, ranks: np.ndarray, output_dir: Path) -> None:
    colors = {"Goalkeeper": "#e41a1c", "Defender": "#377eb8",
              "Midfielder": "#4daf4a", "Forward": "#ff7f00"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CDF of ranks
    ax = axes[0]
    sorted_ranks = np.sort(ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    ax.plot(sorted_ranks, cdf, "k-", linewidth=2)
    for k, ls in [(1, ":"), (5, "--"), (10, "-.")]:
        ax.axvline(k, color="gray", linestyle=ls, alpha=0.5, label=f"rank={k}")
    ax.set_xlabel("Self-retrieval rank")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Split-Half: Rank CDF")
    ax.set_xlim(0, min(50, max(ranks) + 2))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Right: Hit@1 and mean cosine by group
    ax = axes[1]
    groups = list(by_group.keys())
    hit1 = [by_group[g]["hit_at_1"] for g in groups]
    bar_colors = [colors.get(g, "#999") for g in groups]
    x = np.arange(len(groups))
    ax.bar(x, hit1, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Hit@1 rate")
    ax.set_title("Split-Half: Hit@1 by Position Group")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "split_half_stability.png", dpi=150)
    plt.close(fig)


# ── 3. Qualitative nearest-neighbor table ────────────────────────────────

NOTABLE_PLAYERS = [
    # Male — Goalkeeper
    (5570,  "Manuel Neuer"),
    # Male — Defenders
    (3669,  "Virgil van Dijk"),
    (3664,  "Trent Alexander-Arnold"),
    # Male — Midfielders
    (5574,  "Toni Kroos"),
    (3089,  "Kevin De Bruyne"),
    (30714, "Jude Bellingham"),
    # Male — Forwards
    (5503,  "Lionel Messi"),
    (5207,  "Cristiano Ronaldo"),
    (3009,  "Kylian Mbappé"),
    (10955, "Harry Kane"),
    # Female — Defenders
    (10178, "Lucy Bronze"),
    (10125, "Wendie Renard"),
    # Female — Midfielders
    (15284, "Aitana Bonmatí"),
    (10143, "Alexia Putellas"),
    # Female — Forward
    (15555, "Lauren Hemp"),
]


def evaluate_qualitative_neighbors(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Optional[Path] = None,
    top_k: int = 5,
    gender_map: Optional[Dict[int, str]] = None,
    notable_players: Optional[List[Tuple[int, str]]] = None,
) -> dict:
    """For well-known players, produce a top-K neighbor table."""
    if notable_players is None:
        notable_players = NOTABLE_PLAYERS

    pids = player_info["player_id"].values
    pid_to_idx = {int(p): i for i, p in enumerate(pids)}

    sim = cosine_similarity(Z)
    np.fill_diagonal(sim, -np.inf)

    gender_arr = None
    if gender_map:
        gender_arr = np.array([gender_map.get(int(p), "unknown") for p in pids])

    tables: List[dict] = []
    lines = ["Qualitative Nearest-Neighbor Table", "=" * 70, ""]

    for pid, name in notable_players:
        if pid not in pid_to_idx:
            continue
        idx = pid_to_idx[pid]
        query_row = player_info.iloc[idx]
        query_pos = str(query_row.get("position_name", "Unknown"))
        query_group = _POS_TO_GROUP.get(query_pos, "Unknown")

        scores = sim[idx].copy()
        if gender_arr is not None:
            mask = gender_arr != gender_arr[idx]
            scores[mask] = -np.inf

        top_idx = np.argsort(scores)[-top_k:][::-1]

        neighbors = []
        for ni in top_idx:
            n_row = player_info.iloc[ni]
            n_pos = str(n_row.get("position_name", "Unknown"))
            neighbors.append({
                "player_id": int(pids[ni]),
                "player_name": str(n_row.get("player_name", "")),
                "position": n_pos,
                "position_group": _POS_TO_GROUP.get(n_pos, "Unknown"),
                "cosine_similarity": float(scores[ni]),
            })

        entry = {
            "query_player_id": pid,
            "query_name": name,
            "query_position": query_pos,
            "query_group": query_group,
            "neighbors": neighbors,
        }
        tables.append(entry)

        lines.append(f"Query: {name} ({query_pos})")
        lines.append(f"{'─'*65}")
        lines.append(f"  {'Rank':<5} {'Player':<30} {'Position':<25} {'Cosine':>8}")
        for rank_i, nb in enumerate(neighbors, 1):
            lines.append(f"  {rank_i:<5} {nb['player_name']:<30} {nb['position']:<25} {nb['cosine_similarity']:>8.4f}")
        lines.append("")

    result = {"qualitative_neighbors": tables}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "qualitative_neighbors_results.json", "w") as f:
            json.dump(result, f, indent=2)
        report_path = output_dir / "qualitative_neighbors_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        if tables:
            _plot_qualitative_neighbors(tables, output_dir)

    return result


def _plot_qualitative_neighbors(tables: List[dict], output_dir: Path) -> None:
    n_queries = len(tables)
    if n_queries == 0:
        return

    fig, ax = plt.subplots(figsize=(15, max(4, n_queries * 1.3 + 1)))
    ax.axis("off")

    col_headers = ["Query Player", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]
    n_cols = len(col_headers)

    cell_text = []
    cell_colors = []
    group_colors = {
        "Goalkeeper": "#ffcccc", "Defender": "#cce5ff",
        "Midfielder": "#ccffcc", "Forward": "#ffe5cc", "Unknown": "#f0f0f0",
    }

    for entry in tables:
        row = [f"{entry['query_name']}\n({entry['query_group'][:3]})"]
        row_colors = [group_colors.get(entry["query_group"], "#f0f0f0")]
        for nb in entry["neighbors"][:5]:
            row.append(f"{nb['player_name']}\n({nb['position_group'][:3]}) {nb['cosine_similarity']:.3f}")
            match = nb["position_group"] == entry["query_group"]
            row_colors.append("#d4edda" if match else "#f8d7da")
        while len(row) < n_cols:
            row.append("")
            row_colors.append("#ffffff")
        cell_text.append(row)
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellColours=cell_colors,
        colColours=["#e0e0e0"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.2)

    ax.set_title("Qualitative Nearest Neighbors (green = same group, red = different)", fontsize=10, pad=20)
    fig.tight_layout()
    fig.savefig(output_dir / "qualitative_neighbors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
