"""
Pseudo ground-truth evaluation for player similarity.

Defines externally sourced player-similarity pairs (from public StatsBomb
articles) and checks where the expected partner ranks in the model's
nearest-neighbour list.  Produces a formatted report with per-pair rank,
cosine similarity, and aggregate hit-rate statistics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityPair:
    """One directional ground-truth expectation."""
    tier: int
    player_a_name: str
    player_a_id: int
    player_b_name: str
    player_b_id: int
    source: str
    note: str = ""


GROUND_TRUTH_PAIRS: List[SimilarityPair] = [
    # ── Tier 1: strong directional expectations ──────────────────
    SimilarityPair(
        tier=1,
        player_a_name="Vivianne Miedema",
        player_a_id=15623,
        player_b_name="Caldentey",
        player_b_id=10161,
        source="StatsBomb: Caldentey is the most similar forward to Miedema (93% similarity).",
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Trent Alexander-Arnold",
        player_a_id=3664,
        player_b_name="Achraf Hakimi",
        player_b_id=5245,
        source="StatsBomb recruitment example: 'no surprise' Hakimi flagged as very similar to TAA.",
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Jordi Alba",
        player_a_id=5211,
        player_b_name="Andrew Robertson",
        player_b_id=3655,
        source="StatsBomb: Robertson is the strongest like-for-like Alba replacement.",
    ),
    # ── Tier 2: good directional expectations ────────────────────
    SimilarityPair(
        tier=2,
        player_a_name="Toni Kroos",
        player_a_id=5574,
        player_b_name="Enzo Fernandez",
        player_b_id=38718,
        source="StatsBomb: Enzo appears in the raw top-5 for Kroos replacement.",
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Trent Alexander-Arnold",
        player_a_id=3664,
        player_b_name="Joakim Maehle",
        player_b_id=16554,
        source="StatsBomb: Maehle is an interesting profile match for TAA.",
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Jadon Sancho",
        player_a_id=12423,
        player_b_name="Ruben Vargas",
        player_b_id=30401,
        source="StatsBomb Sancho replacement article: Vargas highlighted as strong candidate.",
        note="Sancho has only 72 possessions (borderline).",
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Jadon Sancho",
        player_a_id=12423,
        player_b_name="Christoph Baumgartner",
        player_b_id=24977,
        source="StatsBomb: Baumgartner's aggressive final-third style resembles Sancho's.",
        note="Sancho has only 72 possessions (borderline). Baumgartner plays more centrally.",
    ),
    # ── Tier 3: weak / conditional expectations ──────────────────
    SimilarityPair(
        tier=3,
        player_a_name="Harry Kane",
        player_a_id=10955,
        player_b_name="Rafael Leao",
        player_b_id=18360,
        source="StatsBomb: Leao rises when build-up/link-play is weighted more heavily.",
        note="Conditional on altered weighting. Very different player types.",
    ),
    SimilarityPair(
        tier=3,
        player_a_name="Harry Kane",
        player_a_id=10955,
        player_b_name="Joao Felix",
        player_b_id=12041,
        source="StatsBomb: Felix is a build-up-oriented Kane analogue under altered weighting.",
        note="Conditional on altered weighting.",
    ),
    SimilarityPair(
        tier=3,
        player_a_name="Felix Uduokhai",
        player_a_id=8245,
        player_b_name="Harry Souttar",
        player_b_id=22293,
        source="Speculative: both near Tarkowski's top-5 in StatsBomb (transitive inference).",
        note="No direct StatsBomb endorsement. Uduokhai has only 80 possessions.",
    ),
]


def evaluate_ground_truth(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Check every ground-truth pair against the embedding space.

    For each pair (A, B):
      - Compute cosine similarity.
      - Find B's rank in A's neighbour list (and vice versa).

    Parameters
    ----------
    Z : (n_players, d) embedding matrix.
    player_info : DataFrame with at least ``player_id`` and ``player_name``.
    output_dir : if provided, saves a text report and JSON results there.

    Returns
    -------
    dict with ``"pairs"`` (per-pair results) and ``"summary"`` (aggregates).
    """
    pids = player_info["player_id"].values
    pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}

    sim_matrix = cosine_similarity(Z)

    pair_results = []

    for pair in GROUND_TRUTH_PAIRS:
        entry = {
            "tier": pair.tier,
            "player_a": pair.player_a_name,
            "player_a_id": pair.player_a_id,
            "player_b": pair.player_b_name,
            "player_b_id": pair.player_b_id,
            "source": pair.source,
            "note": pair.note,
        }

        idx_a = pid_to_idx.get(pair.player_a_id)
        idx_b = pid_to_idx.get(pair.player_b_id)

        if idx_a is None or idx_b is None:
            missing = []
            if idx_a is None:
                missing.append(f"{pair.player_a_name} ({pair.player_a_id})")
            if idx_b is None:
                missing.append(f"{pair.player_b_name} ({pair.player_b_id})")
            entry["status"] = "MISSING"
            entry["missing"] = missing
            entry["cosine_sim"] = None
            entry["rank_b_in_a"] = None
            entry["rank_a_in_b"] = None
            pair_results.append(entry)
            continue

        cos_sim = float(sim_matrix[idx_a, idx_b])

        rank_b_in_a = _get_rank(sim_matrix, idx_a, idx_b)
        rank_a_in_b = _get_rank(sim_matrix, idx_b, idx_a)

        entry["status"] = "OK"
        entry["cosine_sim"] = round(cos_sim, 4)
        entry["rank_b_in_a"] = rank_b_in_a
        entry["rank_a_in_b"] = rank_a_in_b
        pair_results.append(entry)

    summary = _compute_summary(pair_results, len(pids))

    result = {"pairs": pair_results, "summary": summary}

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_report(pair_results, summary, len(pids), output_dir)
        json_path = output_dir / "ground_truth_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def _get_rank(sim_matrix: np.ndarray, query_idx: int, target_idx: int) -> int:
    """Return 1-based rank of target in query's similarity list."""
    sims = sim_matrix[query_idx].copy()
    sims[query_idx] = -np.inf
    sorted_indices = np.argsort(sims)[::-1]
    rank = int(np.where(sorted_indices == target_idx)[0][0]) + 1
    return rank


def _compute_summary(pair_results: list, n_players: int) -> dict:
    """Aggregate statistics across all evaluated pairs."""
    evaluated = [p for p in pair_results if p["status"] == "OK"]
    missing = [p for p in pair_results if p["status"] == "MISSING"]

    if not evaluated:
        return {
            "n_pairs_total": len(pair_results),
            "n_evaluated": 0,
            "n_missing": len(missing),
        }

    ranks_a_to_b = [p["rank_b_in_a"] for p in evaluated]
    ranks_b_to_a = [p["rank_a_in_b"] for p in evaluated]
    all_ranks = ranks_a_to_b + ranks_b_to_a
    cosines = [p["cosine_sim"] for p in evaluated]

    def hit_rate(ranks, k):
        return sum(1 for r in ranks if r <= k) / len(ranks)

    by_tier = {}
    for tier in sorted(set(p["tier"] for p in evaluated)):
        tier_pairs = [p for p in evaluated if p["tier"] == tier]
        t_ranks = ([p["rank_b_in_a"] for p in tier_pairs]
                   + [p["rank_a_in_b"] for p in tier_pairs])
        by_tier[f"tier_{tier}"] = {
            "n_pairs": len(tier_pairs),
            "mean_rank": round(np.mean(t_ranks), 1),
            "median_rank": int(np.median(t_ranks)),
            "hit_at_5": round(hit_rate(t_ranks, 5), 3),
            "hit_at_10": round(hit_rate(t_ranks, 10), 3),
            "hit_at_20": round(hit_rate(t_ranks, 20), 3),
        }

    return {
        "n_pairs_total": len(pair_results),
        "n_evaluated": len(evaluated),
        "n_missing": len(missing),
        "n_players_in_embeddings": n_players,
        "mean_cosine": round(float(np.mean(cosines)), 4),
        "mean_rank": round(float(np.mean(all_ranks)), 1),
        "median_rank": int(np.median(all_ranks)),
        "hit_at_5": round(hit_rate(all_ranks, 5), 3),
        "hit_at_10": round(hit_rate(all_ranks, 10), 3),
        "hit_at_20": round(hit_rate(all_ranks, 20), 3),
        "hit_at_50": round(hit_rate(all_ranks, 50), 3),
        "by_tier": by_tier,
    }


def _write_report(
    pair_results: list, summary: dict, n_players: int, output_dir: Path
) -> None:
    """Write a human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  PSEUDO GROUND-TRUTH EVALUATION REPORT")
    lines.append("=" * 78)
    lines.append("")

    n_eval = summary["n_evaluated"]
    n_miss = summary["n_missing"]
    lines.append(f"Players in embedding space : {n_players}")
    lines.append(f"Ground-truth pairs total   : {summary['n_pairs_total']}")
    lines.append(f"  evaluated                : {n_eval}")
    lines.append(f"  skipped (player missing) : {n_miss}")
    lines.append("")

    # ── Per-pair results ──────────────────────────────────────────
    lines.append("-" * 78)
    lines.append("  PER-PAIR RESULTS")
    lines.append("-" * 78)
    lines.append("")

    for p in pair_results:
        tier_label = f"[Tier {p['tier']}]"
        lines.append(f"{tier_label}  {p['player_a']}  <-->  {p['player_b']}")

        if p["status"] == "MISSING":
            lines.append(f"  STATUS : SKIPPED — missing: {', '.join(p['missing'])}")
        else:
            cos = p["cosine_sim"]
            r_ab = p["rank_b_in_a"]
            r_ba = p["rank_a_in_b"]

            lines.append(f"  cosine similarity : {cos:.4f}")
            lines.append(
                f"  rank of {_short(p['player_b']):20s} in {_short(p['player_a']):20s} neighbours : "
                f"{r_ab:>4d} / {n_players}"
            )
            lines.append(
                f"  rank of {_short(p['player_a']):20s} in {_short(p['player_b']):20s} neighbours : "
                f"{r_ba:>4d} / {n_players}"
            )
            avg = (r_ab + r_ba) / 2
            if avg <= 5:
                verdict = "STRONG MATCH"
            elif avg <= 10:
                verdict = "GOOD MATCH"
            elif avg <= 20:
                verdict = "MODERATE"
            elif avg <= 50:
                verdict = "WEAK"
            else:
                verdict = "NOT MATCHED"
            lines.append(f"  verdict : {verdict}  (avg rank {avg:.0f})")

        if p.get("note"):
            lines.append(f"  note    : {p['note']}")
        lines.append(f"  source  : {p['source']}")
        lines.append("")

    # ── Summary statistics ────────────────────────────────────────
    lines.append("-" * 78)
    lines.append("  AGGREGATE STATISTICS")
    lines.append("-" * 78)
    lines.append("")

    if n_eval > 0:
        lines.append(f"  Overall (across {n_eval} evaluated pairs, {n_eval * 2} directional queries):")
        lines.append(f"    Mean cosine similarity : {summary['mean_cosine']:.4f}")
        lines.append(f"    Mean rank              : {summary['mean_rank']:.1f}")
        lines.append(f"    Median rank            : {summary['median_rank']}")
        lines.append(f"    Hit@5                  : {summary['hit_at_5']:.1%}")
        lines.append(f"    Hit@10                 : {summary['hit_at_10']:.1%}")
        lines.append(f"    Hit@20                 : {summary['hit_at_20']:.1%}")
        lines.append(f"    Hit@50                 : {summary['hit_at_50']:.1%}")
        lines.append("")

        for tier_key, tier_stats in summary.get("by_tier", {}).items():
            lines.append(f"  {tier_key.replace('_', ' ').title()} ({tier_stats['n_pairs']} pairs, "
                         f"{tier_stats['n_pairs'] * 2} queries):")
            lines.append(f"    Mean rank  : {tier_stats['mean_rank']:.1f}")
            lines.append(f"    Median rank: {tier_stats['median_rank']}")
            lines.append(f"    Hit@5      : {tier_stats['hit_at_5']:.1%}")
            lines.append(f"    Hit@10     : {tier_stats['hit_at_10']:.1%}")
            lines.append(f"    Hit@20     : {tier_stats['hit_at_20']:.1%}")
            lines.append("")
    else:
        lines.append("  No pairs could be evaluated (all players missing).")
        lines.append("")

    lines.append("=" * 78)

    report_path = output_dir / "ground_truth_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _short(name: str, max_len: int = 20) -> str:
    """Truncate a name for tabular display."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 1] + "."
