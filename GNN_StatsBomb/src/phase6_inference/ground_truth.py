"""
Pseudo ground-truth evaluation for player similarity.

Defines externally sourced player-similarity pairs and checks where
the expected partner ranks in the model's nearest-neighbour list.
Produces a formatted report with per-pair rank, cosine similarity,
aggregate hit-rate statistics, and diagnostic visualizations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    # Pairs with broad consensus from analytics, media, and domain experts.
    SimilarityPair(
        tier=1,
        player_a_name="Luka Modrić",
        player_a_id=5463,
        player_b_name="Toni Kroos",
        player_b_id=5574,
        source=(
            "Real Madrid's midfield partnership for a decade; both are "
            "tempo-controlling deep-lying playmakers renowned for press-"
            "resistance and metronomic passing (Managing Madrid; FBref)."
        ),
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Virgil van Dijk",
        player_a_id=3669,
        player_b_name="Rúben Dias",
        player_b_id=5206,
        source=(
            "Both are dominant ball-playing centre-backs who defend through "
            "positioning and anticipation; widely compared in Premier League "
            "analytics (Opta Analyst; The Athletic)."
        ),
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Jordi Alba",
        player_a_id=5211,
        player_b_name="Andrew Robertson",
        player_b_id=3655,
        source="StatsBomb: Robertson is the strongest like-for-like Alba replacement.",
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Trent Alexander-Arnold",
        player_a_id=3664,
        player_b_name="Achraf Hakimi",
        player_b_id=5245,
        source=(
            "StatsBomb recruitment example: Hakimi flagged as very similar "
            "to TAA — both redefine the full-back role with elite passing "
            "range and progressive carries."
        ),
        note="TAA has only 87 possessions in the 360 data (borderline).",
    ),
    SimilarityPair(
        tier=1,
        player_a_name="Aitana Bonmatí",
        player_a_id=15284,
        player_b_name="Alexia Putellas",
        player_b_id=10143,
        source=(
            "Barcelona and Spain's midfield engine; consecutive Ballon d'Or "
            "Féminin winners with overlapping profiles in ball retention, "
            "progressive passing, and spatial intelligence "
            "(Pep Guardiola: 'Bonmatí is like the women's Iniesta'; "
            "Total Football Analysis scouting report)."
        ),
    ),
    # ── Tier 2: good directional expectations ────────────────────
    SimilarityPair(
        tier=2,
        player_a_name="Bukayo Saka",
        player_a_id=22084,
        player_b_name="Ousmane Dembélé",
        player_b_id=5477,
        source=(
            "Both are right-wing dribblers with high take-on rates and "
            "creative final-third output; FBref's statistical similarity "
            "model ranks Dembélé among Saka's closest matches."
        ),
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Harry Kane",
        player_a_id=10955,
        player_b_name="Robert Lewandowski",
        player_b_id=5668,
        source=(
            "Universally compared as the top two pure #9 strikers of "
            "their generation — both combine clinical finishing with "
            "deep link-up play and intelligent movement (BBC Sport; "
            "Opta; StatsBomb shot-profile data)."
        ),
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Florian Wirtz",
        player_a_id=40724,
        player_b_name="Kevin De Bruyne",
        player_b_id=3089,
        source=(
            "Yahoo Sports Euro 2024 preview: 'Germany have their own "
            "Kevin De Bruyne' — both are creative attacking midfielders "
            "who combine elite passing with significant goalscoring threat."
        ),
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Wendie Renard",
        player_a_id=10125,
        player_b_name="Millie Bright",
        player_b_id=4642,
        source=(
            "Both are tall, commanding centre-backs who dominate "
            "aerially and progress the ball from deep; widely regarded "
            "as the two best CBs in women's football (FIFA.com; "
            "Her Football Hub)."
        ),
    ),
    SimilarityPair(
        tier=2,
        player_a_name="Lauren Hemp",
        player_a_id=15555,
        player_b_name="María Caldentey",
        player_b_id=10161,
        source=(
            "Both are creative left-wing forwards with high progressive-"
            "carry volume and chance-creation rates; similar role for "
            "their respective national teams (FBref; PlanetFootball)."
        ),
    ),
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
        player_a_name="Jude Bellingham",
        player_a_id=30714,
        player_b_name="Antoine Griezmann",
        player_b_id=5487,
        source=(
            "Both operate as goal-threat attacking midfielders who arrive "
            "late into the box; Lampard (cited by BBC Sport) and Opta "
            "analytics highlight shared 'goalscoring CM' profiles."
        ),
    ),
    # ── Tier 3: conditional / weaker expectations ────────────────
    SimilarityPair(
        tier=3,
        player_a_name="Jamal Musiala",
        player_a_id=39565,
        player_b_name="Phil Foden",
        player_b_id=4354,
        source=(
            "Both are versatile left-side attackers with exceptional "
            "close-control dribbling; compared by talkSPORT and Yahoo "
            "Sports as the best young attacking talents in Europe."
        ),
        note="Musiala's dribbling is more stepover-heavy; Foden is more positional.",
    ),
    SimilarityPair(
        tier=3,
        player_a_name="Lamine Yamal",
        player_a_id=316046,
        player_b_name="Nico Williams",
        player_b_id=68574,
        source=(
            "Spain's Euro 2024 wide-forward duo; Yamal (RW) and Williams "
            "(LW) are both young, explosive inverted wingers — Goal.com "
            "and FBref scouting similarity scores show overlapping profiles."
        ),
        note="Different sides: Yamal is RW, Williams is LW. Only 252/317 possessions.",
    ),
    SimilarityPair(
        tier=3,
        player_a_name="Manuel Neuer",
        player_a_id=5570,
        player_b_name="Gianluigi Donnarumma",
        player_b_id=7036,
        source=(
            "Both are imposing, distribution-oriented goalkeepers who "
            "play a high line; consecutive Euro GK-of-the-tournament "
            "winners (Neuer Euro 2020 squad, Donnarumma Euro 2020 POTM)."
        ),
        note="Neuer is a more aggressive sweeper; behavioural similarity may be moderate.",
    ),
]


def evaluate_ground_truth(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Optional[Path] = None,
    similarity_metric: str = "cosine",
    gender_map: Optional[Dict[int, str]] = None,
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

    sim_matrix = _compute_similarity_matrix(Z, similarity_metric)

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
            entry["similarity"] = None
            entry["rank_b_in_a"] = None
            entry["rank_a_in_b"] = None
            pair_results.append(entry)
            continue

        cos_sim = float(sim_matrix[idx_a, idx_b])

        gender_mask_a = None
        gender_mask_b = None
        if gender_map:
            gender_a = gender_map.get(pair.player_a_id)
            gender_b = gender_map.get(pair.player_b_id)
            if gender_a:
                gender_mask_a = np.array([
                    gender_map.get(int(pid)) == gender_a for pid in pids
                ])
            if gender_b:
                gender_mask_b = np.array([
                    gender_map.get(int(pid)) == gender_b for pid in pids
                ])

        rank_b_in_a = _get_rank(sim_matrix, idx_a, idx_b, mask=gender_mask_a)
        rank_a_in_b = _get_rank(sim_matrix, idx_b, idx_a, mask=gender_mask_b)

        if gender_mask_a is not None:
            gallery_a = int(gender_mask_a.sum()) - 1
        else:
            gallery_a = len(pids) - 1
        if gender_mask_b is not None:
            gallery_b = int(gender_mask_b.sum()) - 1
        else:
            gallery_b = len(pids) - 1

        entry["status"] = "OK"
        entry["similarity"] = round(cos_sim, 4)
        entry["rank_b_in_a"] = rank_b_in_a
        entry["rank_a_in_b"] = rank_a_in_b
        entry["gallery_size_a"] = gallery_a
        entry["gallery_size_b"] = gallery_b
        pair_results.append(entry)

    summary = _compute_summary(pair_results, len(pids))
    summary["similarity_metric"] = similarity_metric
    summary["gender_filtered"] = gender_map is not None

    result = {"pairs": pair_results, "summary": summary}

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_report(pair_results, summary, len(pids), output_dir,
                       gender_map=gender_map, pids=pids)
        json_path = output_dir / "ground_truth_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        _generate_visualizations(pair_results, summary, output_dir)

    return result


def _compute_similarity_matrix(Z: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return cosine_similarity(Z)
    if metric == "euclidean":
        dist = cdist(Z, Z, metric="euclidean")
        return 1.0 / (1.0 + dist)
    return cosine_similarity(Z)


def _bootstrap_ci_mean(
    values: list,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Percentile bootstrap CI for the mean."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _get_rank(
    sim_matrix: np.ndarray,
    query_idx: int,
    target_idx: int,
    mask: Optional[np.ndarray] = None,
) -> int:
    """Return 1-based rank of target in query's similarity list.

    If *mask* is provided (boolean array, same length as sim row),
    only indices where mask is True are considered.
    """
    sims = sim_matrix[query_idx].copy()
    sims[query_idx] = -np.inf
    if mask is not None:
        sims[~mask] = -np.inf
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
    cosines = [p["similarity"] for p in evaluated]

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

    rank_ci_lo, rank_ci_hi = _bootstrap_ci_mean(all_ranks, seed=42)
    hit10_ci_lo, hit10_ci_hi = _bootstrap_ci_mean(
        [1.0 if r <= 10 else 0.0 for r in all_ranks], seed=137,
    )

    return {
        "n_pairs_total": len(pair_results),
        "n_evaluated": len(evaluated),
        "n_missing": len(missing),
        "n_players_in_embeddings": n_players,
        "mean_similarity": round(float(np.mean(cosines)), 4),
        "mean_rank": round(float(np.mean(all_ranks)), 1),
        "mean_rank_ci_95": [round(rank_ci_lo, 1), round(rank_ci_hi, 1)],
        "median_rank": int(np.median(all_ranks)),
        "hit_at_5": round(hit_rate(all_ranks, 5), 3),
        "hit_at_10": round(hit_rate(all_ranks, 10), 3),
        "hit_at_10_ci_95": [round(hit10_ci_lo, 3), round(hit10_ci_hi, 3)],
        "hit_at_20": round(hit_rate(all_ranks, 20), 3),
        "hit_at_50": round(hit_rate(all_ranks, 50), 3),
        "by_tier": by_tier,
    }


def _write_report(
    pair_results: list, summary: dict, n_players: int, output_dir: Path,
    gender_map: Optional[Dict[int, str]] = None,
    pids: Optional[np.ndarray] = None,
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
    if gender_map and pids is not None:
        n_m = sum(1 for pid in pids if gender_map.get(int(pid)) == "male")
        n_f = sum(1 for pid in pids if gender_map.get(int(pid)) == "female")
        lines.append(f"  male gallery             : {n_m}")
        lines.append(f"  female gallery           : {n_f}")
    lines.append(f"Similarity metric          : {summary.get('similarity_metric', 'cosine')}")
    lines.append(f"Gender filtering           : {'Yes' if summary.get('gender_filtered') else 'No'}")
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
            sim = p["similarity"]
            r_ab = p["rank_b_in_a"]
            r_ba = p["rank_a_in_b"]
            gal_a = p.get("gallery_size_a", n_players - 1)
            gal_b = p.get("gallery_size_b", n_players - 1)

            lines.append(
                f"  {summary.get('similarity_metric', 'cosine')} similarity : {sim:.4f}"
            )
            lines.append(
                f"  rank of {p['player_b']:30s} in {p['player_a']:30s} neighbours : "
                f"{r_ab:>4d} / {gal_a}"
            )
            lines.append(
                f"  rank of {p['player_a']:30s} in {p['player_b']:30s} neighbours : "
                f"{r_ba:>4d} / {gal_b}"
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
        lines.append(
            f"    Mean {summary.get('similarity_metric', 'cosine')} similarity : "
            f"{summary['mean_similarity']:.4f}"
        )
        rank_ci = summary.get("mean_rank_ci_95", [])
        ci_str = f"  95% CI [{rank_ci[0]:.1f}, {rank_ci[1]:.1f}]" if len(rank_ci) == 2 else ""
        lines.append(f"    Mean rank              : {summary['mean_rank']:.1f}{ci_str}")
        lines.append(f"    Median rank            : {summary['median_rank']}")
        lines.append(f"    Hit@5                  : {summary['hit_at_5']:.1%}")
        hit10_ci = summary.get("hit_at_10_ci_95", [])
        ci_str = f"  95% CI [{hit10_ci[0]:.1%}, {hit10_ci[1]:.1%}]" if len(hit10_ci) == 2 else ""
        lines.append(f"    Hit@10                 : {summary['hit_at_10']:.1%}{ci_str}")
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


# ── Visualizations ──────────────────────────────────────────────────────

_TIER_COLORS = {1: "#2563eb", 2: "#f59e0b", 3: "#94a3b8"}
_VERDICT_COLORS = {
    "STRONG MATCH": "#16a34a",
    "GOOD MATCH": "#22c55e",
    "MODERATE": "#eab308",
    "WEAK": "#f97316",
    "NOT MATCHED": "#dc2626",
}


def _generate_visualizations(
    pair_results: list, summary: dict, output_dir: Path
) -> None:
    """Generate all ground-truth diagnostic plots."""
    evaluated = [p for p in pair_results if p["status"] == "OK"]
    if not evaluated:
        return
    _plot_rank_overview(evaluated, output_dir)
    _plot_hit_at_k(summary, output_dir)
    _plot_similarity_vs_rank(evaluated, output_dir)


def _verdict(avg_rank: float) -> str:
    if avg_rank <= 5:
        return "STRONG MATCH"
    if avg_rank <= 10:
        return "GOOD MATCH"
    if avg_rank <= 20:
        return "MODERATE"
    if avg_rank <= 50:
        return "WEAK"
    return "NOT MATCHED"


def _plot_rank_overview(evaluated: list, output_dir: Path) -> None:
    """Horizontal dumbbell chart: A→B and B→A ranks per pair."""
    n = len(evaluated)
    fig, ax = plt.subplots(figsize=(13, max(5, 0.5 * n + 1.5)))

    labels = []
    for i, p in enumerate(evaluated):
        labels.append(f"[T{p['tier']}] {p['player_a']} ↔ {p['player_b']}")

    y = np.arange(n)

    for i, p in enumerate(evaluated):
        r_ab = p["rank_b_in_a"]
        r_ba = p["rank_a_in_b"]
        avg = (r_ab + r_ba) / 2
        verd = _verdict(avg)
        color = _VERDICT_COLORS.get(verd, "#94a3b8")

        ax.plot([r_ab, r_ba], [i, i], "-", color=color, linewidth=2.5,
                alpha=0.5, zorder=1)
        ax.scatter(r_ab, i, s=50, color=_TIER_COLORS.get(p["tier"], "#64748b"),
                   edgecolors="white", linewidth=0.6, zorder=3, marker="o")
        ax.scatter(r_ba, i, s=50, color=_TIER_COLORS.get(p["tier"], "#64748b"),
                   edgecolors="white", linewidth=0.6, zorder=3, marker="s")

        x_right = max(r_ab, r_ba)
        ax.text(x_right + 2, i, f"avg={avg:.0f} {verd}",
                fontsize=7, va="center", color=color, fontweight="bold")

    for k_val, ls, lbl in [(5, "--", "Top-5"), (10, "-.", "Top-10"),
                            (20, ":", "Top-20")]:
        ax.axvline(k_val, color="#cbd5e1", linestyle=ls, linewidth=0.8,
                   zorder=0, label=lbl)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (lower is better)", fontsize=10)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_title("Ground-Truth Pair Ranks\n"
                  "(○ = B in A's neighbours, □ = A in B's neighbours)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "gt_rank_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hit_at_k(summary: dict, output_dir: Path) -> None:
    """Grouped bar chart of hit@k for each tier and overall."""
    by_tier = summary.get("by_tier", {})
    if not by_tier and summary.get("n_evaluated", 0) == 0:
        return

    k_levels = [5, 10, 20]
    tier_labels = sorted(by_tier.keys())
    groups = tier_labels + ["Overall"]
    n_groups = len(groups)
    n_k = len(k_levels)

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_groups), 5))
    x = np.arange(n_groups)
    width = 0.22
    colors = ["#2563eb", "#f59e0b", "#22c55e"]

    for j, k in enumerate(k_levels):
        vals = []
        for t_key in tier_labels:
            vals.append(by_tier[t_key].get(f"hit_at_{k}", 0))
        vals.append(summary.get(f"hit_at_{k}", 0))
        bars = ax.bar(x + j * width, vals, width, label=f"Hit@{k}",
                      color=colors[j], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x + width)
    display_labels = [k.replace("_", " ").title() for k in tier_labels] + ["Overall"]
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel("Hit Rate", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Hit@K by Tier", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "gt_hit_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_vs_rank(evaluated: list, output_dir: Path) -> None:
    """Scatter: cosine similarity vs average bidirectional rank."""
    if len(evaluated) < 2:
        return

    cos_sims = [p["similarity"] for p in evaluated]
    avg_ranks = [(p["rank_b_in_a"] + p["rank_a_in_b"]) / 2 for p in evaluated]
    tiers = [p["tier"] for p in evaluated]

    fig, ax = plt.subplots(figsize=(9, 6))

    for tier in sorted(set(tiers)):
        mask = [t == tier for t in tiers]
        c = [cos_sims[i] for i in range(len(cos_sims)) if mask[i]]
        r = [avg_ranks[i] for i in range(len(avg_ranks)) if mask[i]]
        ax.scatter(c, r, s=70, alpha=0.8, zorder=3,
                   color=_TIER_COLORS.get(tier, "#94a3b8"),
                   edgecolors="white", linewidth=0.6,
                   label=f"Tier {tier}")

    for i, p in enumerate(evaluated):
        ax.annotate(f"{p['player_a']} – {p['player_b']}",
                    (cos_sims[i], avg_ranks[i]),
                    fontsize=5.5, textcoords="offset points", xytext=(5, 3),
                    color="#64748b")

    ax.set_xlabel("Cosine Similarity", fontsize=10)
    ax.set_ylabel("Average Rank (lower = better)", fontsize=10)
    ax.set_title("Similarity vs Retrieval Rank\n"
                  "(ideal: high similarity, low rank)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "gt_similarity_vs_rank.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
