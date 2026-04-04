"""
Empirical Behavioral Fidelity — observed-action policy distance.

Unlike the model-based policy diagnostic (which computes JS divergence
between the model's own predicted action distributions), this module
measures behavioral similarity from *actual observed decisions*.

For each player, events are bucketed by coarse game state
(pitch third × under-pressure).  Within each bucket the empirical
action-type histogram is accumulated.  Pairwise JS divergence between
players is then computed over shared buckets, and Spearman ρ correlates
embedding cosine distance with this empirical policy distance.

This removes the self-referential loop: the metric judges the embedding
against real human decisions, not the model's surrogate policy.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from ..config import EVENT_TYPES, POSITION_GROUPS

# ── Constants ────────────────────────────────────────────────────────────

_ACTION_TYPE_START = 0
_ACTION_TYPE_END = 14

_PITCH_ZONE_START = 104
_PITCH_ZONE_END = 113

_UNDER_PRESSURE_IDX = 96

_N_PITCH_ZONES = 9
_N_ZONES_PER_ROW = 3  # y_zones

_COARSE_ACTION_MAP: Dict[str, str] = {}
for _et in EVENT_TYPES:
    if _et == "Pass":
        _COARSE_ACTION_MAP[_et] = "Pass"
    elif _et == "Carry":
        _COARSE_ACTION_MAP[_et] = "Carry"
    elif _et == "Shot":
        _COARSE_ACTION_MAP[_et] = "Shot"
    elif _et == "Dribble":
        _COARSE_ACTION_MAP[_et] = "Dribble"
    else:
        _COARSE_ACTION_MAP[_et] = "Other"

COARSE_ACTIONS: List[str] = ["Pass", "Carry", "Shot", "Dribble", "Other"]
_COARSE_IDX = {a: i for i, a in enumerate(COARSE_ACTIONS)}

_POS_TO_GROUP: Dict[str, str] = {}
for _group, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _group


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) between two probability vectors."""
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _bootstrap_ci(
    values: List[float],
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Core class ───────────────────────────────────────────────────────────

class EmpiricalBehavioralFidelity:
    """Compute observed-action behavioral fidelity for player embeddings."""

    def __init__(
        self,
        features: np.ndarray,
        metadata,
        Z: np.ndarray,
        player_info,
        gender_map: Optional[Dict[int, str]] = None,
        min_events_per_bucket: int = 15,
        min_shared_buckets: int = 3,
    ):
        self.features = features
        self.metadata = metadata
        self.Z = Z
        self.player_info = player_info
        self.gender_map = gender_map
        self.min_events_per_bucket = min_events_per_bucket
        self.min_shared_buckets = min_shared_buckets

        pids = player_info["player_id"].values
        self._pid_list: List[int] = [int(p) for p in pids]
        self._pid_set = set(self._pid_list)
        self._pid_to_z_idx: Dict[int, int] = {
            int(p): i for i, p in enumerate(pids)
        }

        self._pid_to_group: Dict[int, str] = {}
        for _, row in player_info.iterrows():
            pid = int(row["player_id"])
            pos_name = str(row["position_name"])
            self._pid_to_group[pid] = _POS_TO_GROUP.get(pos_name, "Unknown")

        self._group_to_pids: Dict[str, List[int]] = defaultdict(list)
        for pid, grp in self._pid_to_group.items():
            self._group_to_pids[grp].append(pid)

        self._histograms: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
        self._build_histograms()

    def _build_histograms(self) -> None:
        """Accumulate per-player, per-bucket empirical action histograms."""
        n_events = self.features.shape[0]
        n_coarse = len(COARSE_ACTIONS)

        raw_counts: Dict[int, Dict[Tuple[int, int], np.ndarray]] = defaultdict(
            lambda: defaultdict(lambda: np.zeros(n_coarse, dtype=np.float64))
        )

        meta_pids = self.metadata["player_id"].fillna(-1).astype(np.int64).values
        meta_etypes = self.metadata["event_type"].values

        for i in range(n_events):
            pid = int(meta_pids[i])
            if pid not in self._pid_set:
                continue

            action_type_vec = self.features[i, _ACTION_TYPE_START:_ACTION_TYPE_END]
            action_idx = int(np.argmax(action_type_vec))
            if action_type_vec[action_idx] < 0.5:
                etype_str = str(meta_etypes[i])
                if etype_str in _COARSE_ACTION_MAP:
                    coarse = _COARSE_ACTION_MAP[etype_str]
                else:
                    coarse = "Other"
            else:
                etype_str = EVENT_TYPES[action_idx] if action_idx < len(EVENT_TYPES) else "Other"
                coarse = _COARSE_ACTION_MAP.get(etype_str, "Other")

            coarse_idx = _COARSE_IDX[coarse]

            zone_vec = self.features[i, _PITCH_ZONE_START:_PITCH_ZONE_END]
            zone_flat = int(np.argmax(zone_vec))
            pitch_third = zone_flat // _N_ZONES_PER_ROW

            under_pressure = int(self.features[i, _UNDER_PRESSURE_IDX] > 0.5)

            bucket = (pitch_third, under_pressure)
            raw_counts[pid][bucket][coarse_idx] += 1.0

        for pid, buckets in raw_counts.items():
            self._histograms[pid] = {}
            for bucket, counts in buckets.items():
                total = counts.sum()
                if total >= self.min_events_per_bucket:
                    self._histograms[pid][bucket] = counts / total

    def _empirical_js(self, pid_a: int, pid_b: int) -> Optional[float]:
        """Mean JS divergence across shared valid buckets, or None."""
        hist_a = self._histograms.get(pid_a, {})
        hist_b = self._histograms.get(pid_b, {})
        shared = set(hist_a.keys()) & set(hist_b.keys())
        if len(shared) < self.min_shared_buckets:
            return None
        js_values = [
            _js_divergence(hist_a[b], hist_b[b]) for b in shared
        ]
        return float(np.mean(js_values))

    def _compute_correlation(self) -> Dict:
        """Within-group Spearman ρ between embedding distance and empirical JS."""
        cos_sim = cosine_similarity(self.Z)
        groups = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        results: Dict = {}
        all_js: List[float] = []
        all_cos: List[float] = []
        group_keys: List[str] = []

        for grp in groups:
            base_pids = self._group_to_pids.get(grp, [])
            if not base_pids:
                continue

            if self.gender_map:
                gender_subsets: Dict[str, List[int]] = defaultdict(list)
                for p in base_pids:
                    g = self.gender_map.get(p)
                    if g:
                        gender_subsets[g].append(p)
                sub_groups = [
                    (f"{grp} ({g})", pids)
                    for g, pids in sorted(gender_subsets.items())
                    if len(pids) >= 5
                ]
            else:
                sub_groups = [(grp, base_pids)]

            for sub_label, pids in sub_groups:
                pids_with_hist = [
                    p for p in pids if p in self._histograms
                ]
                if len(pids_with_hist) < 5:
                    continue

                print(f"    {sub_label}: {len(pids_with_hist)} players with histograms ...")

                js_pairs: List[float] = []
                cos_pairs: List[float] = []
                n_skipped = 0

                for i in range(len(pids_with_hist)):
                    for j in range(i + 1, len(pids_with_hist)):
                        pa, pb = pids_with_hist[i], pids_with_hist[j]
                        js_val = self._empirical_js(pa, pb)
                        if js_val is None:
                            n_skipped += 1
                            continue
                        idx_a = self._pid_to_z_idx[pa]
                        idx_b = self._pid_to_z_idx[pb]
                        cos_dist = 1.0 - cos_sim[idx_a, idx_b]
                        js_pairs.append(js_val)
                        cos_pairs.append(cos_dist)

                if len(js_pairs) < 10:
                    print(f"      Only {len(js_pairs)} valid pairs (skipped {n_skipped}), need ≥10")
                    continue

                rho, p_val = spearmanr(js_pairs, cos_pairs)

                results[sub_label] = {
                    "n_players": len(pids_with_hist),
                    "n_valid_pairs": len(js_pairs),
                    "n_skipped_pairs": n_skipped,
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p_val), 6),
                    "mean_empirical_js": round(float(np.mean(js_pairs)), 6),
                    "mean_cosine_dist": round(float(np.mean(cos_pairs)), 4),
                }
                group_keys.append(sub_label)
                all_js.extend(js_pairs)
                all_cos.extend(cos_pairs)

                print(f"      {len(js_pairs):,} valid pairs, ρ = {rho:.4f}, p = {p_val:.4e}")

        if len(all_js) >= 10:
            rho_all, p_all = spearmanr(all_js, all_cos)
            results["overall"] = {
                "n_valid_pairs": len(all_js),
                "spearman_rho": round(float(rho_all), 4),
                "p_value": round(float(p_all), 6),
                "mean_empirical_js": round(float(np.mean(all_js)), 6),
                "mean_cosine_dist": round(float(np.mean(all_cos)), 4),
            }

        return results

    def _compute_substitute_quality(
        self, k: int = 10, seed: int = 42,
    ) -> Dict:
        """Top-K vs random-K empirical JS for query players."""
        rng = np.random.default_rng(seed)
        cos_sim = cosine_similarity(self.Z)
        poss_lookup = dict(zip(
            self.player_info["player_id"].values,
            self.player_info["n_possessions"].values,
        ))

        candidates: List[int] = []
        for grp in ["Defender", "Midfielder", "Forward", "Goalkeeper"]:
            grp_pids = [
                p for p in self._group_to_pids.get(grp, [])
                if p in self._histograms
            ]
            if len(grp_pids) < k + 5:
                continue
            sorted_by_poss = sorted(
                grp_pids, key=lambda p: int(poss_lookup.get(p, 0)),
                reverse=True,
            )
            candidates.extend(sorted_by_poss[:3])

        per_query: List[Dict] = []
        for pid in candidates:
            idx = self._pid_to_z_idx[pid]
            grp = self._pid_to_group[pid]
            pool = [
                p for p in self._group_to_pids[grp]
                if p != pid and p in self._histograms
            ]
            if self.gender_map:
                qg = self.gender_map.get(pid)
                pool = [p for p in pool if self.gender_map.get(p) == qg]
            if len(pool) < k + k:
                continue

            grp_indices = [self._pid_to_z_idx[p] for p in pool]
            sims = cos_sim[idx, grp_indices]
            sorted_order = np.argsort(sims)[::-1]
            top_k_pids = [pool[i] for i in sorted_order[:k]]
            remaining = [pool[i] for i in sorted_order[k:]]
            random_pick = rng.choice(len(remaining), size=k, replace=False)
            random_k_pids = [remaining[i] for i in random_pick]

            js_top = [
                self._empirical_js(pid, p) for p in top_k_pids
            ]
            js_rand = [
                self._empirical_js(pid, p) for p in random_k_pids
            ]
            js_top = [v for v in js_top if v is not None]
            js_rand = [v for v in js_rand if v is not None]
            if not js_top or not js_rand:
                continue

            mean_top = float(np.mean(js_top))
            mean_rand = float(np.mean(js_rand))
            ratio = mean_rand / mean_top if mean_top > 1e-9 else float("nan")

            name_row = self.player_info[self.player_info["player_id"] == pid]
            name = str(name_row.iloc[0]["player_name"]) if len(name_row) else str(pid)

            per_query.append({
                "player_id": pid,
                "player_name": name,
                "position_group": grp,
                "mean_js_top_k": round(mean_top, 6),
                "mean_js_random_k": round(mean_rand, 6),
                "ratio": round(ratio, 4),
                "n_valid_top_k": len(js_top),
                "n_valid_random_k": len(js_rand),
            })

        if not per_query:
            return {"k": k, "n_queries": 0, "per_query": []}

        agg_top = np.mean([q["mean_js_top_k"] for q in per_query])
        agg_rand = np.mean([q["mean_js_random_k"] for q in per_query])
        agg_ratio = float(agg_rand / agg_top) if agg_top > 1e-9 else float("nan")

        ci_lo, ci_hi = _bootstrap_ci(
            [q["ratio"] for q in per_query], rng=rng,
        ) if len(per_query) >= 3 else (float("nan"), float("nan"))

        return {
            "k": k,
            "n_queries": len(per_query),
            "aggregate_mean_js_top_k": round(float(agg_top), 6),
            "aggregate_mean_js_random_k": round(float(agg_rand), 6),
            "aggregate_ratio": round(agg_ratio, 4),
            "ratio_ci_95_lo": round(ci_lo, 4),
            "ratio_ci_95_hi": round(ci_hi, 4),
            "per_query": per_query,
        }

    def _histogram_stats(self) -> Dict:
        """Summary statistics about the histogram construction."""
        total_players = len(self._pid_list)
        players_with_hist = sum(1 for p in self._pid_list if p in self._histograms)
        bucket_counts: List[int] = []
        for pid in self._pid_list:
            if pid in self._histograms:
                bucket_counts.append(len(self._histograms[pid]))

        all_buckets = set()
        for hist in self._histograms.values():
            all_buckets.update(hist.keys())

        return {
            "total_embedded_players": total_players,
            "players_with_valid_histograms": players_with_hist,
            "min_events_per_bucket": self.min_events_per_bucket,
            "min_shared_buckets": self.min_shared_buckets,
            "bucket_definition": "pitch_third x under_pressure (6 buckets)",
            "coarse_action_categories": COARSE_ACTIONS,
            "mean_valid_buckets_per_player": round(
                float(np.mean(bucket_counts)) if bucket_counts else 0, 2
            ),
            "median_valid_buckets_per_player": round(
                float(np.median(bucket_counts)) if bucket_counts else 0, 1
            ),
            "unique_buckets_observed": len(all_buckets),
        }

    def run(self, output_dir, seed: int = 42) -> Dict:
        """Run full empirical behavioral evaluation and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Building empirical action histograms ...")
        stats = self._histogram_stats()
        print(f"  Players with valid histograms: "
              f"{stats['players_with_valid_histograms']}/{stats['total_embedded_players']}")
        print(f"  Mean valid buckets/player: {stats['mean_valid_buckets_per_player']}")

        print("  Computing empirical policy-distance correlation ...")
        correlation = self._compute_correlation()

        print("  Computing empirical substitute quality ...")
        substitute = self._compute_substitute_quality(seed=seed)

        result = {
            "histogram_stats": stats,
            "correlation": correlation,
            "substitute_quality": substitute,
        }

        json_path = output_dir / "empirical_behavioral_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        report_path = output_dir / "empirical_behavioral_report.txt"
        self._write_report(report_path, result)

        _generate_visualizations(result, output_dir)

        return result

    def _write_report(self, path: Path, result: Dict) -> None:
        lines = [
            "=" * 60,
            "EMPIRICAL BEHAVIORAL FIDELITY REPORT",
            "=" * 60,
            "",
            "This metric uses OBSERVED player actions (not model predictions)",
            "to assess whether embedding neighbors behave similarly.",
            "",
        ]

        stats = result["histogram_stats"]
        lines.extend([
            "-" * 60,
            "HISTOGRAM CONSTRUCTION",
            "-" * 60,
            f"  Bucket definition       : {stats['bucket_definition']}",
            f"  Coarse action categories: {', '.join(stats['coarse_action_categories'])}",
            f"  Min events per bucket   : {stats['min_events_per_bucket']}",
            f"  Min shared buckets      : {stats['min_shared_buckets']}",
            f"  Embedded players        : {stats['total_embedded_players']}",
            f"  Players with valid hist : {stats['players_with_valid_histograms']}",
            f"  Mean buckets/player     : {stats['mean_valid_buckets_per_player']}",
            f"  Median buckets/player   : {stats['median_valid_buckets_per_player']}",
            "",
        ])

        corr = result["correlation"]
        lines.extend([
            "-" * 60,
            "1. EMPIRICAL POLICY DISTANCE vs COSINE DISTANCE CORRELATION",
            "-" * 60,
            "",
        ])

        if "overall" in corr:
            ov = corr["overall"]
            lines.append(
                f"  Overall (pooled): Spearman rho = {ov['spearman_rho']:.4f}  "
                f"p = {ov['p_value']:.4e}  ({ov['n_valid_pairs']:,} pairs)"
            )
            lines.append(
                f"           mean empirical JS = {ov['mean_empirical_js']:.6f}  "
                f"mean cosine dist = {ov['mean_cosine_dist']:.4f}"
            )
            lines.append("")

        grp_keys = [k for k in corr if k != "overall"]
        for grp in sorted(grp_keys):
            g = corr[grp]
            lines.append(
                f"  {grp:25s}: rho = {g['spearman_rho']:+.4f}  "
                f"(p = {g['p_value']:.4e})  "
                f"{g['n_players']} players, {g['n_valid_pairs']:,} valid / "
                f"{g['n_skipped_pairs']:,} skipped pairs"
            )

        lines.extend([
            "",
            "  Interpretation:",
            "    This metric is NOT self-referential: it compares embedding",
            "    distance against JS divergence of OBSERVED action distributions",
            "    in matched game states (pitch third x pressure).",
            "    rho > 0.3 : Embedding captures real behavioral structure",
            "    rho 0.1-0.3: Weak signal — embedding partially reflects behavior",
            "    rho ~ 0.0 : Embedding geometry unrelated to observed behavior",
            "",
            "  Caveats:",
            "    - Coarse 6-bucket states do not fully match situations",
            "    - Empirical distributions are noisy (sparse events per bucket)",
            "    - Team-context confounding remains (a player's action mix depends",
            "      on their team's tactical setup, not just their own tendencies)",
            "",
        ])

        sub = result["substitute_quality"]
        lines.extend([
            "-" * 60,
            "2. EMPIRICAL SUBSTITUTE QUALITY (top-K vs random same-group)",
            "-" * 60,
            "",
        ])

        if sub["n_queries"] > 0:
            ci_lo = sub.get("ratio_ci_95_lo", float("nan"))
            ci_hi = sub.get("ratio_ci_95_hi", float("nan"))
            ci_str = ""
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ci_str = f"  95% CI [{ci_lo:.2f}, {ci_hi:.2f}]"
            lines.append(
                f"  K = {sub['k']}, queries = {sub['n_queries']}"
            )
            lines.append(
                f"  Aggregate: JS(top-K) = {sub['aggregate_mean_js_top_k']:.6f}  "
                f"JS(random-K) = {sub['aggregate_mean_js_random_k']:.6f}  "
                f"ratio = {sub['aggregate_ratio']:.4f}{ci_str}"
            )
            lines.append("")
            for q in sub["per_query"]:
                lines.append(
                    f"  {q['player_name']:25s} ({q['position_group'][:3]})  "
                    f"JS(top-K)={q['mean_js_top_k']:.6f}  "
                    f"JS(rand)={q['mean_js_random_k']:.6f}  "
                    f"ratio={q['ratio']:.2f}  "
                    f"[{q['n_valid_top_k']}/{sub['k']} top, "
                    f"{q['n_valid_random_k']}/{sub['k']} rand valid]"
                )
        else:
            lines.append("  No valid queries (insufficient shared buckets).")

        lines.extend([
            "",
            "  Interpretation:",
            "    ratio > 1.5 : Cosine neighbours are substantially better",
            "                  behavioral matches even by observed data",
            "    ratio ~ 1.0 : No empirical advantage",
            "    This is a stronger signal than model-based substitute quality",
            "    because it uses actual observed decisions.",
            "",
        ])

        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))


# ── Visualizations ───────────────────────────────────────────────────────

_GROUP_COLORS = {
    "Goalkeeper": "#ef4444", "Defender": "#3b82f6",
    "Midfielder": "#22c55e", "Forward": "#f97316",
}


def _generate_visualizations(result: Dict, output_dir: Path) -> None:
    _plot_correlation_by_group(result, output_dir)
    _plot_substitute_quality(result, output_dir)


def _plot_correlation_by_group(result: Dict, output_dir: Path) -> None:
    corr = result.get("correlation", {})
    if not corr:
        return

    group_keys = [k for k in corr if k != "overall"]
    if not group_keys:
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(group_keys)), 5))

    labels = []
    rhos = []
    colors = []
    for k in sorted(group_keys):
        labels.append(k)
        rhos.append(corr[k].get("spearman_rho", 0))
        base_grp = k.split(" (")[0]
        colors.append(_GROUP_COLORS.get(base_grp, "#94a3b8"))

    x = np.arange(len(labels))
    bars = ax.bar(x, rhos, color=colors, edgecolor="white", linewidth=0.5,
                  width=0.6)

    for bar, v, k in zip(bars, rhos, sorted(group_keys)):
        p_val = corr[k].get("p_value", 1.0)
        sig = ""
        if p_val < 0.001:
            sig = " ***"
        elif p_val < 0.01:
            sig = " **"
        elif p_val < 0.05:
            sig = " *"
        y_pos = v + 0.01 if v >= 0 else v - 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{v:+.3f}{sig}", ha="center", fontsize=8, fontweight="bold")

    if "overall" in corr:
        ov_rho = corr["overall"].get("spearman_rho", 0)
        ax.axhline(ov_rho, color="#1e293b", linestyle="--", linewidth=1.2,
                   label=f"Overall ρ = {ov_rho:+.3f}")

    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ (empirical JS ↔ cosine dist)", fontsize=10)
    ax.set_title("Empirical Behavioral Fidelity\n"
                 "(observed-action JS vs embedding distance; "
                 "* p<.05, ** p<.01, *** p<.001)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "eb_correlation_by_group.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_substitute_quality(result: Dict, output_dir: Path) -> None:
    sub = result.get("substitute_quality", {})
    per_query = sub.get("per_query", [])
    if not per_query:
        return

    sorted_q = sorted(per_query, key=lambda q: q["ratio"], reverse=True)
    n = len(sorted_q)

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * n + 1.5)))

    labels = []
    ratios = []
    colors = []
    for q in sorted_q:
        labels.append(f"{q['player_name']} ({q['position_group'][:3]})")
        ratios.append(q["ratio"])
        colors.append(_GROUP_COLORS.get(q["position_group"], "#94a3b8"))

    y = np.arange(n)
    bars = ax.barh(y, ratios, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65)
    ax.axvline(1.0, color="#dc2626", linewidth=1.2, linestyle="--",
               label="Ratio = 1 (no advantage)")
    ax.axvline(1.5, color="#16a34a", linewidth=0.8, linestyle=":",
               label="Ratio = 1.5 (substantial)")

    for bar, v in zip(bars, ratios):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8, color="#475569")

    agg = sub.get("aggregate_ratio", 0)
    ci_lo = sub.get("ratio_ci_95_lo", float("nan"))
    ci_hi = sub.get("ratio_ci_95_hi", float("nan"))
    ci_str = ""
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        ci_str = f" [{ci_lo:.2f}, {ci_hi:.2f}]"

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("JS(random) / JS(top-K)  (higher = cosine neighbours are better)",
                  fontsize=9)
    ax.set_title(f"Empirical Substitute Quality: Top-K vs Random\n"
                 f"Aggregate ratio = {agg:.2f}{ci_str}  (using observed actions)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "eb_substitute_quality.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
