"""
Policy Distance Diagnostic & Substitute-Specific Evaluation.

Two complementary diagnostics that test whether cosine similarity in the
learned embedding space corresponds to actual behavioral similarity:

1. **Policy Distance Correlation** — Sample canonical game situations,
   compute predicted action distributions for all players via FiLM
   conditioning, measure pairwise Jensen-Shannon divergence, and
   correlate with pairwise cosine distance.  High correlation means
   cosine neighbours genuinely behave alike.

2. **Substitute Quality** — For a set of query players, compare the
   behavioral similarity (JS divergence) of their top-K cosine
   neighbours versus random same-position-group players.  A ratio > 1
   means cosine retrieval finds better behavioral matches than chance.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..config import POSITIONS, POSITION_GROUPS
from ..phase3_graph.masking import mask_future_info
from ..phase4_model import PlayerSimilarityModel

_POS_NAME_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}


def _derangement(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a derangement (no fixed points) of *arr*.

    For n=2, always returns the swap.  For n>=3, uses rejection sampling
    with an expected ~e/(e-1) ≈ 1.58 attempts.
    """
    n = len(arr)
    if n == 2:
        return arr[::-1].copy()
    result = arr.copy()
    for _ in range(1000):
        rng.shuffle(result)
        if not np.any(result == arr):
            return result
    return np.roll(arr, 1)


def _mantel_permutation_p(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Mantel-style permutation test for correlation between two distance matrices.

    Shuffles row/column labels of *dist_b*, re-extracts upper-triangle pairs,
    and computes Spearman rho on each permutation.  Returns the proportion of
    permuted rhos >= observed rho (one-sided positive test).
    """
    n = dist_a.shape[0]
    triu_r, triu_c = np.triu_indices(n, k=1)
    a_pairs = dist_a[triu_r, triu_c]

    observed_rho, _ = spearmanr(a_pairs, dist_b[triu_r, triu_c])

    if not np.isfinite(observed_rho):
        return float("nan")

    count_ge = 0
    idx = np.arange(n)
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        b_perm = dist_b[np.ix_(perm, perm)]
        perm_rho, _ = spearmanr(a_pairs, b_perm[triu_r, triu_c])
        if np.isfinite(perm_rho) and perm_rho >= observed_rho:
            count_ge += 1

    return (count_ge + 1) / (n_perm + 1)


def _bootstrap_ci(
    values: List[float],
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of *values*."""
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


def _holm_bonferroni_correct(
    results: Dict, group_keys: List[str]
) -> None:
    """Apply Holm-Bonferroni correction in-place to group-level permutation p-values."""
    pvals = []
    for key in group_keys:
        if key in results and "p_value_permutation" in results[key]:
            p = results[key]["p_value_permutation"]
            if isinstance(p, float) and not np.isfinite(p):
                continue
            pvals.append((p, key))
    if not pvals:
        return
    pvals.sort()
    m = len(pvals)
    prev_adj = 0.0
    for rank_i, (p, key) in enumerate(pvals):
        adjusted = min(1.0, p * (m - rank_i))
        adjusted = max(adjusted, prev_adj)
        prev_adj = adjusted
        results[key]["p_value_holm"] = round(adjusted, 6)

_POS_TO_GROUP: Dict[str, str] = {}
for _group, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _group


class PolicyDiagnostic:
    """Runs policy-distance and substitute-quality diagnostics."""

    def __init__(
        self,
        model: PlayerSimilarityModel,
        device: torch.device,
        Z: np.ndarray,
        player_info,
        graphs: list,
        metadata,
        gender_map: Optional[Dict[int, str]] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.Z = Z
        self.player_info = player_info
        self.graphs = graphs
        self.metadata = metadata
        self.gender_map = gender_map

        pids = player_info["player_id"].values
        self._pid_list: List[int] = [int(p) for p in pids]
        self._pid_to_z_idx: Dict[int, int] = {
            int(p): i for i, p in enumerate(pids)
        }

        self._pid_to_group: Dict[int, str] = {}
        self._pid_to_pos_idx: Dict[int, int] = {}
        for _, row in player_info.iterrows():
            pid = int(row["player_id"])
            pos_name = str(row["position_name"])
            self._pid_to_group[pid] = _POS_TO_GROUP.get(pos_name, "Unknown")
            self._pid_to_pos_idx[pid] = _POS_NAME_TO_IDX.get(
                pos_name, len(POSITIONS) - 1
            )

        self._group_to_pids: Dict[str, List[int]] = defaultdict(list)
        for pid, grp in self._pid_to_group.items():
            self._group_to_pids[grp].append(pid)

        self._event_lookup: Dict[int, Tuple[int, int]] = {}
        for g_idx, g in enumerate(graphs):
            if hasattr(g, "event_indices_global"):
                for local, gidx in enumerate(
                    g.event_indices_global.tolist()
                ):
                    self._event_lookup[int(gidx)] = (g_idx, local)

    # ── Phase A: Sample canonical situations ─────────────────────────

    def _sample_situations(
        self, n: int, seed: int
    ) -> List[Tuple[int, int, int]]:
        """Sample *n* situations stratified by actor position group.

        Returns list of (global_event_idx, graph_idx, local_event_idx).
        """
        rng = np.random.default_rng(seed)
        eligible_pids = set(self._pid_list)

        by_group: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        meta = self.metadata
        for global_idx, (g_idx, local_idx) in self._event_lookup.items():
            if global_idx not in meta.index:
                continue
            row = meta.loc[global_idx]
            pid = int(row.get("player_id", -1))
            if pid not in eligible_pids:
                continue
            grp = self._pid_to_group.get(pid, "Unknown")
            if grp == "Unknown":
                continue
            by_group[grp].append((global_idx, g_idx, local_idx))

        groups = [g for g in ["Goalkeeper", "Defender", "Midfielder", "Forward"]
                  if g in by_group]
        per_group = max(1, n // len(groups)) if groups else n
        sampled: List[Tuple[int, int, int]] = []

        for grp in groups:
            pool = by_group[grp]
            k = min(per_group, len(pool))
            chosen = rng.choice(len(pool), size=k, replace=False)
            sampled.extend(pool[i] for i in chosen)

        if len(sampled) > n:
            chosen = rng.choice(len(sampled), size=n, replace=False)
            sampled = [sampled[i] for i in chosen]

        return sampled

    # ── Phase B: Batch FiLM computation ──────────────────────────────

    def _encode_situations(
        self, situations: List[Tuple[int, int, int]]
    ) -> List[torch.Tensor]:
        """Encode each situation graph and extract the target h_event vector.

        Returns a list of (d,) tensors, one per situation, on self.device.
        """
        h_ev_list: List[torch.Tensor] = []

        for global_idx, g_idx, local_idx in tqdm(
            situations, desc="Encoding situations", leave=False
        ):
            g = self.graphs[g_idx]
            g_masked = g.clone()
            masked_x = mask_future_info(g_masked["event"].x.numpy())
            g_masked["event"].x = torch.tensor(
                masked_x, dtype=torch.float32
            )
            g_dev = g_masked.to(self.device)

            with torch.no_grad():
                out = self.model.encode_possession_counterfactual(g_dev)
                h_ev_list.append(out["event"][local_idx])

        return h_ev_list

    def _apply_film_and_heads(
        self,
        h_ev_list: List[torch.Tensor],
        z: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Apply FiLM conditioning and action heads for a given Z matrix.

        Args:
            h_ev_list: per-situation h_event vectors from _encode_situations.
            z: (n_players, d) player embeddings.
            pos_idx: (n_players,) position indices.

        Returns dict with keys 'action_type', 'angle_bin', 'length_bin',
        each of shape (n_situations, n_players, n_classes).
        """
        n_players = z.shape[0]
        at_list, ang_list, ln_list = [], [], []

        with torch.no_grad():
            for h_ev in h_ev_list:
                h_ev_batch = h_ev.unsqueeze(0).expand(n_players, -1)
                h_cond = self.model.film_condition(
                    h_ev_batch, z, pos_idx=pos_idx
                )
                at_list.append(
                    F.softmax(self.model.action_type_head(h_cond), dim=-1)
                    .cpu().numpy()
                )
                ang_list.append(
                    F.softmax(self.model.angle_bin_head(h_cond), dim=-1)
                    .cpu().numpy()
                )
                ln_list.append(
                    F.softmax(self.model.length_bin_head(h_cond), dim=-1)
                    .cpu().numpy()
                )

        return {
            "action_type": np.stack(at_list),
            "angle_bin": np.stack(ang_list),
            "length_bin": np.stack(ln_list),
        }

    def _compute_distributions(
        self, situations: List[Tuple[int, int, int]]
    ) -> Tuple[Dict[str, np.ndarray], List[torch.Tensor]]:
        """Compute action distributions for all players in all situations.

        Returns (dists, h_ev_list) where dists has keys 'action_type',
        'angle_bin', 'length_bin' each of shape (n_situations, n_players,
        n_classes), and h_ev_list is the cached situation encodings.
        """
        z_all = torch.tensor(self.Z, dtype=torch.float32, device=self.device)
        pos_idx_all = torch.tensor(
            [self._pid_to_pos_idx[p] for p in self._pid_list],
            dtype=torch.long,
            device=self.device,
        )
        h_ev_list = self._encode_situations(situations)
        dists = self._apply_film_and_heads(h_ev_list, z_all, pos_idx_all)
        return dists, h_ev_list

    # ── Phase C: Policy distance correlation ─────────────────────────

    def _build_js_matrix(
        self, dists: Dict[str, np.ndarray], indices: List[int]
    ) -> np.ndarray:
        """Build pairwise mean-JS matrix for a subset of players (vectorised).

        indices: player indices into the distributions arrays.
        Returns (n, n) symmetric matrix of mean JS divergences.
        """
        n = len(indices)
        n_sit = dists["action_type"].shape[0]
        js_accum = np.zeros((n, n), dtype=np.float64)

        for key in ("action_type", "angle_bin", "length_bin"):
            arr = dists[key][:, indices, :]  # (n_sit, n, C)
            for s in range(n_sit):
                P = arr[s]  # (n, C)
                eps = 1e-12
                P_safe = np.clip(P, eps, None)
                # (n, 1, C) vs (1, n, C) → M is (n, n, C)
                M = 0.5 * (P_safe[:, None, :] + P_safe[None, :, :])
                kl_pm = np.sum(
                    P_safe[:, None, :] * np.log2(P_safe[:, None, :] / M),
                    axis=-1,
                )
                kl_qm = np.sum(
                    P_safe[None, :, :] * np.log2(P_safe[None, :, :] / M),
                    axis=-1,
                )
                js_accum += 0.5 * (kl_pm + kl_qm)

        return js_accum / (n_sit * 3)

    def _compute_js_correlation(
        self,
        dists: Dict[str, np.ndarray],
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> Dict:
        """Compute within-group JS distance and correlate with cosine distance.

        Uses a Mantel-style row-permutation test instead of the parametric
        Spearman p-value, because distance-matrix pairs share players and
        are therefore non-i.i.d.  Applies Holm-Bonferroni correction across
        all group-level tests.

        When ``self.gender_map`` is set, each position group is further
        split by gender so that correlations are computed within
        same-gender-and-position cohorts.
        """
        perm_rng = np.random.default_rng(seed + 999)
        cos_sim = cosine_similarity(self.Z)

        groups = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        results = {}

        all_js = []
        all_cos_dist = []
        group_keys: List[str] = []

        for grp in groups:
            base_pids = self._group_to_pids.get(grp, [])
            if not base_pids:
                continue

            if self.gender_map:
                gender_subsets = defaultdict(list)
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
                if len(pids) < 5:
                    continue

                indices = [self._pid_to_z_idx[p] for p in pids]
                n = len(indices)

                print(f"    {sub_label}: {n} players, {n*(n-1)//2:,} pairs ...")
                js_mat = self._build_js_matrix(dists, indices)

                triu_r, triu_c = np.triu_indices(n, k=1)
                js_pairs = js_mat[triu_r, triu_c]

                idx_arr = np.array(indices)
                cos_sub = cos_sim[np.ix_(idx_arr, idx_arr)]
                cos_dist_mat = 1.0 - cos_sub
                cos_pairs = cos_dist_mat[triu_r, triu_c]

                valid = np.isfinite(js_pairs) & np.isfinite(cos_pairs)
                js_pairs = js_pairs[valid]
                cos_pairs = cos_pairs[valid]

                if len(js_pairs) < 10:
                    continue

                rho, _ = spearmanr(js_pairs, cos_pairs)

                perm_p = _mantel_permutation_p(
                    js_mat, cos_dist_mat, n_permutations, perm_rng,
                )

                results[sub_label] = {
                    "n_players": n,
                    "n_pairs": int(len(js_pairs)),
                    "spearman_rho": round(float(rho), 4),
                    "p_value_permutation": round(float(perm_p), 6),
                    "n_permutations": n_permutations,
                    "mean_js": round(float(js_pairs.mean()), 4),
                    "mean_cosine_dist": round(float(cos_pairs.mean()), 4),
                }
                group_keys.append(sub_label)

                all_js.extend(js_pairs.tolist())
                all_cos_dist.extend(cos_pairs.tolist())

        if len(all_js) >= 10:
            rho, _ = spearmanr(all_js, all_cos_dist)
            results["overall"] = {
                "n_pairs": len(all_js),
                "spearman_rho": round(float(rho), 4),
                "p_value_note": "pooled from within-group pairs; not independently testable",
                "mean_js": round(float(np.mean(all_js)), 4),
                "mean_cosine_dist": round(float(np.mean(all_cos_dist)), 4),
            }

        _holm_bonferroni_correct(results, group_keys)

        return results

    # ── Phase D: Substitute quality ──────────────────────────────────

    def _js_between_one_and_many(
        self,
        dists: Dict[str, np.ndarray],
        query_idx: int,
        target_indices: List[int],
    ) -> np.ndarray:
        """Vectorised mean JS between one query and multiple targets.

        Returns (len(target_indices),) array of mean JS divergences.
        """
        n_sit = dists["action_type"].shape[0]
        n_targets = len(target_indices)
        js_accum = np.zeros(n_targets, dtype=np.float64)
        eps = 1e-12

        for key in ("action_type", "angle_bin", "length_bin"):
            arr = dists[key]  # (n_sit, n_players, C)
            q = np.clip(arr[:, query_idx, :], eps, None)   # (n_sit, C)
            t = np.clip(arr[:, target_indices, :], eps, None)  # (n_sit, n_t, C)
            q_exp = q[:, None, :]  # (n_sit, 1, C)
            M = 0.5 * (q_exp + t)  # (n_sit, n_t, C)
            kl_qm = np.sum(q_exp * np.log2(q_exp / M), axis=-1)  # (n_sit, n_t)
            kl_tm = np.sum(t * np.log2(t / M), axis=-1)  # (n_sit, n_t)
            js_accum += np.sum(0.5 * (kl_qm + kl_tm), axis=0)

        return js_accum / (n_sit * 3)

    def _evaluate_substitutes(
        self,
        dists: Dict[str, np.ndarray],
        n_queries: int,
        k: int = 10,
        seed: int = 42,
    ) -> Dict:
        """For n_queries players, compare JS of top-K neighbours vs random.

        When ``self.gender_map`` is set, top-K neighbours and random
        baselines are restricted to same-gender players within each
        position group.
        """
        rng = np.random.default_rng(seed)
        cos_sim = cosine_similarity(self.Z)

        poss_lookup = dict(zip(
            self.player_info["player_id"].values,
            self.player_info["n_possessions"].values,
        ))

        candidates = []
        for grp in ["Defender", "Midfielder", "Forward", "Goalkeeper"]:
            grp_pids = self._group_to_pids.get(grp, [])
            if len(grp_pids) < k + 5:
                continue
            sorted_by_poss = sorted(
                grp_pids, key=lambda p: int(poss_lookup.get(p, 0)),
                reverse=True,
            )
            per_grp = max(1, n_queries // 4)
            n_top = (per_grp + 1) // 2
            n_rand = per_grp - n_top
            top_pids = sorted_by_poss[:n_top]
            remaining = sorted_by_poss[n_top:]
            if remaining and n_rand > 0:
                rand_pick = rng.choice(
                    len(remaining),
                    size=min(n_rand, len(remaining)),
                    replace=False,
                )
                rand_pids = [remaining[i] for i in rand_pick]
            else:
                rand_pids = []
            candidates.extend(top_pids + rand_pids)

        candidates = candidates[:n_queries]

        per_query = []
        for pid in candidates:
            idx = self._pid_to_z_idx[pid]
            grp = self._pid_to_group[pid]
            grp_pids = self._group_to_pids[grp]

            if self.gender_map:
                query_gender = self.gender_map.get(pid)
                pool = [
                    p for p in grp_pids
                    if p != pid and self.gender_map.get(p) == query_gender
                ]
            else:
                pool = [p for p in grp_pids if p != pid]

            grp_indices = [self._pid_to_z_idx[p] for p in pool]

            if len(grp_indices) < k + k:
                continue

            sims = cos_sim[idx, grp_indices]
            sorted_order = np.argsort(sims)[::-1]
            top_k_indices = [grp_indices[i] for i in sorted_order[:k]]

            remaining = [grp_indices[i] for i in sorted_order[k:]]
            random_pick = rng.choice(len(remaining), size=k, replace=False)
            random_k_indices = [remaining[i] for i in random_pick]

            js_nb = self._js_between_one_and_many(dists, idx, top_k_indices)
            js_rn = self._js_between_one_and_many(dists, idx, random_k_indices)

            mean_nb = float(np.nanmean(js_nb))
            mean_rn = float(np.nanmean(js_rn))
            ratio = mean_rn / mean_nb if mean_nb > 1e-9 else float("nan")

            name_row = self.player_info[self.player_info["player_id"] == pid]
            name = str(name_row.iloc[0]["player_name"]) if len(name_row) else str(pid)

            per_query.append({
                "player_id": pid,
                "player_name": name,
                "position_group": grp,
                "mean_js_top_k": round(mean_nb, 6),
                "mean_js_random_k": round(mean_rn, 6),
                "ratio": round(ratio, 4),
            })

        agg_nb = np.mean([q["mean_js_top_k"] for q in per_query]) if per_query else 0
        agg_rn = np.mean([q["mean_js_random_k"] for q in per_query]) if per_query else 0
        agg_ratio = float(agg_rn / agg_nb) if agg_nb > 1e-9 else float("nan")

        ci_lo, ci_hi = _bootstrap_ci(
            [q["ratio"] for q in per_query], rng=rng,
        ) if len(per_query) >= 3 else (float("nan"), float("nan"))

        return {
            "k": k,
            "n_queries": len(per_query),
            "gender_filtered": self.gender_map is not None,
            "aggregate_mean_js_top_k": round(float(agg_nb), 6),
            "aggregate_mean_js_random_k": round(float(agg_rn), 6),
            "aggregate_ratio": round(agg_ratio, 4),
            "ratio_ci_95_lo": round(ci_lo, 4),
            "ratio_ci_95_hi": round(ci_hi, 4),
            "per_query": per_query,
        }

    # ── Phase E: FiLM sensitivity ───────────────────────────────────

    def _compute_film_sensitivity(
        self,
        dists_correct: Dict[str, np.ndarray],
        h_ev_list: List[torch.Tensor],
        seed: int = 42,
    ) -> Dict:
        """Measure how much z_p changes predictions (FiLM effect size).

        For each player, replace their z_p with a random same-group
        player's z_p and measure JS divergence against the correct
        predictions.  High JS = FiLM is load-bearing.
        """
        rng = np.random.default_rng(seed + 777)
        n_players = len(self._pid_list)

        shuffled_Z = self.Z.copy()
        groups = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        for grp in groups:
            grp_pids = self._group_to_pids.get(grp, [])
            if not grp_pids:
                continue

            if self.gender_map:
                gender_subsets: Dict[str, List[int]] = defaultdict(list)
                for p in grp_pids:
                    g = self.gender_map.get(p)
                    if g:
                        gender_subsets[g].append(p)
                subsets = list(gender_subsets.values())
            else:
                subsets = [grp_pids]

            for subset in subsets:
                if len(subset) < 2:
                    continue
                sub_indices = np.array([self._pid_to_z_idx[p] for p in subset])
                shuffled_indices = _derangement(sub_indices, rng)
                shuffled_Z[sub_indices] = self.Z[shuffled_indices]

        z_shuf = torch.tensor(shuffled_Z, dtype=torch.float32, device=self.device)
        pos_idx_all = torch.tensor(
            [self._pid_to_pos_idx[p] for p in self._pid_list],
            dtype=torch.long,
            device=self.device,
        )

        print("    Computing shuffled distributions ...")
        dists_shuffled = self._apply_film_and_heads(h_ev_list, z_shuf, pos_idx_all)

        n_sit = dists_correct["action_type"].shape[0]
        eps = 1e-12
        js_per_player = np.zeros(n_players, dtype=np.float64)

        for key in ("action_type", "angle_bin", "length_bin"):
            P = np.clip(dists_correct[key], eps, None)  # (n_sit, n_players, C)
            Q = np.clip(dists_shuffled[key], eps, None)
            M = 0.5 * (P + Q)
            kl_pq = np.sum(P * np.log2(P / M), axis=-1)  # (n_sit, n_players)
            kl_qp = np.sum(Q * np.log2(Q / M), axis=-1)
            js_per_player += np.sum(0.5 * (kl_pq + kl_qp), axis=0)

        js_per_player /= (n_sit * 3)

        per_group = {}
        for grp in groups:
            grp_pids = self._group_to_pids.get(grp, [])
            if not grp_pids:
                continue
            grp_indices = [self._pid_to_z_idx[p] for p in grp_pids]
            grp_js = js_per_player[grp_indices]
            per_group[grp] = {
                "n_players": len(grp_indices),
                "mean_js": round(float(grp_js.mean()), 6),
                "median_js": round(float(np.median(grp_js)), 6),
                "std_js": round(float(grp_js.std()), 6),
                "min_js": round(float(grp_js.min()), 6),
                "max_js": round(float(grp_js.max()), 6),
            }

        return {
            "overall_mean_js": round(float(js_per_player.mean()), 6),
            "overall_median_js": round(float(np.median(js_per_player)), 6),
            "overall_std_js": round(float(js_per_player.std()), 6),
            "per_group": per_group,
        }

    # ── Orchestrator ─────────────────────────────────────────────────

    def run(
        self,
        output_dir,
        n_situations: int = 200,
        n_query_players: int = 10,
        seed: int = 42,
    ) -> Dict:
        """Run full diagnostic and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Sampling {n_situations} canonical situations ...")
        situations = self._sample_situations(n_situations, seed)
        print(f"  Sampled {len(situations)} situations")

        print("  Computing action distributions for all players ...")
        dists, h_ev_list = self._compute_distributions(situations)
        print(f"  Distributions: {dists['action_type'].shape}")

        print("  Computing policy-distance correlation ...")
        correlation = self._compute_js_correlation(dists)

        print("  Evaluating substitute quality ...")
        substitute = self._evaluate_substitutes(
            dists, n_query_players, seed=seed
        )

        print("  Computing FiLM sensitivity ...")
        film_sensitivity = self._compute_film_sensitivity(
            dists, h_ev_list, seed=seed
        )

        result = {
            "n_situations": len(situations),
            "n_players": len(self._pid_list),
            "gender_filtered": self.gender_map is not None,
            "correlation": correlation,
            "substitute_quality": substitute,
            "film_sensitivity": film_sensitivity,
        }

        json_path = output_dir / "policy_diagnostic_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        report_path = output_dir / "policy_diagnostic_report.txt"
        self._write_report(report_path, result)

        _generate_visualizations(result, output_dir)

        return result

    def _write_report(self, path: Path, result: Dict) -> None:
        lines = [
            "=" * 60,
            "POLICY DIAGNOSTIC REPORT",
            "=" * 60,
            "",
            f"Situations sampled : {result['n_situations']}",
            f"Players evaluated  : {result['n_players']}",
            f"Gender filtering   : {'Yes' if result.get('gender_filtered') else 'No'}",
            "",
            "-" * 60,
            "1. POLICY DISTANCE vs COSINE DISTANCE CORRELATION",
            "-" * 60,
            "",
        ]

        corr = result["correlation"]
        if "overall" in corr:
            ov = corr["overall"]
            lines.append(
                f"  Overall (pooled): Spearman rho = {ov['spearman_rho']:.4f}  "
                f"({ov['n_pairs']:,} pairs)"
            )
            lines.append(
                f"           mean JS = {ov['mean_js']:.4f}  "
                f"mean cosine dist = {ov['mean_cosine_dist']:.4f}"
            )
            lines.append("")

        grp_keys = [k for k in corr if k != "overall"]
        for grp in sorted(grp_keys):
            g = corr[grp]
            p_str = f"p_perm = {g['p_value_permutation']:.4f}"
            if "p_value_holm" in g:
                p_str += f", p_holm = {g['p_value_holm']:.4f}"
            lines.append(
                f"  {grp:25s}: rho = {g['spearman_rho']:+.4f}  "
                f"({p_str})  "
                f"{g['n_players']} players, {g['n_pairs']:,} pairs"
            )

        lines.extend([
            "",
            "  Interpretation:",
            "    rho > 0.5 : Cosine distance is a strong proxy for behavioral distance",
            "    rho 0.3-0.5: Moderate — cosine captures some behavioral structure",
            "    rho < 0.3 : Weak — embedding geometry does not reflect behavior well",
            "",
            "  Statistical notes:",
            "    - p-values are from Mantel-style row-permutation tests (not parametric),",
            "      because distance-matrix pairs share players and violate i.i.d. assumptions.",
            "    - Group-level p-values are Holm-Bonferroni corrected for multiple testing.",
            "    - Overall rho is pooled from within-group pairs (descriptive, not tested).",
            "",
            "-" * 60,
            "2. SUBSTITUTE QUALITY (top-K neighbours vs random same-group)",
            "-" * 60,
            "",
        ])

        sub = result["substitute_quality"]
        lines.append(
            f"  K = {sub['k']}, queries = {sub['n_queries']}"
        )
        ci_lo = sub.get("ratio_ci_95_lo", float("nan"))
        ci_hi = sub.get("ratio_ci_95_hi", float("nan"))
        ci_str = ""
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ci_str = f"  95% CI [{ci_lo:.2f}, {ci_hi:.2f}]"
        lines.append(
            f"  Aggregate: JS(top-K) = {sub['aggregate_mean_js_top_k']:.6f}  "
            f"JS(random-K) = {sub['aggregate_mean_js_random_k']:.6f}  "
            f"ratio = {sub['aggregate_ratio']:.4f}{ci_str}"
        )
        lines.append("")

        for q in sub["per_query"]:
            lines.append(
                f"  {q['player_name']:25s} ({q['position_group']:10s})  "
                f"JS(top-K)={q['mean_js_top_k']:.6f}  "
                f"JS(rand)={q['mean_js_random_k']:.6f}  "
                f"ratio={q['ratio']:.2f}"
            )

        lines.extend([
            "",
            "  Interpretation:",
            "    ratio > 1.5 : Cosine neighbours are substantially better behavioral matches",
            "    ratio 1.1-1.5: Mild advantage — cosine captures some relevant structure",
            "    ratio ~ 1.0 : No advantage — cosine similarity is not finding behavioral matches",
            "    ratio < 1.0 : Cosine neighbours are *worse* than random (pathological)",
            "",
            "  Statistical notes:",
            "    - Query players are selected via a mixed strategy: ~half top-possession,",
            "      ~half random from each position group, to mitigate high-volume bias.",
            "    - 95% bootstrap CI on the aggregate ratio is reported when n >= 3 queries.",
            "",
        ])

        film = result.get("film_sensitivity")
        if film:
            lines.extend([
                "-" * 60,
                "3. FiLM SENSITIVITY (correct z_p vs shuffled same-group z_p)",
                "-" * 60,
                "",
                f"  Overall: mean JS = {film['overall_mean_js']:.6f}  "
                f"median = {film['overall_median_js']:.6f}  "
                f"std = {film['overall_std_js']:.6f}",
                "",
            ])
            for grp in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                if grp in film["per_group"]:
                    g = film["per_group"][grp]
                    lines.append(
                        f"  {grp:12s}: mean = {g['mean_js']:.6f}  "
                        f"median = {g['median_js']:.6f}  "
                        f"range = [{g['min_js']:.6f}, {g['max_js']:.6f}]"
                    )
            lines.extend([
                "",
                "  Interpretation:",
                "    JS > 0.1  : z_p strongly changes predictions — FiLM is load-bearing",
                "    JS 0.01-0.1: Moderate effect — player identity influences but does not dominate",
                "    JS < 0.01 : Weak effect — model may be largely ignoring z_p",
                "",
                "  Statistical notes:",
                "    - Shuffle uses guaranteed derangements (no identity permutations).",
                "    - Groups with < 2 players are excluded from the shuffle.",
                "",
            ])

        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))


# ── Visualizations ──────────────────────────────────────────────────────

_GROUP_COLORS = {
    "Goalkeeper": "#ef4444", "Defender": "#3b82f6",
    "Midfielder": "#22c55e", "Forward": "#f97316",
}


def _generate_visualizations(result: Dict, output_dir: Path) -> None:
    """Generate all policy-diagnostic plots."""
    _plot_correlation_by_group(result, output_dir)
    _plot_substitute_quality(result, output_dir)
    _plot_film_sensitivity(result, output_dir)


def _plot_correlation_by_group(result: Dict, output_dir: Path) -> None:
    """Bar chart: Spearman rho (JS vs cosine distance) per position group."""
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
        p_val = corr[k].get("p_value_holm", corr[k].get("p_value_permutation"))
        sig = ""
        if p_val is not None and np.isfinite(p_val):
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
    ax.set_ylabel("Spearman ρ (JS dist ↔ cosine dist)", fontsize=10)
    ax.set_title("Policy Distance Correlation\n"
                  "(positive ρ = cosine captures behavioural structure; "
                  "* p<.05, ** p<.01, *** p<.001)",
                  fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "pd_correlation_by_group.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_substitute_quality(result: Dict, output_dir: Path) -> None:
    """Horizontal bar: ratio of JS(random) / JS(top-K) per query player."""
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
    ax.set_title(f"Substitute Quality: Top-K vs Random\n"
                 f"Aggregate ratio = {agg:.2f}{ci_str}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "pd_substitute_quality.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_film_sensitivity(result: Dict, output_dir: Path) -> None:
    """Bar chart: FiLM effect size (mean JS) per position group."""
    film = result.get("film_sensitivity")
    if not film or not film.get("per_group"):
        return

    groups_order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    present = [g for g in groups_order if g in film["per_group"]]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(present))
    means = [film["per_group"][g]["mean_js"] for g in present]
    stds = [film["per_group"][g]["std_js"] for g in present]
    colors = [_GROUP_COLORS.get(g, "#94a3b8") for g in present]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55,
                  error_kw={"elinewidth": 1.2, "capsize": 4, "color": "#64748b"})

    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

    ax.axhline(0.1, color="#16a34a", linestyle="--", linewidth=0.8,
               label="JS = 0.1 (strong effect)")
    ax.axhline(0.01, color="#eab308", linestyle=":", linewidth=0.8,
               label="JS = 0.01 (weak threshold)")

    overall = film.get("overall_mean_js", 0)
    ax.axhline(overall, color="#1e293b", linestyle="-.", linewidth=1,
               label=f"Overall mean = {overall:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=10)
    ax.set_ylabel("Mean JS Divergence (correct z vs shuffled z)", fontsize=10)
    ax.set_title("FiLM Sensitivity: How Much Does Player Identity Matter?\n"
                 "(higher JS = predictions change more when z_p is swapped)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "pd_film_sensitivity.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
