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
from typing import Dict, List, Tuple

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
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.Z = Z
        self.player_info = player_info
        self.graphs = graphs
        self.metadata = metadata

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
            if global_idx >= len(meta):
                continue
            row = meta.iloc[global_idx]
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
        self, dists: Dict[str, np.ndarray]
    ) -> Dict:
        """Compute within-group JS distance and correlate with cosine distance."""
        cos_sim = cosine_similarity(self.Z)

        groups = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        results = {}

        all_js = []
        all_cos_dist = []

        for grp in groups:
            pids = self._group_to_pids.get(grp, [])
            if len(pids) < 5:
                continue

            indices = [self._pid_to_z_idx[p] for p in pids]
            n = len(indices)

            print(f"    {grp}: {n} players, {n*(n-1)//2:,} pairs ...")
            js_mat = self._build_js_matrix(dists, indices)

            triu_r, triu_c = np.triu_indices(n, k=1)
            js_pairs = js_mat[triu_r, triu_c]

            idx_arr = np.array(indices)
            cos_sub = cos_sim[np.ix_(idx_arr, idx_arr)]
            cos_pairs = 1.0 - cos_sub[triu_r, triu_c]

            valid = np.isfinite(js_pairs) & np.isfinite(cos_pairs)
            js_pairs = js_pairs[valid]
            cos_pairs = cos_pairs[valid]

            if len(js_pairs) < 10:
                continue

            rho, pval = spearmanr(js_pairs, cos_pairs)

            results[grp] = {
                "n_players": n,
                "n_pairs": int(len(js_pairs)),
                "spearman_rho": round(float(rho), 4),
                "p_value": float(pval),
                "mean_js": round(float(js_pairs.mean()), 4),
                "mean_cosine_dist": round(float(cos_pairs.mean()), 4),
            }

            all_js.extend(js_pairs.tolist())
            all_cos_dist.extend(cos_pairs.tolist())

        if len(all_js) >= 10:
            rho, pval = spearmanr(all_js, all_cos_dist)
            results["overall"] = {
                "n_pairs": len(all_js),
                "spearman_rho": round(float(rho), 4),
                "p_value": float(pval),
                "mean_js": round(float(np.mean(all_js)), 4),
                "mean_cosine_dist": round(float(np.mean(all_cos_dist)), 4),
            }

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
        """For n_queries players, compare JS of top-K neighbours vs random."""
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
            candidates.extend(sorted_by_poss[:per_grp])

        candidates = candidates[:n_queries]

        per_query = []
        for pid in candidates:
            idx = self._pid_to_z_idx[pid]
            grp = self._pid_to_group[pid]
            grp_pids = self._group_to_pids[grp]
            grp_indices = [self._pid_to_z_idx[p] for p in grp_pids if p != pid]

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

        return {
            "k": k,
            "n_queries": len(per_query),
            "aggregate_mean_js_top_k": round(float(agg_nb), 6),
            "aggregate_mean_js_random_k": round(float(agg_rn), 6),
            "aggregate_ratio": round(agg_ratio, 4),
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
            if len(grp_pids) < 2:
                continue
            grp_indices = np.array([self._pid_to_z_idx[p] for p in grp_pids])
            shuffled_indices = grp_indices.copy()
            rng.shuffle(shuffled_indices)
            while np.any(shuffled_indices == grp_indices):
                rng.shuffle(shuffled_indices)
                # Prevent infinite loop for very small groups
                if len(grp_indices) <= 2:
                    break
            shuffled_Z[grp_indices] = self.Z[shuffled_indices]

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
            "correlation": correlation,
            "substitute_quality": substitute,
            "film_sensitivity": film_sensitivity,
        }

        json_path = output_dir / "policy_diagnostic_results.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        report_path = output_dir / "policy_diagnostic_report.txt"
        self._write_report(report_path, result)

        return result

    def _write_report(self, path: Path, result: Dict) -> None:
        lines = [
            "=" * 60,
            "POLICY DIAGNOSTIC REPORT",
            "=" * 60,
            "",
            f"Situations sampled : {result['n_situations']}",
            f"Players evaluated  : {result['n_players']}",
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
                f"  Overall: Spearman rho = {ov['spearman_rho']:.4f}  "
                f"(p = {ov['p_value']:.2e}, {ov['n_pairs']:,} pairs)"
            )
            lines.append(
                f"           mean JS = {ov['mean_js']:.4f}  "
                f"mean cosine dist = {ov['mean_cosine_dist']:.4f}"
            )
            lines.append("")

        for grp in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            if grp in corr:
                g = corr[grp]
                lines.append(
                    f"  {grp:12s}: rho = {g['spearman_rho']:+.4f}  "
                    f"(p = {g['p_value']:.2e})  "
                    f"{g['n_players']} players, {g['n_pairs']:,} pairs"
                )

        lines.extend([
            "",
            "  Interpretation:",
            "    rho > 0.5 : Cosine distance is a strong proxy for behavioral distance",
            "    rho 0.3-0.5: Moderate — cosine captures some behavioral structure",
            "    rho < 0.3 : Weak — embedding geometry does not reflect behavior well",
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
        lines.append(
            f"  Aggregate: JS(top-K) = {sub['aggregate_mean_js_top_k']:.6f}  "
            f"JS(random-K) = {sub['aggregate_mean_js_random_k']:.6f}  "
            f"ratio = {sub['aggregate_ratio']:.4f}"
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
            ])

        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))
