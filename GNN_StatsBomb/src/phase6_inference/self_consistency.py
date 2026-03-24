"""
Self-consistency evaluation for player embeddings.

Tests whether the model assigns stable identity by checking if the same
player, observed in different competitions, is recognised as their own
closest match.  Two complementary tests are run:

1. **Competition-split**: For players appearing in >=2 competitions with
   sufficient possessions in each, pool embeddings per competition and
   check self-retrieval rank.

2. **Random-half**: For all players with >=100 possessions (regardless of
   competition count), randomly split possessions 50/50, pool each half,
   and check self-retrieval rank.  This tests embedding stability without
   the confound of competition context shift.

Both tests use **open-set retrieval**: non-testable players whose total
possessions meet the inference threshold are pooled and added to the
retrieval gallery as distractors, so that ranks reflect the full
search population rather than only the testable subset.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from ..config import InferenceConfig, POSITION_GROUPS
from ..phase3_graph.masking import mask_future_info
from ..phase4_model.model import PlayerSimilarityModel

BATCH_SIZE = 32

_POS_TO_GROUP: Dict[str, str] = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _p in _positions:
        _POS_TO_GROUP[_p] = _grp


class SelfConsistencyEvaluator:
    """Evaluate embedding self-consistency across competition splits."""

    def __init__(
        self,
        model: PlayerSimilarityModel,
        device: torch.device,
        config: InferenceConfig,
        min_poss_per_split: int = 50,
        random_half_min_poss: int = 100,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config = config
        self.min_poss_per_split = min_poss_per_split
        self.random_half_min_poss = random_half_min_poss
        self.seed = seed

    # ------------------------------------------------------------------
    # Step 1: encode all graphs, collecting per-possession embeddings
    #         tagged with (player_id, match_id)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_all(
        self, graphs: List[HeteroData]
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Return {player_id: [(match_id, h_player), ...]}."""
        result: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)

        n_batches = (len(graphs) + BATCH_SIZE - 1) // BATCH_SIZE
        for start in tqdm(
            range(0, len(graphs), BATCH_SIZE),
            total=n_batches,
            desc="Encoding possessions",
        ):
            batch_raw = graphs[start : start + BATCH_SIZE]
            pid_maps = [self._extract_player_mappings(g) for g in batch_raw]
            match_ids = [int(g.match_id) for g in batch_raw]

            masked = [self._preprocess(g) for g in batch_raw]
            batched = Batch.from_data_list(masked).to(self.device)
            out = self.model.encode_possession(batched)
            h_player = out["player"].cpu().numpy()

            offsets = np.zeros(len(masked) + 1, dtype=np.int64)
            for i, g in enumerate(masked):
                offsets[i + 1] = offsets[i] + g["player"].x.shape[0]

            for gi, pid_map in enumerate(pid_maps):
                mid = match_ids[gi]
                off = int(offsets[gi])
                for pid, local_node in pid_map.items():
                    gn = off + local_node
                    if gn < h_player.shape[0]:
                        result[pid].append((mid, h_player[gn]))

        return result

    # ------------------------------------------------------------------
    # Step 2: build match_id -> competition_id mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_match_comp_map(metadata: pd.DataFrame) -> Dict[int, int]:
        pairs = metadata[["match_id", "competition_id"]].drop_duplicates()
        return dict(zip(pairs["match_id"].values, pairs["competition_id"].values))

    # ------------------------------------------------------------------
    # Step 3: competition-split test
    # ------------------------------------------------------------------

    def _competition_split_test(
        self,
        raw_embs: Dict[int, List[Tuple[int, np.ndarray]]],
        match_to_comp: Dict[int, int],
        metadata: pd.DataFrame,
    ) -> dict:
        """Split by competition, pool each side, measure self-retrieval.

        Non-testable players (those without two eligible competitions)
        whose total possession count meets ``min_samples_per_player`` are
        pooled into a distractor gallery so that retrieval ranks reflect
        the full inference-time search population.
        """
        pooling = self.model.pooling
        pooling.eval()

        min_poss_distractor = self.config.min_samples_per_player

        player_ids: List[int] = []
        z_a_list: List[np.ndarray] = []
        z_b_list: List[np.ndarray] = []
        player_meta: List[dict] = []

        testable_pids: set[int] = set()
        z_distractor_list: List[np.ndarray] = []

        for pid, emb_list in raw_embs.items():
            by_comp: Dict[int, List[np.ndarray]] = defaultdict(list)
            for mid, vec in emb_list:
                cid = match_to_comp.get(mid)
                if cid is not None:
                    by_comp[cid].append(vec)

            eligible = {
                c: vecs
                for c, vecs in by_comp.items()
                if len(vecs) >= self.min_poss_per_split
            }
            if len(eligible) >= 2:
                sorted_comps = sorted(eligible.keys(), key=lambda c: -len(eligible[c]))
                comp_a, comp_b = sorted_comps[0], sorted_comps[1]
                vecs_a, vecs_b = eligible[comp_a], eligible[comp_b]

                z_a = self._pool(pooling, vecs_a)
                z_b = self._pool(pooling, vecs_b)

                player_ids.append(pid)
                z_a_list.append(z_a)
                z_b_list.append(z_b)
                testable_pids.add(pid)

                name, pos = self._lookup_player(pid, metadata)
                player_meta.append({
                    "player_id": pid,
                    "player_name": name,
                    "position_name": pos,
                    "position_group": _POS_TO_GROUP.get(pos, "Unknown"),
                    "comp_a": int(comp_a),
                    "comp_b": int(comp_b),
                    "n_poss_a": len(vecs_a),
                    "n_poss_b": len(vecs_b),
                })
            else:
                all_vecs = [v for _, v in emb_list]
                if len(all_vecs) >= min_poss_distractor:
                    z_distractor_list.append(self._pool(pooling, all_vecs))

        if not z_a_list:
            return {"n_players": 0}

        Z_a = np.stack(z_a_list)
        Z_b = np.stack(z_b_list)
        Z_distractor = np.stack(z_distractor_list) if z_distractor_list else None

        return self._compute_retrieval_metrics(
            Z_a, Z_b, player_ids, player_meta,
            label="competition_split",
            Z_distractor=Z_distractor,
        )

    # ------------------------------------------------------------------
    # Step 4: random-half test
    # ------------------------------------------------------------------

    def _random_half_test(
        self,
        raw_embs: Dict[int, List[Tuple[int, np.ndarray]]],
        metadata: pd.DataFrame,
    ) -> dict:
        """Random 50/50 split, pool each half, measure self-retrieval.

        Players with fewer than ``random_half_min_poss`` possessions but
        at least ``min_samples_per_player`` are pooled into a distractor
        gallery so that retrieval ranks reflect the full search population.
        """
        pooling = self.model.pooling
        pooling.eval()
        rng = random.Random(self.seed)

        min_poss_distractor = self.config.min_samples_per_player

        player_ids: List[int] = []
        z_a_list: List[np.ndarray] = []
        z_b_list: List[np.ndarray] = []
        player_meta: List[dict] = []
        z_distractor_list: List[np.ndarray] = []

        for pid, emb_list in raw_embs.items():
            n_poss = len(emb_list)

            if n_poss >= self.random_half_min_poss:
                vecs = [v for _, v in emb_list]
                indices = list(range(len(vecs)))
                rng.shuffle(indices)
                mid = len(indices) // 2
                half_a = [vecs[i] for i in indices[:mid]]
                half_b = [vecs[i] for i in indices[mid:]]

                z_a = self._pool(pooling, half_a)
                z_b = self._pool(pooling, half_b)

                player_ids.append(pid)
                z_a_list.append(z_a)
                z_b_list.append(z_b)

                name, pos = self._lookup_player(pid, metadata)
                player_meta.append({
                    "player_id": pid,
                    "player_name": name,
                    "position_name": pos,
                    "position_group": _POS_TO_GROUP.get(pos, "Unknown"),
                    "n_poss_total": len(vecs),
                    "n_poss_a": len(half_a),
                    "n_poss_b": len(half_b),
                })
            elif n_poss >= min_poss_distractor:
                all_vecs = [v for _, v in emb_list]
                z_distractor_list.append(self._pool(pooling, all_vecs))

        if not z_a_list:
            return {"n_players": 0}

        Z_a = np.stack(z_a_list)
        Z_b = np.stack(z_b_list)
        Z_distractor = np.stack(z_distractor_list) if z_distractor_list else None

        return self._compute_retrieval_metrics(
            Z_a, Z_b, player_ids, player_meta,
            label="random_half",
            Z_distractor=Z_distractor,
        )

    # ------------------------------------------------------------------
    # Retrieval metric computation (shared by both tests)
    # ------------------------------------------------------------------

    def _compute_retrieval_metrics(
        self,
        Z_a: np.ndarray,
        Z_b: np.ndarray,
        player_ids: List[int],
        player_meta: List[dict],
        label: str,
        Z_distractor: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute self-retrieval ranks and cosine statistics.

        When *Z_distractor* is provided, the retrieval gallery is extended
        with these additional player embeddings (pooled from non-testable
        players).  This makes the ranking task harder and closer to the
        real inference-time search population.

        Cosine statistics (self-cosine, cross-cosine, margin) are always
        computed from the testable-player matrix only, so they remain
        comparable regardless of whether distractors are present.
        """
        n = Z_a.shape[0]

        # --- Cosine statistics (testable-only, independent of gallery) ---
        sim_ab = cosine_similarity(Z_a, Z_b)  # (n, n)
        self_cosines = np.array([sim_ab[i, i] for i in range(n)])
        off_diag = sim_ab[~np.eye(n, dtype=bool)]
        cross_cosine_mean = float(np.mean(off_diag))

        # --- Build retrieval galleries ---
        # A→B: query Z_a[i], gallery = [Z_b | Z_distractor]
        # B→A: query Z_b[i], gallery = [Z_a | Z_distractor]
        # Self-match target for player i is always at index i.
        if Z_distractor is not None and len(Z_distractor) > 0:
            gallery_b = np.vstack([Z_b, Z_distractor])
            gallery_a = np.vstack([Z_a, Z_distractor])
        else:
            gallery_b = Z_b
            gallery_a = Z_a

        gallery_size = gallery_b.shape[0]

        sim_a_gallery = cosine_similarity(Z_a, gallery_b)  # (n, gallery_size)
        sim_b_gallery = cosine_similarity(Z_b, gallery_a)  # (n, gallery_size)

        ranks_a_to_b = np.zeros(n, dtype=int)
        ranks_b_to_a = np.zeros(n, dtype=int)
        for i in range(n):
            sorted_b = np.argsort(sim_a_gallery[i])[::-1]
            ranks_a_to_b[i] = int(np.where(sorted_b == i)[0][0]) + 1

            sorted_a = np.argsort(sim_b_gallery[i])[::-1]
            ranks_b_to_a[i] = int(np.where(sorted_a == i)[0][0]) + 1

        all_ranks = np.concatenate([ranks_a_to_b, ranks_b_to_a])

        def hit_rate(ranks, k):
            return float(np.mean(ranks <= k))

        summary = {
            "label": label,
            "n_players": n,
            "gallery_size": gallery_size,
            "n_distractors": gallery_size - n,
            "self_cosine_mean": round(float(np.mean(self_cosines)), 4),
            "self_cosine_median": round(float(np.median(self_cosines)), 4),
            "self_cosine_std": round(float(np.std(self_cosines)), 4),
            "cross_cosine_mean": round(cross_cosine_mean, 4),
            "cosine_margin": round(float(np.mean(self_cosines)) - cross_cosine_mean, 4),
            "mean_rank_a_to_b": round(float(np.mean(ranks_a_to_b)), 1),
            "mean_rank_b_to_a": round(float(np.mean(ranks_b_to_a)), 1),
            "mean_rank_all": round(float(np.mean(all_ranks)), 1),
            "median_rank_all": int(np.median(all_ranks)),
            "hit_at_1": round(hit_rate(all_ranks, 1), 4),
            "hit_at_5": round(hit_rate(all_ranks, 5), 4),
            "hit_at_10": round(hit_rate(all_ranks, 10), 4),
            "hit_at_20": round(hit_rate(all_ranks, 20), 4),
            "hit_at_50": round(hit_rate(all_ranks, 50), 4),
        }

        by_group: Dict[str, dict] = {}
        groups = set(m["position_group"] for m in player_meta)
        for grp in sorted(groups):
            idxs = [i for i, m in enumerate(player_meta) if m["position_group"] == grp]
            if not idxs:
                continue
            grp_ranks = np.concatenate([ranks_a_to_b[idxs], ranks_b_to_a[idxs]])
            grp_cos = self_cosines[idxs]
            by_group[grp] = {
                "n_players": len(idxs),
                "self_cosine_mean": round(float(np.mean(grp_cos)), 4),
                "mean_rank": round(float(np.mean(grp_ranks)), 1),
                "median_rank": int(np.median(grp_ranks)),
                "hit_at_1": round(hit_rate(grp_ranks, 1), 4),
                "hit_at_5": round(hit_rate(grp_ranks, 5), 4),
                "hit_at_10": round(hit_rate(grp_ranks, 10), 4),
            }
        summary["by_position_group"] = by_group

        per_player = []
        for i, meta in enumerate(player_meta):
            per_player.append({
                **meta,
                "self_cosine": round(float(self_cosines[i]), 4),
                "rank_a_to_b": int(ranks_a_to_b[i]),
                "rank_b_to_a": int(ranks_b_to_a[i]),
                "avg_rank": round((int(ranks_a_to_b[i]) + int(ranks_b_to_a[i])) / 2, 1),
            })
        per_player.sort(key=lambda x: x["avg_rank"])

        return {"summary": summary, "per_player": per_player}

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(
        self,
        graphs: List[HeteroData],
        metadata: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """Run both self-consistency tests and return combined results."""
        print("\nStep 1/3: Encoding all possession graphs...")
        raw_embs = self._encode_all(graphs)
        print(f"  Collected embeddings for {len(raw_embs)} players")

        print("\nStep 2/3: Competition-split test...")
        match_to_comp = self._build_match_comp_map(metadata)
        comp_result = self._competition_split_test(raw_embs, match_to_comp, metadata)
        cs = comp_result.get("summary", {})
        n_comp = cs.get("n_players", comp_result.get("n_players", 0))
        gallery_comp = cs.get("gallery_size", n_comp)
        print(f"  Testable players: {n_comp}  (gallery: {gallery_comp})")

        print("\nStep 3/3: Random-half test...")
        rand_result = self._random_half_test(raw_embs, metadata)
        rs = rand_result.get("summary", {})
        n_rand = rs.get("n_players", rand_result.get("n_players", 0))
        gallery_rand = rs.get("gallery_size", n_rand)
        print(f"  Testable players: {n_rand}  (gallery: {gallery_rand})")

        result = {
            "competition_split": comp_result,
            "random_half": rand_result,
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_report(result, output_dir)
            json_path = output_dir / "self_consistency_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    # ------------------------------------------------------------------
    # Report writing
    # ------------------------------------------------------------------

    @staticmethod
    def _write_report(result: dict, output_dir: Path) -> None:
        lines: list[str] = []

        for test_key, test_label in [
            ("competition_split", "COMPETITION-SPLIT SELF-CONSISTENCY"),
            ("random_half", "RANDOM-HALF SELF-CONSISTENCY"),
        ]:
            data = result[test_key]
            if "summary" not in data:
                lines.append(f"{'=' * 78}")
                lines.append(f"  {test_label}")
                lines.append(f"{'=' * 78}")
                lines.append(f"  No testable players.")
                lines.append("")
                continue

            s = data["summary"]
            lines.append("=" * 78)
            lines.append(f"  {test_label}")
            lines.append("=" * 78)
            lines.append("")
            lines.append(f"  Testable players           : {s['n_players']}")
            lines.append(f"  Gallery size               : {s['gallery_size']}  ({s['n_players']} testable + {s['n_distractors']} distractors)")
            lines.append(f"  Self-cosine (mean/med/std)  : {s['self_cosine_mean']:.4f} / {s['self_cosine_median']:.4f} / {s['self_cosine_std']:.4f}")
            lines.append(f"  Cross-cosine (mean)         : {s['cross_cosine_mean']:.4f}")
            lines.append(f"  Cosine margin (self - cross): {s['cosine_margin']:.4f}")
            lines.append("")
            lines.append(f"  Self-retrieval ranks ({s['n_players'] * 2} queries, gallery of {s['gallery_size']}):")
            lines.append(f"    Mean rank   : {s['mean_rank_all']:.1f}")
            lines.append(f"    Median rank : {s['median_rank_all']}")
            lines.append(f"    Hit@1       : {s['hit_at_1']:.1%}")
            lines.append(f"    Hit@5       : {s['hit_at_5']:.1%}")
            lines.append(f"    Hit@10      : {s['hit_at_10']:.1%}")
            lines.append(f"    Hit@20      : {s['hit_at_20']:.1%}")
            lines.append(f"    Hit@50      : {s['hit_at_50']:.1%}")
            lines.append("")

            by_grp = s.get("by_position_group", {})
            if by_grp:
                lines.append("  By position group:")
                for grp, gs in by_grp.items():
                    lines.append(
                        f"    {grp:12s} (n={gs['n_players']:>3d}): "
                        f"cos={gs['self_cosine_mean']:.4f}  "
                        f"mean_rank={gs['mean_rank']:.1f}  "
                        f"hit@1={gs['hit_at_1']:.1%}  "
                        f"hit@5={gs['hit_at_5']:.1%}  "
                        f"hit@10={gs['hit_at_10']:.1%}"
                    )
                lines.append("")

            per_player = data.get("per_player", [])
            if per_player:
                lines.append("-" * 78)
                n_show = min(20, len(per_player))
                lines.append(f"  Top {n_show} best self-retrievals:")
                for p in per_player[:n_show]:
                    lines.append(
                        f"    {p['player_name']:30s}  "
                        f"cos={p['self_cosine']:.4f}  "
                        f"rank(A->B)={p['rank_a_to_b']:>3d}  "
                        f"rank(B->A)={p['rank_b_to_a']:>3d}  "
                        f"avg={p['avg_rank']:.0f}"
                    )
                lines.append("")

                lines.append(f"  Bottom {n_show} worst self-retrievals:")
                for p in per_player[-n_show:]:
                    lines.append(
                        f"    {p['player_name']:30s}  "
                        f"cos={p['self_cosine']:.4f}  "
                        f"rank(A->B)={p['rank_a_to_b']:>3d}  "
                        f"rank(B->A)={p['rank_b_to_a']:>3d}  "
                        f"avg={p['avg_rank']:.0f}"
                    )
                lines.append("")

            lines.append("")

        report_path = output_dir / "self_consistency_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(g: HeteroData) -> HeteroData:
        g_masked = g.clone()
        masked_x = mask_future_info(g_masked["event"].x.numpy())
        g_masked["event"].x = torch.tensor(masked_x, dtype=torch.float32)
        return g_masked

    @staticmethod
    def _extract_player_mappings(g: HeteroData) -> Dict[int, int]:
        if ("player", "acts_in", "event") not in g.edge_types:
            return {}
        if not hasattr(g, "event_player_ids"):
            return {}
        edge_index = g[("player", "acts_in", "event")].edge_index
        player_idx, event_idx = edge_index
        event_pids = g.event_player_ids
        pid_to_node: Dict[int, int] = {}
        for p_node, ev in zip(player_idx.tolist(), event_idx.tolist()):
            if ev < len(event_pids):
                pid_to_node[int(event_pids[ev])] = p_node
        return pid_to_node

    def _pool(self, pooling, vecs: List[np.ndarray]) -> np.ndarray:
        t = torch.tensor(np.stack(vecs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = pooling.pool_single(t)
        return z.cpu().numpy()

    @staticmethod
    def _lookup_player(pid: int, metadata: pd.DataFrame) -> Tuple[str, str]:
        rows = metadata[metadata["player_id"] == pid]
        if len(rows) == 0:
            return "", "Unknown"
        name = str(rows.iloc[0].get("player_name", ""))
        pos = str(rows["position_name"].mode().iloc[0])
        return name, pos
