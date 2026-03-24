"""
Phase 7 – Report builder (orchestrator).

For each query player:
  1. Find nearest neighbours via cosine similarity (Phase 6 embeddings).
  2. Sample real game situations from the query player's events.
  3. For each situation, use ``SituationComparator`` to predict what the
     query **and** every candidate would do, holding the state fixed.
  4. Emit a text report + per-situation bar-chart visualisations.
  5. Generate PCA neighbourhood plots.

This directly answers: "If placed in the same situation, would these
players act similarly?"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import Config, POSITION_GROUPS, validate_graph_config_match
from ..phase3_graph import PossessionGraphBuilder
from ..phase3_graph.masking import mask_future_info
from ..phase4_model import PlayerSimilarityModel
from ..phase6_inference import SimilaritySearcher

from .embedding_viz import compute_pca_coords, plot_pca_global, plot_pca_neighbourhood
from .situation_comparison import SituationComparator


_POS_TO_GROUP: Dict[str, str] = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _grp


@dataclass
class ReportConfig:
    num_random_queries: int = 5
    random_seed: int = 42
    query_player_ids: Optional[List[int]] = None
    situations_per_query: int = 5
    top_k_neighbours: int = 5


class ReportBuilder:
    """
    End-to-end Phase 7 analysis runner.
    """

    def __init__(self, config: Config, report_cfg: Optional[ReportConfig] = None):
        self.config = config
        self.rcfg = report_cfg or ReportConfig()
        self.searcher = SimilaritySearcher(config.inference)

    # ── data loading ──────────────────────────────────────────────

    def _load_embeddings(self) -> Tuple[np.ndarray, pd.DataFrame]:
        emb_dir = Path(self.config.inference.embedding_output_dir)
        Z = np.load(emb_dir / "player_embeddings.npy")
        info = pd.read_parquet(emb_dir / "player_info.parquet")
        return Z, info

    def _load_event_metadata(self) -> pd.DataFrame:
        out_dir = Path(self.config.data.output_dir)
        return pd.read_parquet(out_dir / "event_metadata.parquet")

    def _load_event_features(self) -> np.ndarray:
        out_dir = Path(self.config.data.output_dir)
        return np.load(out_dir / "event_features.npy")

    def _load_model(self, device: torch.device) -> PlayerSimilarityModel:
        model = PlayerSimilarityModel(self.config.model, graph_config=self.config.graph)
        ckpt_path = Path(self.config.training.checkpoint_dir) / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def _load_graphs(self) -> List:
        out_dir = Path(self.config.data.output_dir)
        graphs = PossessionGraphBuilder.load(str(out_dir / self.config.graphs_filename))
        validate_graph_config_match(graphs, self.config.graph.split_context_edges)
        return graphs

    # ── query selection ───────────────────────────────────────────

    def _choose_queries(self, player_info: pd.DataFrame) -> List[int]:
        if self.rcfg.query_player_ids:
            available = set(player_info["player_id"].tolist())
            return [p for p in self.rcfg.query_player_ids if p in available]

        rng = np.random.default_rng(self.rcfg.random_seed)
        pids = player_info["player_id"].values
        n = min(self.rcfg.num_random_queries, len(pids))
        idx = rng.choice(len(pids), size=n, replace=False)
        return pids[idx].tolist()

    # ── event sampling ────────────────────────────────────────────

    def _sample_events(
        self,
        meta: pd.DataFrame,
        player_id: int,
        comparator: SituationComparator,
        n: int,
    ) -> List[int]:
        """
        Pick *n* events from *player_id* that exist in the graph lookup
        and are spread across different pitch zones.
        """
        sub = meta[meta["player_id"] == player_id]
        if sub.empty:
            return []

        valid_idx = [
            int(i)
            for i in sub.index
            if int(i) in comparator._event_lookup
        ]
        if not valid_idx:
            return []

        if len(valid_idx) <= n:
            return valid_idx

        rng = np.random.default_rng(self.rcfg.random_seed + int(player_id))
        return rng.choice(valid_idx, size=n, replace=False).tolist()

    # ── report generation for one query ───────────────────────────

    def _build_single_report(
        self,
        query_pid: int,
        Z: np.ndarray,
        player_info: pd.DataFrame,
        meta: pd.DataFrame,
        sim_matrix: np.ndarray,
        comparator: SituationComparator,
        pid_to_coord: Dict[int, np.ndarray],
        output_dir: Path,
    ) -> None:
        results = self.searcher.find_similar_players(
            query_player_id=query_pid,
            Z=Z,
            player_info=player_info,
            similarity_matrix=sim_matrix,
            top_k=self.rcfg.top_k_neighbours,
        )
        if results.empty:
            return

        neighbour_pids = [int(x) for x in results["player_id"].tolist()]
        all_pids = [query_pid] + neighbour_pids

        q_row = player_info[player_info["player_id"] == query_pid].iloc[0]
        q_name = q_row.get("player_name", "")
        q_pos = q_row.get("position_name", "")
        q_group = _POS_TO_GROUP.get(q_pos, "Other")

        lines: List[str] = []
        lines.append(f"Query: {q_name}  (id={query_pid})")
        lines.append(f"Position: {q_pos} ({q_group})")
        lines.append(f"Possessions: {int(q_row.get('n_possessions', 0))}")
        lines.append("")
        lines.append("NEAREST NEIGHBOURS (cosine similarity):")
        for _, row in results.iterrows():
            cg = _POS_TO_GROUP.get(row.get("position_name", ""), "Other")
            lines.append(
                f"  {row['player_name']:30s}  pos={row['position_name']:20s}  "
                f"sim={row['similarity']:.4f}  possessions={int(row.get('n_possessions', 0))}"
            )
        lines.append("")

        # ── situation-level comparisons ──
        event_indices = self._sample_events(
            meta, query_pid, comparator, self.rcfg.situations_per_query,
        )

        if event_indices:
            lines.append("=" * 62)
            lines.append("SITUATION-LEVEL COMPARISON")
            lines.append(
                "For each real game event from the query player, we predict"
            )
            lines.append(
                "what the query AND each candidate would do in that same state."
            )
            lines.append("=" * 62)

            for sit_num, ev_idx in enumerate(event_indices, 1):
                result = comparator.compare(ev_idx, all_pids, meta)
                if result is None:
                    continue

                lines.append(f"\n--- Situation {sit_num} ---")
                lines.extend(comparator.format_comparison(result))

                fname_action = f"situation_{query_pid}_s{sit_num}_actions.png"
                comparator.plot_comparison(result, output_dir, fname_action)

                fname_dir = f"situation_{query_pid}_s{sit_num}_direction.png"
                comparator.plot_direction_comparison(result, output_dir, fname_dir)
        else:
            lines.append("(No events found in graph for this query player.)")

        # ── save text report ──
        report_path = output_dir / f"report_{query_pid}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # ── PCA neighbourhood ──
        plot_pca_neighbourhood(
            query_pid, neighbour_pids, pid_to_coord, player_info, output_dir,
        )

    # ── public entry point ────────────────────────────────────────

    def run(self, output_dir: Optional[Path] = None) -> None:
        """Run the full Phase 7 analysis."""
        if output_dir is None:
            output_dir = Path(self.config.inference.embedding_output_dir) / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading embeddings …")
        Z, player_info = self._load_embeddings()
        if Z.size == 0 or player_info.empty:
            print("No embeddings found. Run Phase 6 first.")
            return

        print("Loading event metadata + features …")
        meta = self._load_event_metadata()
        features = self._load_event_features()

        print("Loading model + graphs …")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        graphs = self._load_graphs()
        model = self._load_model(device)

        print("Setting up situation comparator …")
        comparator = SituationComparator(
            model=model,
            device=device,
            graphs=graphs,
            Z=Z,
            player_info=player_info,
            event_features=features,
        )

        sim_matrix = self.searcher.compute_similarity_matrix(Z)

        print("Computing PCA coordinates …")
        pid_to_coord = compute_pca_coords(Z, player_info)

        print("Generating global PCA plots (group, subgroup, position) …")
        plot_pca_global(Z, player_info, output_dir, pid_to_coord=pid_to_coord)

        query_pids = self._choose_queries(player_info)
        print(f"Query players ({len(query_pids)}): {query_pids}")

        for pid in query_pids:
            name = player_info.loc[
                player_info["player_id"] == pid, "player_name"
            ].values
            name = name[0] if len(name) > 0 else str(pid)
            print(f"\n  Analysing {name} (id={pid}) …")
            self._build_single_report(
                query_pid=pid,
                Z=Z,
                player_info=player_info,
                meta=meta,
                sim_matrix=sim_matrix,
                comparator=comparator,
                pid_to_coord=pid_to_coord,
                output_dir=output_dir,
            )

        print(f"\nAll reports saved to {output_dir}")
