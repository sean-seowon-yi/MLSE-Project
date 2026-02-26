"""
Phase 6-A: Generate global player trait embeddings ``z_p``.

Procedure:
  1. Run the trained GNN encoder over batched possession graphs.
  2. For each possession, extract the player-node embeddings.
  3. Group per-possession embeddings by player_id.
  4. Aggregate via attention-weighted pooling → one ``z_p`` per player.
  5. Filter players below a minimum sample threshold.
  6. Save the embedding matrix and metadata to disk.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from ..config import InferenceConfig, get_config
from ..phase3_graph.masking import mask_future_info
from ..phase4_model.model import PlayerSimilarityModel
from ..phase4_model.pooling import AttentionPooling


INFERENCE_BATCH_SIZE = 32


class EmbeddingGenerator:
    """
    Generates global player embeddings from a trained model.
    """

    def __init__(
        self,
        model: PlayerSimilarityModel,
        device: torch.device,
        config: InferenceConfig,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.config = config

    def _preprocess_graph(self, g: HeteroData) -> HeteroData:
        """Apply future-info masking to a graph (same as during training)."""
        g_masked = g.clone()
        masked_x = mask_future_info(g_masked["event"].x.numpy())
        g_masked["event"].x = torch.tensor(masked_x, dtype=torch.float32)
        return g_masked

    def _extract_player_mappings(self, g: HeteroData) -> Dict[int, int]:
        """Return {player_id: player_node_idx} for one graph."""
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

    @torch.no_grad()
    def generate(
        self,
        graphs: List[HeteroData],
        metadata: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute ``z_p`` for every player with sufficient data.

        Uses mini-batch processing with PyG's Batch for ~10-50x speedup
        over single-graph iteration.
        """
        player_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        bs = INFERENCE_BATCH_SIZE
        n_batches = (len(graphs) + bs - 1) // bs

        for batch_start in tqdm(range(0, len(graphs), bs),
                                total=n_batches,
                                desc="Encoding possessions (batched)"):
            batch_graphs_raw = graphs[batch_start : batch_start + bs]

            pid_maps = [self._extract_player_mappings(g) for g in batch_graphs_raw]
            masked_graphs = [self._preprocess_graph(g) for g in batch_graphs_raw]

            batched = Batch.from_data_list(masked_graphs)
            batched = batched.to(self.device)

            out_dict = self.model.encode_possession(batched)
            h_player_all = out_dict["player"].cpu().numpy()

            player_offsets = np.zeros(len(masked_graphs) + 1, dtype=np.int64)
            for i, g in enumerate(masked_graphs):
                player_offsets[i + 1] = player_offsets[i] + g["player"].x.shape[0]

            for graph_idx, pid_map in enumerate(pid_maps):
                offset = int(player_offsets[graph_idx])
                for pid, local_node in pid_map.items():
                    global_node = offset + local_node
                    if global_node < h_player_all.shape[0]:
                        player_embeddings[pid].append(h_player_all[global_node])

        # Attention-pool per player
        pooling = self.model.pooling
        pooling.eval()

        player_ids_out: List[int] = []
        z_list: List[np.ndarray] = []

        for pid, emb_list in player_embeddings.items():
            if len(emb_list) < self.config.min_samples_per_player:
                continue
            emb_tensor = torch.tensor(
                np.stack(emb_list), dtype=torch.float32
            ).to(self.device)
            z_i = pooling.pool_single(emb_tensor).cpu().numpy()
            z_list.append(z_i)
            player_ids_out.append(pid)

        if not z_list:
            return np.empty((0, 0)), pd.DataFrame()

        Z = np.stack(z_list, axis=0)  # (n_players, d)

        # Build player info
        info_rows = []
        for pid in player_ids_out:
            name = ""
            pos = "Unknown"
            if metadata is not None and "player_id" in metadata.columns:
                rows = metadata[metadata["player_id"] == pid]
                if len(rows) > 0:
                    name = str(rows.iloc[0].get("player_name", ""))
                    pos = str(rows["position_name"].mode().iloc[0])
            info_rows.append({
                "player_id": pid,
                "player_name": name,
                "position_name": pos,
                "n_possessions": len(player_embeddings[pid]),
            })

        player_info = pd.DataFrame(info_rows)
        return Z, player_info

    def save(
        self,
        Z: np.ndarray,
        player_info: pd.DataFrame,
        output_dir: str,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(str(out / "player_embeddings.npy"), Z)
        player_info.to_parquet(str(out / "player_info.parquet"), index=False)
        print(f"Saved {Z.shape[0]} player embeddings to {out}/")
