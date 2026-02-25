"""
PyTorch Dataset and utilities for training on possession graphs.

Key responsibilities:
  - Wrap HeteroData graphs into an indexable Dataset.
  - Apply ``mask_future_info`` to event features at sample time
    (the original features are needed for target extraction).
  - Compute action targets on-the-fly via ``ActionTargetEncoder``.
  - Provide a collate function compatible with PyG batching of HeteroData.
  - Train / val / test splitting by match_id to prevent data leakage.
"""

import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch

from ..config import ModelConfig, TrainingConfig, get_config
from ..phase3_graph.masking import mask_future_info
from .action_targets import ActionTargetEncoder


class PossessionGraphDataset(Dataset):
    """
    Dataset of per-possession heterogeneous graphs.

    Each ``__getitem__`` returns a dict with:
      "data"           : HeteroData (event features future-masked)
      "action_targets" : dict of LongTensors (type, angle, length)
      "outcome_target" : FloatTensor (2,)
    """

    def __init__(
        self,
        graphs: List[HeteroData],
        event_features: np.ndarray,
        model_config: ModelConfig,
    ):
        self.graphs = graphs
        self.event_features = event_features
        self.target_encoder = ActionTargetEncoder(model_config)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict:
        g = self.graphs[idx]

        # Extract un-masked features for target computation
        global_indices = g.event_indices_global.numpy()
        raw_feats = self.event_features[global_indices]  # (T, 122)

        # Compute action targets from un-masked features
        action_targets = self.target_encoder.encode(raw_feats, g.event_types)

        # Apply future-info masking to the event features stored on the graph
        masked_event_x = mask_future_info(g["event"].x.numpy())
        g_copy = g.clone()
        g_copy["event"].x = torch.tensor(masked_event_x, dtype=torch.float32)

        # Outcome target
        outcome = torch.tensor(
            [float(g.ends_in_shot), float(g.ends_in_goal)],
            dtype=torch.float32,
        )

        return {
            "data": g_copy,
            "action_targets": action_targets,
            "outcome_target": outcome,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate for PossessionGraphDataset.

    Uses PyG ``Batch.from_data_list`` which handles HeteroData natively.
    """
    data_list = [item["data"] for item in batch]
    batched_data = Batch.from_data_list(data_list)

    # Stack action targets across the batch (concatenated event dim)
    action_type = torch.cat([item["action_targets"]["action_type"] for item in batch])
    angle_bin = torch.cat([item["action_targets"]["angle_bin"] for item in batch])
    length_bin = torch.cat([item["action_targets"]["length_bin"] for item in batch])

    # Stack outcome targets per possession
    outcome_targets = torch.stack([item["outcome_target"] for item in batch])

    return {
        "data": batched_data,
        "action_targets": {
            "action_type": action_type,
            "angle_bin": angle_bin,
            "length_bin": length_bin,
        },
        "outcome_targets": outcome_targets,
    }


def train_val_test_split(
    graphs: List[HeteroData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[HeteroData], List[HeteroData], List[HeteroData]]:
    """
    Split graphs by match_id to avoid data leakage.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    match_to_graphs: Dict[int, List[HeteroData]] = {}
    for g in graphs:
        mid = g.match_id
        match_to_graphs.setdefault(mid, []).append(g)

    match_ids = list(match_to_graphs.keys())
    rng.shuffle(match_ids)

    n = len(match_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = match_ids[:n_train]
    val_ids = match_ids[n_train:n_train + n_val]
    test_ids = match_ids[n_train + n_val:]

    train = [g for mid in train_ids for g in match_to_graphs[mid]]
    val = [g for mid in val_ids for g in match_to_graphs[mid]]
    test = [g for mid in test_ids for g in match_to_graphs[mid]]

    return train, val, test
