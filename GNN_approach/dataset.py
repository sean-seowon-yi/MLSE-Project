"""
PyTorch Dataset for Player Similarity Training.

This module provides:
1. Custom Dataset class for graph data with masking
2. Data loading and batching utilities
3. Train/val/test splitting functionality
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict, Optional
import numpy as np
import random
from pathlib import Path

from config import DataConfig, TrainingConfig, get_config
from graph_assembly import GraphSnapshot, create_masked_graph


class PlayerGraphDataset(Dataset):
    """
    Dataset for player similarity training with masking.
    
    Each sample is a graph snapshot with one randomly masked player.
    The model's task is to predict the masked player's position.
    
    Attributes:
        graphs: List of GraphSnapshot objects
        mask_all_players: If True, create one sample per player per graph
    """
    
    def __init__(
        self, 
        graphs: List[GraphSnapshot],
        mask_all_players: bool = False,
        random_mask: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            graphs: List of graph snapshots
            mask_all_players: If True, each graph produces 22 samples (one per player)
            random_mask: If True, randomly select player to mask (used if mask_all_players=False)
            seed: Random seed for reproducibility
        """
        self.graphs = graphs
        self.mask_all_players = mask_all_players
        self.random_mask = random_mask
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Pre-compute sample indices
        if mask_all_players:
            # Create index mapping: sample_idx -> (graph_idx, player_idx)
            self.sample_map = []
            for graph_idx, graph in enumerate(graphs):
                num_players = graph.data.num_nodes
                for player_idx in range(num_players):
                    self.sample_map.append((graph_idx, player_idx))
        else:
            self.sample_map = None
    
    def __len__(self) -> int:
        if self.mask_all_players:
            return len(self.sample_map)
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample with masking applied.
        
        Returns:
            Dictionary containing:
            - 'data': Masked PyG Data object
            - 'target': True position of masked player [2]
            - 'mask_idx': Index of masked player
            - 'player_id': ID of masked player
            - 'player_role': Role of masked player
            - 'match_id': Source match ID
            - 'frame_number': Source frame number
        """
        if self.mask_all_players:
            graph_idx, mask_idx = self.sample_map[idx]
        else:
            graph_idx = idx
            if self.random_mask:
                mask_idx = random.randint(0, self.graphs[graph_idx].data.num_nodes - 1)
            else:
                mask_idx = 0  # Always mask first player
        
        graph = self.graphs[graph_idx]
        
        # Create masked version
        masked_data, target = create_masked_graph(graph, mask_idx)
        
        return {
            'data': masked_data,
            'target': target,
            'mask_idx': mask_idx,
            'player_id': graph.player_ids[mask_idx],
            'player_role': graph.player_roles[mask_idx],
            'team_id': graph.team_ids[mask_idx],
            'match_id': graph.match_id,
            'frame_number': graph.frame_number
        }


class EmbeddingDataset(Dataset):
    """
    Dataset for embedding generation (inference mode).
    
    No masking is applied - used to generate embeddings for all players.
    """
    
    def __init__(self, graphs: List[GraphSnapshot]):
        self.graphs = graphs
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a graph without masking.
        
        Returns:
            Dictionary with data and metadata
        """
        graph = self.graphs[idx]
        
        return {
            'data': graph.data,
            'player_ids': graph.player_ids,
            'player_roles': graph.player_roles,
            'team_ids': graph.team_ids,
            'match_id': graph.match_id,
            'frame_number': graph.frame_number
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching graph data.
    
    Handles variable-sized graphs using PyG's Batch utility.
    """
    # Separate data and metadata
    data_list = [item['data'] for item in batch]
    
    # Batch graphs using PyG
    batched_data = Batch.from_data_list(data_list)
    
    # Collect targets and metadata
    targets = torch.stack([item['target'] for item in batch])
    mask_indices = [item['mask_idx'] for item in batch]
    player_ids = [item['player_id'] for item in batch]
    player_roles = [item['player_role'] for item in batch]
    team_ids = [item['team_id'] for item in batch]
    match_ids = [item['match_id'] for item in batch]
    frame_numbers = [item['frame_number'] for item in batch]
    
    # Compute global mask indices in batched graph
    # In a batched graph, nodes are concatenated, so we need to offset indices
    global_mask_indices = []
    offset = 0
    for i, data in enumerate(data_list):
        global_mask_indices.append(offset + mask_indices[i])
        offset += data.num_nodes
    
    return {
        'data': batched_data,
        'target': targets,
        'mask_idx': mask_indices,
        'global_mask_idx': global_mask_indices,
        'player_id': player_ids,
        'player_role': player_roles,
        'team_id': team_ids,
        'match_id': match_ids,
        'frame_number': frame_numbers
    }


def collate_fn_inference(batch: List[Dict]) -> Dict:
    """Collate function for inference (no masking)."""
    data_list = [item['data'] for item in batch]
    batched_data = Batch.from_data_list(data_list)
    
    return {
        'data': batched_data,
        'player_ids': [item['player_ids'] for item in batch],
        'player_roles': [item['player_roles'] for item in batch],
        'team_ids': [item['team_ids'] for item in batch],
        'match_id': [item['match_id'] for item in batch],
        'frame_number': [item['frame_number'] for item in batch],
        'batch_sizes': [item['data'].num_nodes for item in batch]
    }


def train_val_test_split(
    graphs: List[GraphSnapshot],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_by_match: bool = True
) -> Tuple[List[GraphSnapshot], List[GraphSnapshot], List[GraphSnapshot]]:
    """
    Split graphs into train/val/test sets.
    
    Args:
        graphs: List of all graphs
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1)
        seed: Random seed
        split_by_match: If True, split by match to avoid data leakage
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    np.random.seed(seed)
    
    if split_by_match:
        # Group graphs by match
        match_graphs = {}
        for graph in graphs:
            match_id = graph.match_id
            if match_id not in match_graphs:
                match_graphs[match_id] = []
            match_graphs[match_id].append(graph)
        
        # Shuffle matches
        match_ids = list(match_graphs.keys())
        random.shuffle(match_ids)
        
        # Split matches
        n_matches = len(match_ids)
        n_train = int(n_matches * train_ratio)
        n_val = int(n_matches * val_ratio)
        
        train_matches = match_ids[:n_train]
        val_matches = match_ids[n_train:n_train + n_val]
        test_matches = match_ids[n_train + n_val:]
        
        # Collect graphs
        train_graphs = [g for m in train_matches for g in match_graphs[m]]
        val_graphs = [g for m in val_matches for g in match_graphs[m]]
        test_graphs = [g for m in test_matches for g in match_graphs[m]]
    else:
        # Simple random split
        indices = list(range(len(graphs)))
        random.shuffle(indices)
        
        n_train = int(len(graphs) * train_ratio)
        n_val = int(len(graphs) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        test_graphs = [graphs[i] for i in test_indices]
    
    return train_graphs, val_graphs, test_graphs


def create_data_loaders(
    train_graphs: List[GraphSnapshot],
    val_graphs: List[GraphSnapshot],
    test_graphs: List[GraphSnapshot],
    batch_size: int = 64,
    num_workers: int = 0,
    mask_all_players: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_graphs, val_graphs, test_graphs: Split graph lists
        batch_size: Batch size
        num_workers: Number of data loading workers
        mask_all_players: Whether to mask all players (more samples)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PlayerGraphDataset(
        train_graphs, 
        mask_all_players=mask_all_players,
        random_mask=True
    )
    val_dataset = PlayerGraphDataset(
        val_graphs,
        mask_all_players=False,  # Fixed masking for validation consistency
        random_mask=False
    )
    test_dataset = PlayerGraphDataset(
        test_graphs,
        mask_all_players=False,
        random_mask=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset functionality
    from data_preparation import SkillCornerDataLoader
    from graph_assembly import GraphAssembler
    
    config = get_config()
    
    # Load and process data
    print("Loading data...")
    loader = SkillCornerDataLoader(config.data)
    match_id = loader.matches_info[0]['id']
    frames = loader.process_match(match_id)
    
    print("Building graphs...")
    assembler = GraphAssembler(config.graph)
    graphs = assembler.frames_to_graphs(frames[:500])
    
    # Split data
    print("\nSplitting data...")
    train, val, test = train_val_test_split(
        graphs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_by_match=False  # Single match, so random split
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train, val, test,
        batch_size=32
    )
    
    # Test batch
    print("\n=== Sample Batch ===")
    batch = next(iter(train_loader))
    print(f"Batch data type: {type(batch['data'])}")
    print(f"Batch nodes: {batch['data'].x.shape}")
    print(f"Batch edges: {batch['data'].edge_index.shape}")
    print(f"Targets shape: {batch['target'].shape}")
    print(f"Mask indices: {batch['mask_idx'][:5]}...")
    print(f"Global mask indices: {batch['global_mask_idx'][:5]}...")
    print(f"Player roles: {batch['player_role'][:5]}...")
