"""
GNN Player Similarity — core package.

Use from project root (GNN_SkillCorner/) as:
    from src.config import get_config, Config
    from src.model import PlayerSimilarityAutoencoder
    ...
"""

from .config import get_config, Config
from .model import PlayerSimilarityAutoencoder
from .data_preparation import SkillCornerDataLoader, FrameData, PlayerFrame
from .graph_assembly import GraphAssembler, GraphSnapshot, save_graphs, load_graphs
from .dataset import train_val_test_split, create_data_loaders, PlayerGraphDataset, EmbeddingDataset
from .train import Trainer, compute_reconstruction_metrics
from .inference import (
    EmbeddingGenerator,
    PlayerProfileBuilder,
    SimilaritySearcher,
    RoleValidator,
    save_profiles,
    load_profiles,
)

__all__ = [
    "get_config",
    "Config",
    "PlayerSimilarityAutoencoder",
    "SkillCornerDataLoader",
    "FrameData",
    "PlayerFrame",
    "GraphAssembler",
    "GraphSnapshot",
    "save_graphs",
    "load_graphs",
    "train_val_test_split",
    "create_data_loaders",
    "PlayerGraphDataset",
    "EmbeddingDataset",
    "Trainer",
    "compute_reconstruction_metrics",
    "EmbeddingGenerator",
    "PlayerProfileBuilder",
    "SimilaritySearcher",
    "RoleValidator",
    "save_profiles",
    "load_profiles",
]
