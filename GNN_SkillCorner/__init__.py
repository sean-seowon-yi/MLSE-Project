"""
GNN Player Similarity System.

A Graph Neural Network approach to learning player similarity
from tracking data, inspired by TacticAI principles.

Entry point: run main.py from this directory.
    python main.py --mode all
"""

from src.config import get_config, Config
from src.model import PlayerSimilarityAutoencoder
from src.data_preparation import SkillCornerDataLoader
from src.graph_assembly import GraphAssembler
from src.train import Trainer
from src.inference import EmbeddingGenerator, PlayerProfileBuilder, SimilaritySearcher

__version__ = "1.0.0"
__all__ = [
    "get_config",
    "Config",
    "PlayerSimilarityAutoencoder",
    "SkillCornerDataLoader",
    "GraphAssembler",
    "Trainer",
    "EmbeddingGenerator",
    "PlayerProfileBuilder",
    "SimilaritySearcher",
]
