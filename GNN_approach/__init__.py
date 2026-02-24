"""
GNN Player Similarity System

A Graph Neural Network approach to learning player similarity
from tracking data, inspired by TacticAI principles.
"""

from .config import get_config, Config
from .model import PlayerSimilarityAutoencoder
from .data_preparation import SkillCornerDataLoader
from .graph_assembly import GraphAssembler
from .train import Trainer
from .inference import EmbeddingGenerator, PlayerProfileBuilder, SimilaritySearcher

__version__ = "1.0.0"
__all__ = [
    'get_config',
    'Config',
    'PlayerSimilarityAutoencoder',
    'SkillCornerDataLoader',
    'GraphAssembler',
    'Trainer',
    'EmbeddingGenerator',
    'PlayerProfileBuilder',
    'SimilaritySearcher'
]
