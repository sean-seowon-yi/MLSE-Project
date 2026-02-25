"""
GNN StatsBomb Player Similarity System.

Entry point: run main.py from this directory.
    python main.py --mode prepare
"""

from src.config import get_config, Config
from src.data_preparation import StatsBombDataLoader
from src.feature_encoder import EventFeatureEncoder
