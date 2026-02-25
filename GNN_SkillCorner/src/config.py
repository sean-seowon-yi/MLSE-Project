"""
Configuration file for GNN Player Similarity System.

This module contains all hyperparameters and configuration settings
for data processing, model architecture, and training.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    skillcorner_base_path: str = "../SkillCorner/data"
    matches_file: str = "matches.json"
    output_dir: str = "./processed_data"
    
    # Sampling parameters
    sampling_rate: int = 10  # Create graph every N frames (10 = 1 second at 10fps)
    
    # Filtering parameters
    min_players_per_team: int = 11  # Require full teams (can lower for red cards)
    require_ball_in_play: bool = True
    
    # Normalization parameters
    pitch_length: float = 105.0  # Standard pitch length in meters
    pitch_width: float = 68.0    # Standard pitch width in meters
    normalize_to_unit: bool = True  # Normalize to [-1, 1] range
    
    # Feature engineering
    sprint_threshold: float = 7.0  # m/s threshold for sprint detection
    
    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    
    # Node features
    node_feature_dim: int = 9  # [x, y, vx, vy, sprint_flag, team_id, dist_to_ball, dist_to_own_goal, dist_to_opp_goal]
    
    # Edge configuration
    fully_connected: bool = True  # Connect all nodes to all others
    include_self_loops: bool = False
    
    # Edge features
    edge_feature_dim: int = 1  # [same_team flag]
    
    # Number of nodes (players)
    num_nodes: int = 22


@dataclass
class ModelConfig:
    """Configuration for GATv2 Autoencoder model."""
    
    # Input dimensions
    input_dim: int = 9  # Must match node_feature_dim
    
    # Encoder (GATv2) configuration
    hidden_dim: int = 64  # Hidden layer dimension
    latent_dim: int = 32  # Final embedding dimension ("Player DNA")
    num_heads: int = 4    # Number of attention heads
    num_encoder_layers: int = 2
    
    # Decoder (MLP) configuration
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    decoder_output_dim: int = 2  # Predict (x, y) position
    
    # Regularization
    dropout: float = 0.1
    
    # Edge feature usage
    use_edge_features: bool = True
    edge_dim: int = 1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Masking strategy
    mask_probability: float = 1.0  # Probability of masking a node (1.0 = always mask one)
    num_masked_nodes: int = 1      # Number of nodes to mask per graph
    
    # Loss function
    loss_type: str = "mse"  # Mean Squared Error for position reconstruction
    variance_loss_weight: float = 0.25  # Weight for embedding variance regularizer
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 10
    
    # Device
    device: str = "cuda"  # Will fallback to cpu if cuda unavailable
    
    # Logging
    log_every_n_batches: int = 100
    wandb_project: Optional[str] = None  # Set to enable W&B logging


@dataclass
class InferenceConfig:
    """Configuration for inference and similarity search."""
    
    # Embedding generation
    embedding_output_dir: str = "./embeddings"
    
    # Aggregation method for player profiles
    aggregation_method: str = "mean"  # Options: mean, median, weighted_mean
    
    # Similarity search
    similarity_metric: str = "cosine"  # Options: cosine, euclidean
    top_k: int = 10  # Number of similar players to retrieve
    
    # Minimum samples for reliable profile
    min_samples_per_player: int = 100  # ~100 seconds of play minimum


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        """Ensure consistency between configs."""
        # Ensure dimensions match
        assert self.graph.node_feature_dim == self.model.input_dim, \
            "Node feature dim must match model input dim"
        assert self.graph.edge_feature_dim == self.model.edge_dim, \
            "Edge feature dim must match model edge dim"
        
        # Create output directories
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.inference.embedding_output_dir, exist_ok=True)


def get_config() -> Config:
    """Factory function to get default configuration."""
    return Config()


if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print("=== Data Configuration ===")
    print(f"  Sampling rate: {config.data.sampling_rate} frames")
    print(f"  Sprint threshold: {config.data.sprint_threshold} m/s")
    
    print("\n=== Graph Configuration ===")
    print(f"  Node features: {config.graph.node_feature_dim}")
    print(f"  Fully connected: {config.graph.fully_connected}")
    
    print("\n=== Model Configuration ===")
    print(f"  Latent dimension: {config.model.latent_dim}")
    print(f"  Attention heads: {config.model.num_heads}")
    
    print("\n=== Training Configuration ===")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
