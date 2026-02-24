"""
Phase 3: Model Architecture

This module implements the Graph Autoencoder with GATv2 encoder.

Architecture:
1. Encoder (GATv2): Creates player embeddings ("Player DNA")
   - Layer 1: Perception - each node attends to all 21 others
   - Layer 2: Compression - condenses to latent representation
   
2. Decoder (MLP): Reconstructs position from embeddings
   - Only used during training
   - Proves the embeddings capture meaningful information

The latent embedding (e.g., 32 dimensions) is the "Player DNA" that
captures tactical role, movement patterns, and spatial behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, List

from config import ModelConfig, get_config


class GATv2Encoder(nn.Module):
    """
    GATv2-based encoder that creates player embeddings.
    
    Uses Graph Attention Networks v2 (GATv2) which fixes the static
    attention problem of the original GAT. GATv2 computes dynamic
    attention that depends on both the query and key nodes.
    
    Architecture:
    - Input: Node features [batch_nodes, input_dim]
    - GATv2 Layer 1: [batch_nodes, hidden_dim * num_heads]
    - GATv2 Layer 2: [batch_nodes, latent_dim]
    
    The output is the "Player DNA" - a dense representation of each
    player's tactical state at this moment.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # First GATv2 layer: Perception
        # Takes raw features, outputs hidden representation
        self.gat1 = GATv2Conv(
            in_channels=config.input_dim,
            out_channels=config.hidden_dim,
            heads=config.num_heads,
            concat=True,  # Concatenate heads
            dropout=config.dropout,
            edge_dim=config.edge_dim if config.use_edge_features else None,
            add_self_loops=False  # We control edges explicitly
        )
        
        # Second GATv2 layer: Compression
        # Compresses to latent space
        self.gat2 = GATv2Conv(
            in_channels=config.hidden_dim * config.num_heads,
            out_channels=config.latent_dim,
            heads=1,  # Single head for final output
            concat=False,
            dropout=config.dropout,
            edge_dim=config.edge_dim if config.use_edge_features else None,
            add_self_loops=False
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(config.hidden_dim * config.num_heads)
        self.norm2 = nn.LayerNorm(config.latent_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            embeddings: Player embeddings [num_nodes, latent_dim]
            attention: Tuple of attention weights (if return_attention=True)
        """
        attention_weights = []
        
        # First GATv2 layer
        if self.config.use_edge_features and edge_attr is not None:
            x, attn1 = self.gat1(
                x, edge_index, edge_attr,
                return_attention_weights=True
            )
        else:
            x, attn1 = self.gat1(
                x, edge_index,
                return_attention_weights=True
            )
        
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        attention_weights.append(attn1)
        
        # Second GATv2 layer
        if self.config.use_edge_features and edge_attr is not None:
            x, attn2 = self.gat2(
                x, edge_index, edge_attr,
                return_attention_weights=True
            )
        else:
            x, attn2 = self.gat2(
                x, edge_index,
                return_attention_weights=True
            )
        
        x = self.norm2(x)
        attention_weights.append(attn2)
        
        if return_attention:
            return x, tuple(attention_weights)
        return x, None


class MLPDecoder(nn.Module):
    """
    MLP decoder for reconstructing player positions.
    
    Takes the latent embedding ("Player DNA") and expands it back
    to predict the original position. This reconstruction task
    forces the encoder to learn meaningful representations.
    
    Architecture:
    - Input: Latent embedding [batch, latent_dim]
    - Hidden layers: [batch, hidden_dims[i]]
    - Output: Predicted position [batch, 2] (x, y)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        layers = []
        input_dim = config.latent_dim
        
        # Hidden layers
        for hidden_dim in config.decoder_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.decoder_output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent embedding to position.
        
        Args:
            z: Latent embedding [batch, latent_dim]
            
        Returns:
            position: Predicted (x, y) position [batch, 2]
        """
        return self.network(z)


class PlayerSimilarityAutoencoder(nn.Module):
    """
    Complete Graph Autoencoder for player similarity learning.
    
    Combines GATv2 encoder and MLP decoder for self-supervised
    learning through position reconstruction.
    
    Training:
    1. Mask one player's position in the input
    2. Encode all players using graph attention
    3. Decode the masked player's embedding
    4. Minimize MSE between prediction and true position
    
    Inference:
    1. Encode all players (no masking)
    2. Extract embeddings as "Player DNA"
    3. Use for similarity search
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.encoder = GATv2Encoder(config)
        self.decoder = MLPDecoder(config)
    
    def encode(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Encode players to latent embeddings.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            embeddings: Player embeddings [num_nodes, latent_dim]
            attention: Attention weights (optional)
        """
        return self.encoder(x, edge_index, edge_attr, return_attention)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent embeddings to positions.
        
        Args:
            z: Latent embeddings [batch, latent_dim]
            
        Returns:
            positions: Predicted positions [batch, 2]
        """
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (encode + decode).
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features (optional)
            mask_idx: Index of masked node (for training)
            
        Returns:
            embeddings: All player embeddings [num_nodes, latent_dim]
            predicted_pos: Predicted position for masked node [1, 2]
            all_predicted: All predicted positions [num_nodes, 2]
        """
        # Encode
        embeddings, _ = self.encoder(x, edge_index, edge_attr)
        
        # Decode all positions
        all_predicted = self.decoder(embeddings)
        
        # Get prediction for masked node if specified
        if mask_idx is not None:
            predicted_pos = all_predicted[mask_idx:mask_idx+1]
        else:
            predicted_pos = all_predicted
        
        return embeddings, predicted_pos, all_predicted
    
    def get_player_embedding(
        self,
        data: Data,
        player_idx: int
    ) -> torch.Tensor:
        """
        Get embedding for a specific player.
        
        Args:
            data: PyG Data object
            player_idx: Index of player
            
        Returns:
            embedding: Player embedding [latent_dim]
        """
        embeddings, _ = self.encoder(
            data.x, data.edge_index, 
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        return embeddings[player_idx]
    
    def get_all_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get embeddings for all players in a graph.
        
        Args:
            data: PyG Data object
            
        Returns:
            embeddings: All player embeddings [num_nodes, latent_dim]
        """
        embeddings, _ = self.encoder(
            data.x, data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        return embeddings


class PositionReconstructionLoss(nn.Module):
    """
    Loss function for position reconstruction.
    
    Computes MSE between predicted and true positions,
    with optional weighting for different axes.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            predicted: Predicted positions [batch, 2]
            target: True positions [batch, 2]
            
        Returns:
            loss: Scalar loss value
        """
        return self.mse(predicted, target)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model architecture
    config = get_config()
    model_config = config.model
    
    # Create model
    model = PlayerSimilarityAutoencoder(model_config)
    print(f"=== Model Architecture ===")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print(f"\n=== Forward Pass Test ===")
    
    # Create dummy data
    num_nodes = 22
    x = torch.randn(num_nodes, model_config.input_dim)
    
    # Create fully connected edge index
    source = []
    target = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                source.append(i)
                target.append(j)
    edge_index = torch.tensor([source, target], dtype=torch.long)
    
    # Edge features
    edge_attr = torch.randn(edge_index.shape[1], model_config.edge_dim)
    
    # Forward pass
    embeddings, predicted_pos, all_predicted = model(
        x, edge_index, edge_attr, mask_idx=0
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Predicted position shape: {predicted_pos.shape}")
    print(f"All predictions shape: {all_predicted.shape}")
    
    # Test loss
    print(f"\n=== Loss Test ===")
    target_pos = torch.randn(1, 2)
    loss_fn = PositionReconstructionLoss()
    loss = loss_fn(predicted_pos, target_pos)
    print(f"Reconstruction loss: {loss.item():.4f}")
    
    # Test attention weights
    print(f"\n=== Attention Weights Test ===")
    embeddings, attention = model.encode(
        x, edge_index, edge_attr, return_attention=True
    )
    print(f"Number of attention layers: {len(attention)}")
    for i, (edge_idx, attn_weights) in enumerate(attention):
        print(f"Layer {i+1}: attention shape {attn_weights.shape}")
