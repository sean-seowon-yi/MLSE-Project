"""
Phase 4: Training Strategy

This module implements the self-supervised training loop using
the masking reconstruction task.

Training Strategy:
1. Take a graph of 22 players
2. Randomly mask one player's position (set to 0,0)
3. Model predicts the masked player's position
4. Minimize MSE between prediction and true position

This forces the model to learn tactical relationships:
- To predict where a Right Back is, it must understand the defensive line
- To predict where a Left Winger is, it must understand attacking patterns
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from config import TrainingConfig, get_config
from model import PlayerSimilarityAutoencoder, PositionReconstructionLoss
from dataset import PlayerGraphDataset, collate_fn


class Trainer:
    """
    Trainer for the Player Similarity Autoencoder.
    
    Handles:
    - Training loop with masking
    - Validation and early stopping
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: PlayerSimilarityAutoencoder,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup loss function
        self.criterion = PositionReconstructionLoss()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            # Move data to device
            data = batch['data'].to(self.device)
            targets = batch['target'].to(self.device)
            global_mask_indices = batch['global_mask_idx']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings, _, all_predictions = self.model(
                data.x, data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
            
            # Get predictions for masked nodes
            predictions = all_predictions[global_mask_indices]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Track per-role performance
        role_losses = {}
        role_counts = {}
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            data = batch['data'].to(self.device)
            targets = batch['target'].to(self.device)
            global_mask_indices = batch['global_mask_idx']
            roles = batch['player_role']
            
            # Forward pass
            embeddings, _, all_predictions = self.model(
                data.x, data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
            
            # Get predictions for masked nodes
            predictions = all_predictions[global_mask_indices]
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Track per-role losses
            for i, role in enumerate(roles):
                if role not in role_losses:
                    role_losses[role] = 0.0
                    role_counts[role] = 0
                
                individual_loss = self.criterion(
                    predictions[i:i+1], targets[i:i+1]
                )
                role_losses[role] += individual_loss.item()
                role_counts[role] += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute average per-role losses
        metrics = {
            'avg_loss': avg_loss,
            'role_losses': {
                role: role_losses[role] / role_counts[role]
                for role in role_losses
            }
        }
        
        return avg_loss, metrics
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history dictionary
        """
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Print per-role losses (top 5 worst)
            role_losses = val_metrics['role_losses']
            sorted_roles = sorted(role_losses.items(), key=lambda x: x[1], reverse=True)
            print("\nHardest roles to predict:")
            for role, loss in sorted_roles[:5]:
                print(f"  {role}: {loss:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                print("  [New best model saved]")
            else:
                self.patience_counter += 1
                print(f"  [No improvement for {self.patience_counter} epochs]")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch, val_loss)
        
        # Save history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def compute_reconstruction_metrics(
    model: PlayerSimilarityAutoencoder,
    loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Compute detailed reconstruction metrics.
    
    Returns metrics like MAE, RMSE, and per-coordinate errors.
    """
    model.eval()
    
    all_errors = []
    x_errors = []
    y_errors = []
    
    with torch.no_grad():
        for batch in loader:
            data = batch['data'].to(device)
            targets = batch['target'].to(device)
            global_mask_indices = batch['global_mask_idx']
            
            # Forward pass
            _, _, all_predictions = model(
                data.x, data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None
            )
            
            predictions = all_predictions[global_mask_indices]
            
            # Compute errors
            errors = torch.abs(predictions - targets)
            all_errors.append(errors)
            x_errors.append(errors[:, 0])
            y_errors.append(errors[:, 1])
    
    all_errors = torch.cat(all_errors)
    x_errors = torch.cat(x_errors)
    y_errors = torch.cat(y_errors)
    
    return {
        'mae': all_errors.mean().item(),
        'rmse': torch.sqrt((all_errors ** 2).mean()).item(),
        'x_mae': x_errors.mean().item(),
        'y_mae': y_errors.mean().item(),
        'x_std': x_errors.std().item(),
        'y_std': y_errors.std().item()
    }


if __name__ == "__main__":
    # Test training loop
    from data_preparation import SkillCornerDataLoader
    from graph_assembly import GraphAssembler
    from dataset import train_val_test_split, create_data_loaders
    
    config = get_config()
    
    # Load data
    print("Loading and processing data...")
    loader = SkillCornerDataLoader(config.data)
    
    # Process first match for testing
    match_id = loader.matches_info[0]['id']
    frames = loader.process_match(match_id)
    
    # Build graphs
    print("Building graphs...")
    assembler = GraphAssembler(config.graph)
    graphs = assembler.frames_to_graphs(frames[:200])  # Small subset for testing
    
    # Split data
    print("Splitting data...")
    train_graphs, val_graphs, test_graphs = train_val_test_split(
        graphs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_by_match=False
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=16  # Small batch for testing
    )
    
    # Create model
    print("Creating model...")
    model = PlayerSimilarityAutoencoder(config.model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Quick test: run 3 epochs
    print("\nRunning quick training test (3 epochs)...")
    config.training.num_epochs = 3
    config.training.patience = 10  # Don't early stop during test
    
    history = trainer.train()
    
    print("\n=== Training Complete ===")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Compute reconstruction metrics
    print("\n=== Reconstruction Metrics ===")
    metrics = compute_reconstruction_metrics(model, test_loader, trainer.device)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
