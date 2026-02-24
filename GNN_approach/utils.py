"""
Utility functions for the GNN Player Similarity System.

Contains helper functions for:
- Visualization (all plots are saved, not displayed)
- Metrics computation
- Data analysis
- Model inspection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path


# Define position groups for interpretable visualization
POSITION_GROUPS = {
    'Goalkeeper': 'Goalkeeper',
    'Right Back': 'Defender',
    'Left Back': 'Defender',
    'Right Center Back': 'Defender',
    'Left Center Back': 'Defender',
    'Center Back': 'Defender',
    'Right Wing Back': 'Defender',
    'Left Wing Back': 'Defender',
    'Right Defensive Midfield': 'Midfielder',
    'Left Defensive Midfield': 'Midfielder',
    'Defensive Midfield': 'Midfielder',
    'Right Midfield': 'Midfielder',
    'Left Midfield': 'Midfielder',
    'Central Midfield': 'Midfielder',
    'Attacking Midfield': 'Midfielder',
    'Right Winger': 'Forward',
    'Left Winger': 'Forward',
    'Right Forward': 'Forward',
    'Left Forward': 'Forward',
    'Center Forward': 'Forward',
    'Striker': 'Forward',
}

# Color scheme for position groups
GROUP_COLORS = {
    'Goalkeeper': '#FFD700',  # Gold
    'Defender': '#1E90FF',    # Blue
    'Midfielder': '#32CD32',  # Green
    'Forward': '#FF4500',     # Red-Orange
    'Unknown': '#808080'      # Gray
}


def get_position_group(role: str) -> str:
    """Map detailed role to position group."""
    return POSITION_GROUPS.get(role, 'Unknown')


def plot_training_history(history: Dict, save_path: str):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'learning_rate'
        save_path: Path to save figure (required)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='#FF5722', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0].set_title('Training Progress', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    axes[0].scatter([best_epoch], [best_loss], color='green', s=100, zorder=5)
    
    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], color='#9C27B0', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Learning Rate', fontsize=11)
    axes[1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_embedding_tsne(
    profiles: pd.DataFrame,
    perplexity: int = 30,
    save_path: str = None
):
    """
    Plot t-SNE visualization of player embeddings colored by position GROUP.
    
    This is the most interpretable visualization - we expect:
    - Goalkeepers clustered separately
    - Defenders grouped together
    - Midfielders grouped together
    - Forwards grouped together
    
    Args:
        profiles: Player profiles with embeddings
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure (required)
    """
    from sklearn.manifold import TSNE
    
    # Extract embeddings
    embeddings = np.stack(profiles['embedding'].values)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Map roles to position groups
    position_groups = profiles['role'].apply(get_position_group).values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each group with consistent colors
    for group in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward', 'Unknown']:
        mask = position_groups == group
        if mask.sum() == 0:
            continue
            
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=GROUP_COLORS[group],
            label=f'{group} (n={mask.sum()})',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('Player Embedding Space by Position Group\n(t-SNE Visualization)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # Remove axis ticks (t-SNE dimensions are arbitrary)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# Detailed role color mapping (grouped by position type with varying shades)
DETAILED_ROLE_COLORS = {
    # Goalkeepers - Gold
    'Goalkeeper': '#FFD700',
    
    # Defenders - Blues (light to dark based on position)
    'Right Back': '#87CEEB',        # Light blue
    'Left Back': '#4169E1',         # Royal blue
    'Right Wing Back': '#00BFFF',   # Deep sky blue
    'Left Wing Back': '#1E90FF',    # Dodger blue
    'Right Center Back': '#0000CD', # Medium blue
    'Left Center Back': '#00008B',  # Dark blue
    'Center Back': '#000080',       # Navy
    
    # Midfielders - Greens (light to dark)
    'Right Midfield': '#90EE90',    # Light green
    'Left Midfield': '#32CD32',     # Lime green
    'Central Midfield': '#228B22',  # Forest green
    'Right Defensive Midfield': '#2E8B57', # Sea green
    'Left Defensive Midfield': '#006400',  # Dark green
    'Defensive Midfield': '#004d00',       # Darker green
    'Attacking Midfield': '#9ACD32', # Yellow green
    
    # Forwards - Reds/Oranges
    'Right Winger': '#FF6347',      # Tomato
    'Left Winger': '#FF4500',       # Orange red
    'Right Forward': '#DC143C',     # Crimson
    'Left Forward': '#B22222',      # Fire brick
    'Center Forward': '#8B0000',    # Dark red
    'Striker': '#FF0000',           # Red
}


def plot_embedding_tsne_detailed(
    profiles: pd.DataFrame,
    perplexity: int = 30,
    save_path: str = None
):
    """
    Plot t-SNE visualization with SPECIFIC positions (not grouped).
    
    Shows detailed role clustering - useful to see if model distinguishes:
    - Left Back vs Right Back
    - Left Winger vs Right Winger
    - Defensive vs Attacking Midfield
    
    Args:
        profiles: Player profiles with embeddings
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure (required)
    """
    from sklearn.manifold import TSNE
    
    # Extract embeddings
    embeddings = np.stack(profiles['embedding'].values)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    roles = profiles['role'].values
    unique_roles = sorted(set(roles))
    
    # Create figure with two panels: main plot + legend
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort roles by position group for better legend organization
    group_order = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3, 'Unknown': 4}
    unique_roles_sorted = sorted(unique_roles, key=lambda r: (group_order.get(get_position_group(r), 5), r))
    
    # Use markers to differentiate Left vs Right
    markers = {
        'Left': 's',    # Square for left-side players
        'Right': '^',   # Triangle for right-side players
        'Center': 'o',  # Circle for central players
        'default': 'o'
    }
    
    def get_marker(role):
        if 'Left' in role:
            return 's'
        elif 'Right' in role:
            return '^'
        else:
            return 'o'
    
    # Plot each role
    for role in unique_roles_sorted:
        mask = roles == role
        if mask.sum() == 0:
            continue
        
        color = DETAILED_ROLE_COLORS.get(role, '#808080')
        marker = get_marker(role)
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            marker=marker,
            label=f'{role} (n={mask.sum()})',
            alpha=0.75,
            s=70,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title('Player Embedding Space by Specific Role\n(t-SNE Visualization | ◯ Center, □ Left, △ Right)', 
                 fontsize=12, fontweight='bold')
    
    # Create legend outside plot area
    ax.legend(
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        ncol=1
    )
    ax.grid(True, alpha=0.2)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_similarity_matrix_by_group(
    similarity_matrix: np.ndarray,
    profiles: pd.DataFrame,
    save_path: str
):
    """
    Plot similarity matrix sorted by position group.
    
    Shows whether the model learns to group similar positions together.
    Expect to see block structure along diagonal if model works well.
    
    Args:
        similarity_matrix: Pairwise similarity scores
        profiles: Player profiles with roles
        save_path: Path to save figure (required)
    """
    # Add position group
    profiles = profiles.copy()
    profiles['group'] = profiles['role'].apply(get_position_group)
    
    # Sort by group then by role within group
    group_order = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward', 'Unknown']
    profiles['group_order'] = profiles['group'].apply(lambda x: group_order.index(x) if x in group_order else 99)
    sorted_df = profiles.sort_values(['group_order', 'role']).reset_index(drop=True)
    
    # Reorder similarity matrix
    sorted_indices = sorted_df.index.values
    # Need to map back to original indices
    original_indices = profiles.index.values
    idx_map = {orig: i for i, orig in enumerate(profiles['player_id'].values)}
    reorder = [idx_map[pid] for pid in sorted_df['player_id'].values]
    
    similarity_sorted = similarity_matrix[reorder][:, reorder]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(similarity_sorted, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=10)
    
    # Add group boundaries and labels
    groups = sorted_df['group'].values
    boundaries = [0]
    current_group = groups[0]
    group_labels = [current_group]
    
    for i, g in enumerate(groups[1:], 1):
        if g != current_group:
            boundaries.append(i)
            group_labels.append(g)
            current_group = g
    boundaries.append(len(groups))
    
    # Draw boundaries
    for b in boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color='black', linewidth=1.5)
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5)
    
    # Add group labels on axes
    label_positions = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries) - 1)]
    ax.set_xticks(label_positions)
    ax.set_xticklabels(group_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(label_positions)
    ax.set_yticklabels(group_labels, fontsize=10)
    
    ax.set_title('Player Similarity Matrix by Position Group\n(Diagonal blocks indicate good grouping)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_role_match_rates(validation_results: Dict, save_path: str):
    """
    Plot bar chart of role match rates from validation.
    
    Args:
        validation_results: Dict with per-role validation metrics
        save_path: Path to save figure (required)
    """
    # Extract per-role metrics
    role_metrics = validation_results.get('per_role_metrics', {})
    if not role_metrics:
        print("  No per-role metrics available for plotting")
        return
    
    # Prepare data
    roles = []
    exact_rates = []
    group_rates = []
    counts = []
    
    for role, metrics in role_metrics.items():
        roles.append(role)
        exact_rates.append(metrics.get('exact_match_rate', 0) * 100)
        group_rates.append(metrics.get('group_match_rate', 0) * 100)
        counts.append(metrics.get('count', 0))
    
    # Sort by group match rate
    sorted_indices = np.argsort(group_rates)[::-1]
    roles = [roles[i] for i in sorted_indices]
    exact_rates = [exact_rates[i] for i in sorted_indices]
    group_rates = [group_rates[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(roles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, group_rates, width, label='Position Group Match', 
                   color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, exact_rates, width, label='Exact Role Match', 
                   color='#4CAF50', alpha=0.8)
    
    ax.set_xlabel('Player Role', fontsize=11)
    ax.set_ylabel('Match Rate (%)', fontsize=11)
    ax.set_title('Similarity Search Accuracy by Role\n(Higher = Better)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roles, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample counts
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width(), bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7, color='gray')
    
    ax.set_ylim(0, max(max(group_rates), 100) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_embedding_statistics(profiles: pd.DataFrame, save_path: str):
    """
    Plot statistics about the learned embeddings.
    
    Shows embedding magnitude distribution and samples per player.
    
    Args:
        profiles: Player profiles with embeddings
        save_path: Path to save figure (required)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Embedding magnitudes by group
    profiles = profiles.copy()
    profiles['group'] = profiles['role'].apply(get_position_group)
    profiles['embedding_norm'] = profiles['embedding'].apply(lambda x: np.linalg.norm(x))
    
    groups = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
    group_norms = [profiles[profiles['group'] == g]['embedding_norm'].values for g in groups]
    group_norms = [g for g in group_norms if len(g) > 0]
    groups = [g for g, n in zip(groups, [profiles[profiles['group'] == g]['embedding_norm'].values for g in groups]) if len(n) > 0]
    
    bp = axes[0].boxplot(group_norms, labels=groups, patch_artist=True)
    for patch, group in zip(bp['boxes'], groups):
        patch.set_facecolor(GROUP_COLORS.get(group, 'gray'))
        patch.set_alpha(0.7)
    
    axes[0].set_xlabel('Position Group', fontsize=11)
    axes[0].set_ylabel('Embedding L2 Norm', fontsize=11)
    axes[0].set_title('Embedding Magnitude by Position', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Samples per player distribution
    axes[1].hist(profiles['num_samples'], bins=30, color='#9C27B0', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel('Number of Frame Samples', fontsize=11)
    axes[1].set_ylabel('Number of Players', fontsize=11)
    axes[1].set_title('Samples per Player Distribution', fontsize=12, fontweight='bold')
    axes[1].axvline(profiles['num_samples'].median(), color='red', linestyle='--', 
                    label=f'Median: {profiles["num_samples"].median():.0f}')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def analyze_similar_players(
    target_player_id: int,
    profiles: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Detailed analysis of similar players.
    
    Args:
        target_player_id: ID of target player
        profiles: Player profiles
        similarity_matrix: Similarity matrix
        top_k: Number of similar players
        
    Returns:
        DataFrame with analysis results
    """
    player_ids = profiles['player_id'].values
    target_idx = np.where(player_ids == target_player_id)[0][0]
    
    target_role = profiles.iloc[target_idx]['role']
    target_team = profiles.iloc[target_idx]['team_id']
    target_group = get_position_group(target_role)
    
    similarities = similarity_matrix[target_idx].copy()
    similarities[target_idx] = -1  # Exclude self
    
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        similar_role = profiles.iloc[idx]['role']
        similar_group = get_position_group(similar_role)
        results.append({
            'rank': rank,
            'player_id': profiles.iloc[idx]['player_id'],
            'role': similar_role,
            'group': similar_group,
            'team_id': profiles.iloc[idx]['team_id'],
            'similarity': similarities[idx],
            'same_role': similar_role == target_role,
            'same_group': similar_group == target_group,
            'same_team': profiles.iloc[idx]['team_id'] == target_team
        })
    
    return pd.DataFrame(results)


def print_model_summary(model: torch.nn.Module):
    """Print a summary of model architecture and parameters."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    print("\nLayer-wise parameter count:")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        print(f"{name}: {num_params:,}")
    
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_metrics(metrics: Dict, path: str):
    """Save metrics dictionary to JSON."""
    import json
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"  Saved: {path}")
