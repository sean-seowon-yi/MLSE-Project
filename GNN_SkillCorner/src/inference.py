"""
Phase 5: Generating "DNA" & Similarity Search

This module handles:
1. Generating embeddings for all players across all frames
2. Aggregating embeddings into player profiles
3. Computing similarity scores between players
4. Validating that similar players share tactical roles
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from .config import InferenceConfig, get_config
from .model import PlayerSimilarityAutoencoder
from .dataset import EmbeddingDataset, collate_fn_inference
from torch.utils.data import DataLoader
from .graph_assembly import GraphSnapshot


class EmbeddingGenerator:
    """
    Generates player embeddings from trained model.
    
    For each 1-second snapshot, generates a latent vector (Player DNA)
    for every player. These vectors capture tactical positioning,
    movement patterns, and relationships with teammates/opponents.
    """
    
    def __init__(
        self,
        model: PlayerSimilarityAutoencoder,
        device: torch.device
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def generate_embeddings(
        self,
        graphs: List[GraphSnapshot],
        batch_size: int = 64
    ) -> Dict[int, List[Dict]]:
        """
        Generate embeddings for all players in all graphs.
        
        Args:
            graphs: List of graph snapshots
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping player_id -> list of embedding records
            Each record contains: embedding, role, team_id, match_id, frame
        """
        dataset = EmbeddingDataset(graphs)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_inference
        )
        
        # Store embeddings per player
        player_embeddings = defaultdict(list)
        
        for batch in tqdm(loader, desc="Generating embeddings"):
            data = batch['data'].to(self.device)
            
            # Get embeddings
            embeddings = self.model.get_all_embeddings(data)
            embeddings = embeddings.cpu().numpy()
            
            # Distribute to players
            offset = 0
            for i, batch_size_i in enumerate(batch['batch_sizes']):
                player_ids = batch['player_ids'][i]
                player_roles = batch['player_roles'][i]
                team_ids = batch['team_ids'][i]
                match_id = batch['match_id'][i]
                frame_number = batch['frame_number'][i]
                
                for j in range(batch_size_i):
                    player_id = player_ids[j]
                    record = {
                        'embedding': embeddings[offset + j],
                        'role': player_roles[j],
                        'team_id': team_ids[j],
                        'match_id': match_id,
                        'frame': frame_number
                    }
                    player_embeddings[player_id].append(record)
                
                offset += batch_size_i
        
        return dict(player_embeddings)


class PlayerProfileBuilder:
    """
    Builds aggregated player profiles from frame-level embeddings.
    
    The profile is created by averaging all embeddings for a player,
    capturing their "habitual" tactical behavior.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def build_profiles(
        self,
        player_embeddings: Dict[int, List[Dict]],
        min_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build aggregated profiles for all players.
        
        Args:
            player_embeddings: Output from EmbeddingGenerator
            min_samples: Minimum samples required for reliable profile
            
        Returns:
            DataFrame with columns:
            - player_id
            - role (most common)
            - team_id
            - num_samples
            - embedding (aggregated)
        """
        if min_samples is None:
            min_samples = self.config.min_samples_per_player
        
        profiles = []
        
        for player_id, records in player_embeddings.items():
            if len(records) < min_samples:
                continue
            
            # Aggregate embeddings
            embeddings = np.array([r['embedding'] for r in records])
            
            if self.config.aggregation_method == 'mean':
                profile_embedding = embeddings.mean(axis=0)
            elif self.config.aggregation_method == 'median':
                profile_embedding = np.median(embeddings, axis=0)
            else:
                profile_embedding = embeddings.mean(axis=0)
            
            # Get most common role
            roles = [r['role'] for r in records]
            most_common_role = max(set(roles), key=roles.count)
            
            # Get team_id (should be consistent)
            team_id = records[0]['team_id']
            
            profiles.append({
                'player_id': player_id,
                'role': most_common_role,
                'team_id': team_id,
                'num_samples': len(records),
                'embedding': profile_embedding
            })
        
        return pd.DataFrame(profiles)


class SimilaritySearcher:
    """
    Performs similarity search between player profiles.
    
    Uses cosine similarity to find players with similar
    movement patterns and tactical behaviors.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def compute_similarity_matrix(
        self,
        profiles: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Args:
            profiles: DataFrame with 'embedding' column
            
        Returns:
            Similarity matrix [num_players, num_players]
        """
        embeddings = np.stack(profiles['embedding'].values)
        
        if self.config.similarity_metric == 'cosine':
            similarity = cosine_similarity(embeddings)
        elif self.config.similarity_metric == 'euclidean':
            distances = cdist(embeddings, embeddings, metric='euclidean')
            # Convert to similarity (inverse distance)
            similarity = 1 / (1 + distances)
        else:
            similarity = cosine_similarity(embeddings)
        
        return similarity
    
    def find_similar_players(
        self,
        target_player_id: int,
        profiles: pd.DataFrame,
        similarity_matrix: np.ndarray,
        top_k: Optional[int] = None,
        exclude_same_team: bool = False
    ) -> pd.DataFrame:
        """
        Find most similar players to target.
        
        Args:
            target_player_id: ID of target player
            profiles: Player profiles DataFrame
            similarity_matrix: Pre-computed similarity matrix
            top_k: Number of similar players to return
            exclude_same_team: Whether to exclude players from same team
            
        Returns:
            DataFrame with similar players and scores
        """
        if top_k is None:
            top_k = self.config.top_k
        
        # Find target player index
        player_ids = profiles['player_id'].values
        target_idx = np.where(player_ids == target_player_id)[0]
        
        if len(target_idx) == 0:
            raise ValueError(f"Player {target_player_id} not found in profiles")
        
        target_idx = target_idx[0]
        
        # Get similarities
        similarities = similarity_matrix[target_idx]
        
        # Get target team if excluding same team
        if exclude_same_team:
            target_team = profiles.iloc[target_idx]['team_id']
            team_mask = profiles['team_id'] != target_team
        else:
            team_mask = np.ones(len(profiles), dtype=bool)
        
        # Don't include self
        team_mask[target_idx] = False
        
        # Sort and get top-k
        valid_indices = np.where(team_mask)[0]
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1][:top_k]]
        
        # Build result DataFrame
        results = []
        for idx in sorted_indices:
            results.append({
                'player_id': profiles.iloc[idx]['player_id'],
                'role': profiles.iloc[idx]['role'],
                'team_id': profiles.iloc[idx]['team_id'],
                'similarity': similarities[idx],
                'num_samples': profiles.iloc[idx]['num_samples']
            })
        
        return pd.DataFrame(results)


class RoleValidator:
    """
    Validates that similar players share tactical roles.
    
    This is the key test: if the model learned roles from geometry
    (without seeing role labels), then similar players should
    share similar roles.
    """
    
    @staticmethod
    def validate_role_consistency(
        profiles: pd.DataFrame,
        similarity_matrix: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Compute role consistency metrics.
        
        For each player, check if their top-k similar players
        share the same role.
        
        Returns:
            Dictionary with validation metrics
        """
        player_ids = profiles['player_id'].values
        roles = profiles['role'].values
        
        exact_matches = []
        position_group_matches = []
        
        # Define position groups (must match utils.POSITION_GROUPS)
        position_groups = {
            'Goalkeeper': ['Goalkeeper'],
            'Defender': ['Left Back', 'Right Back', 'Center Back', 
                        'Left Center Back', 'Right Center Back',
                        'Right Wing Back', 'Left Wing Back'],
            'Midfield': ['Defensive Midfield', 'Central Midfield',
                        'Left Defensive Midfield', 'Right Defensive Midfield',
                        'Attacking Midfield', 'Left Midfield', 'Right Midfield'],
            'Forward': ['Left Winger', 'Right Winger', 'Center Forward',
                       'Left Wing', 'Right Wing', 'Striker',
                       'Left Forward', 'Right Forward']
        }
        
        # Create reverse mapping
        role_to_group = {}
        for group, group_roles in position_groups.items():
            for role in group_roles:
                role_to_group[role] = group
        
        for i in range(len(profiles)):
            target_role = roles[i]
            target_group = role_to_group.get(target_role, 'Other')
            
            # Get top-k similar (excluding self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Count matches
            num_exact = sum(1 for idx in top_indices if roles[idx] == target_role)
            num_group = sum(
                1 for idx in top_indices 
                if role_to_group.get(roles[idx], 'Other') == target_group
            )
            
            exact_matches.append(num_exact / top_k)
            position_group_matches.append(num_group / top_k)
        
        # Compute per-role metrics
        role_metrics = {}
        for role in set(roles):
            role_mask = roles == role
            if role_mask.sum() > 0:
                role_metrics[role] = {
                    'exact_match_rate': np.mean([exact_matches[i] for i in range(len(roles)) if role_mask[i]]),
                    'group_match_rate': np.mean([position_group_matches[i] for i in range(len(roles)) if role_mask[i]]),
                    'count': int(role_mask.sum())
                }
        
        return {
            'overall_exact_match_rate': np.mean(exact_matches),
            'overall_group_match_rate': np.mean(position_group_matches),
            'per_role_metrics': role_metrics
        }
    
    @staticmethod
    def print_validation_report(metrics: Dict):
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("ROLE VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nOverall Exact Role Match Rate: {metrics['overall_exact_match_rate']:.2%}")
        print(f"Overall Position Group Match Rate: {metrics['overall_group_match_rate']:.2%}")
        
        print("\n--- Per-Role Metrics ---")
        per_role = metrics['per_role_metrics']
        sorted_roles = sorted(per_role.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for role, data in sorted_roles:
            print(f"\n{role} (n={data['count']}):")
            print(f"  Exact match: {data['exact_match_rate']:.2%}")
            print(f"  Group match: {data['group_match_rate']:.2%}")


def save_embeddings(
    player_embeddings: Dict[int, List[Dict]],
    path: str
):
    """Save raw embeddings to disk."""
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for player_id, records in player_embeddings.items():
        serializable[player_id] = [
            {
                'embedding': r['embedding'].tolist(),
                'role': r['role'],
                'team_id': r['team_id'],
                'match_id': r['match_id'],
                'frame': r['frame']
            }
            for r in records
        ]
    
    with open(path, 'w') as f:
        json.dump(serializable, f)


def save_profiles(profiles: pd.DataFrame, path: str):
    """Save profiles to disk."""
    # Convert embeddings to lists
    profiles_copy = profiles.copy()
    profiles_copy['embedding'] = profiles_copy['embedding'].apply(lambda x: x.tolist())
    profiles_copy.to_json(path, orient='records', indent=2)


def load_profiles(path: str) -> pd.DataFrame:
    """Load profiles from disk."""
    profiles = pd.read_json(path, orient='records')
    profiles['embedding'] = profiles['embedding'].apply(np.array)
    return profiles


if __name__ == "__main__":
    # Test inference pipeline
    from .data_preparation import SkillCornerDataLoader
    from .graph_assembly import GraphAssembler
    
    config = get_config()
    
    # Load data
    print("Loading data...")
    loader = SkillCornerDataLoader(config.data)
    match_id = loader.matches_info[0]['id']
    frames = loader.process_match(match_id)
    
    # Build graphs
    print("Building graphs...")
    assembler = GraphAssembler(config.graph)
    graphs = assembler.frames_to_graphs(frames[:200])
    
    # Create untrained model (for testing pipeline)
    print("Creating model...")
    model = PlayerSimilarityAutoencoder(config.model)
    device = torch.device('cpu')
    
    # Generate embeddings
    print("\n=== Generating Embeddings ===")
    generator = EmbeddingGenerator(model, device)
    player_embeddings = generator.generate_embeddings(graphs, batch_size=32)
    print(f"Generated embeddings for {len(player_embeddings)} players")
    
    # Build profiles
    print("\n=== Building Profiles ===")
    profile_builder = PlayerProfileBuilder(config.inference)
    profiles = profile_builder.build_profiles(player_embeddings, min_samples=10)
    print(f"Built profiles for {len(profiles)} players")
    print(profiles[['player_id', 'role', 'num_samples']].head(10))
    
    # Compute similarity
    print("\n=== Computing Similarity ===")
    searcher = SimilaritySearcher(config.inference)
    similarity_matrix = searcher.compute_similarity_matrix(profiles)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Find similar players for first player
    if len(profiles) > 0:
        target_id = profiles.iloc[0]['player_id']
        target_role = profiles.iloc[0]['role']
        print(f"\n=== Similar Players to {target_id} ({target_role}) ===")
        similar = searcher.find_similar_players(target_id, profiles, similarity_matrix, top_k=5)
        print(similar)
    
    # Validate role consistency
    print("\n=== Role Validation ===")
    validator = RoleValidator()
    metrics = validator.validate_role_consistency(profiles, similarity_matrix, top_k=5)
    validator.print_validation_report(metrics)
