"""
Phase 2: Graph Assembly

This module handles:
1. Converting processed frames into graph structures
2. Building node features (position, velocity, sprint, team)
3. Building edge features (teammate vs opponent connections)
4. Creating fully connected graphs for GNN processing

Each graph represents a single 1-second snapshot of the match,
containing 22 nodes (players) with edges connecting every pair.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .config import GraphConfig, get_config
from .data_preparation import FrameData, PlayerFrame


@dataclass
class GraphSnapshot:
    """
    A single graph snapshot ready for GNN processing.
    
    Attributes:
        data: PyTorch Geometric Data object
        player_ids: List of player IDs in node order
        player_roles: List of player roles in node order
        match_id: Source match ID
        frame_number: Source frame number
    """
    data: Data
    player_ids: List[int]
    player_roles: List[str]
    team_ids: List[int]
    match_id: int
    frame_number: int


class GraphAssembler:
    """
    Assembles graph snapshots from processed frame data.
    
    Key responsibilities:
    - Convert player features to node feature tensors
    - Create fully connected edge index
    - Compute edge features (same team indicator)
    - Package into PyTorch Geometric Data objects
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.num_nodes = config.num_nodes
    
    def _build_node_features(
        self, 
        players: List[PlayerFrame], 
        reference_team_id: int,
        ball_x: float = 0.0,
        ball_y: float = 0.0
    ) -> torch.Tensor:
        """
        Build node feature matrix from player data.
        
        Feature vector per player (9 features):
        [x, y, vx, vy, sprint_flag, team_flag, dist_to_ball, dist_to_own_goal, dist_to_opp_goal]
        
        - x, y: Normalized position [-1, 1]
        - vx, vy: Velocity components
        - sprint_flag: 1 if sprinting, 0 otherwise
        - team_flag: 0 if same team as reference player, 1 if opponent
        - dist_to_ball: Euclidean distance to ball (normalized)
        - dist_to_own_goal: Distance to own goal (helps distinguish defenders)
        - dist_to_opp_goal: Distance to opponent goal (helps distinguish attackers)
        
        Args:
            players: List of PlayerFrame objects
            reference_team_id: Team ID to use as reference (0 = teammate, 1 = opponent)
            ball_x: Ball x coordinate (normalized)
            ball_y: Ball y coordinate (normalized)
            
        Returns:
            Tensor of shape [num_nodes, node_feature_dim]
        """
        features = []
        
        # Goal positions (in normalized coordinates where team attacks towards +x)
        # Own goal is at x=-1 (left), opponent goal is at x=+1 (right)
        own_goal_x, own_goal_y = -1.0, 0.0
        opp_goal_x, opp_goal_y = 1.0, 0.0
        
        # Ball is already normalized in [-1, 1] by data_preparation (team-aware)
        norm_ball_x = ball_x
        norm_ball_y = ball_y
        
        for player in players:
            # Position features
            x = player.x
            y = player.y
            
            # Velocity features
            vx = player.vx
            vy = player.vy
            
            # Sprint flag (binary)
            sprint = 1.0 if player.is_sprinting else 0.0
            
            # Team flag: 0 = teammate of reference, 1 = opponent
            team_flag = 0.0 if player.team_id == reference_team_id else 1.0
            
            # Distance to ball (Euclidean, scaled to [0, ~2.8] for unit square)
            dist_to_ball = np.sqrt((x - norm_ball_x)**2 + (y - norm_ball_y)**2)
            # Normalize to roughly [0, 1] by dividing by max possible distance (~2.8)
            dist_to_ball = dist_to_ball / 2.83
            
            # Goal-relative distances (crucial for role distinction)
            # For the reference team, own goal is at -1, opponent at +1
            # For opponents, it's flipped
            if team_flag == 0.0:  # Same team as reference (attacking right)
                dist_to_own_goal = np.sqrt((x - own_goal_x)**2 + (y - own_goal_y)**2) / 2.83
                dist_to_opp_goal = np.sqrt((x - opp_goal_x)**2 + (y - opp_goal_y)**2) / 2.83
            else:  # Opponent team (attacking left from their perspective)
                dist_to_own_goal = np.sqrt((x - opp_goal_x)**2 + (y - opp_goal_y)**2) / 2.83
                dist_to_opp_goal = np.sqrt((x - own_goal_x)**2 + (y - own_goal_y)**2) / 2.83
            
            feature_vector = [x, y, vx, vy, sprint, team_flag, dist_to_ball, dist_to_own_goal, dist_to_opp_goal]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_edge_index(self, num_nodes: int) -> torch.Tensor:
        """
        Build fully connected edge index (excluding self-loops).
        
        Creates edges from every node to every other node.
        This allows the attention mechanism to learn which
        connections are important.
        
        Args:
            num_nodes: Number of nodes in graph
            
        Returns:
            Edge index tensor of shape [2, num_edges]
        """
        if self.config.fully_connected:
            # Create all pairs
            source = []
            target = []
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j or self.config.include_self_loops:
                        source.append(i)
                        target.append(j)
            
            return torch.tensor([source, target], dtype=torch.long)
        else:
            # Could implement distance-based connectivity here
            raise NotImplementedError("Only fully connected graphs supported")
    
    def _build_edge_features(
        self, 
        players: List[PlayerFrame], 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Build edge feature matrix.
        
        Edge features:
        [same_team_flag]
        
        - same_team_flag: 1 if both players on same team, 0 if opponents
        
        This tells the message passing layers whether incoming
        information is from a friendly or hostile player.
        
        Args:
            players: List of PlayerFrame objects
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Edge feature tensor of shape [num_edges, edge_feature_dim]
        """
        num_edges = edge_index.shape[1]
        edge_features = []
        
        team_ids = [p.team_id for p in players]
        
        for i in range(num_edges):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            
            # Same team indicator
            same_team = 1.0 if team_ids[source_idx] == team_ids[target_idx] else 0.0
            
            edge_features.append([same_team])
        
        return torch.tensor(edge_features, dtype=torch.float32)
    
    def frame_to_graph(
        self, 
        frame: FrameData, 
        reference_player_idx: Optional[int] = None
    ) -> GraphSnapshot:
        """
        Convert a single frame to a graph snapshot.
        
        Args:
            frame: Processed frame data
            reference_player_idx: Index of player to use as reference for team flags.
                                 If None, uses first player's team.
                                 
        Returns:
            GraphSnapshot object containing PyG Data and metadata
        """
        players = frame.players
        
        # Determine reference team for team flags
        if reference_player_idx is not None:
            reference_team_id = players[reference_player_idx].team_id
        else:
            reference_team_id = players[0].team_id
        
        # Build node features (including ball-relative features)
        node_features = self._build_node_features(
            players, 
            reference_team_id,
            ball_x=frame.ball_x,
            ball_y=frame.ball_y
        )
        
        # Build edge index
        edge_index = self._build_edge_index(len(players))
        
        # Build edge features
        edge_features = self._build_edge_features(players, edge_index)
        
        # Store original positions for reconstruction target
        positions = torch.tensor(
            [[p.x, p.y] for p in players],
            dtype=torch.float32
        )
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=positions,  # Target for reconstruction
            num_nodes=len(players)
        )
        
        # Create snapshot with metadata
        snapshot = GraphSnapshot(
            data=data,
            player_ids=[p.player_id for p in players],
            player_roles=[p.role_name for p in players],
            team_ids=[p.team_id for p in players],
            match_id=frame.match_id,
            frame_number=frame.frame_number
        )
        
        return snapshot
    
    def frames_to_graphs(self, frames: List[FrameData]) -> List[GraphSnapshot]:
        """
        Convert multiple frames to graph snapshots.
        
        Args:
            frames: List of processed frame data
            
        Returns:
            List of GraphSnapshot objects
        """
        graphs = []
        
        for frame in tqdm(frames, desc="Building graphs"):
            try:
                graph = self.frame_to_graph(frame)
                graphs.append(graph)
            except Exception as e:
                print(f"Error building graph for frame {frame.frame_number}: {e}")
                continue
        
        return graphs


def create_masked_graph(
    graph: GraphSnapshot, 
    mask_idx: int
) -> Tuple[Data, torch.Tensor]:
    """
    Create a masked version of a graph for self-supervised training.
    
    The masking task:
    - Set the position (x, y) of one player to (0, 0)
    - The model must predict the original position
    
    Args:
        graph: Original graph snapshot
        mask_idx: Index of node to mask
        
    Returns:
        Tuple of (masked_data, target_position)
    """
    # Clone the data
    data = graph.data.clone()
    
    # Store original position as target
    target_position = data.y[mask_idx].clone()
    
    # Mask the position in node features (indices 0, 1 are x, y)
    data.x[mask_idx, 0] = 0.0  # x
    data.x[mask_idx, 1] = 0.0  # y
    
    # Also mask velocity since it depends on position
    data.x[mask_idx, 2] = 0.0  # vx
    data.x[mask_idx, 3] = 0.0  # vy
    
    # Mask tactical features that leak position (dist_to_ball, dist_to_own_goal, dist_to_opp_goal)
    if data.x.size(1) >= 9:
        data.x[mask_idx, 6] = 0.0
        data.x[mask_idx, 7] = 0.0
        data.x[mask_idx, 8] = 0.0
    
    # Add mask indicator
    data.mask_idx = mask_idx
    
    return data, target_position


def collate_graphs(graphs: List[GraphSnapshot]) -> Dict:
    """
    Collate multiple graphs into a batch-ready format.
    
    Returns a dictionary with:
    - 'data_list': List of PyG Data objects
    - 'player_ids': List of player ID lists
    - 'player_roles': List of role lists
    - 'metadata': List of (match_id, frame_number) tuples
    """
    return {
        'data_list': [g.data for g in graphs],
        'player_ids': [g.player_ids for g in graphs],
        'player_roles': [g.player_roles for g in graphs],
        'team_ids': [g.team_ids for g in graphs],
        'metadata': [(g.match_id, g.frame_number) for g in graphs]
    }


def save_graphs(graphs: List[GraphSnapshot], path: str):
    """Save graph snapshots to disk."""
    import pickle
    
    # Convert to serializable format
    serializable = []
    for g in graphs:
        serializable.append({
            'data': g.data,
            'player_ids': g.player_ids,
            'player_roles': g.player_roles,
            'team_ids': g.team_ids,
            'match_id': g.match_id,
            'frame_number': g.frame_number
        })
    
    with open(path, 'wb') as f:
        pickle.dump(serializable, f)
    
    print(f"Saved {len(graphs)} graphs to {path}")


def load_graphs(path: str) -> List[GraphSnapshot]:
    """Load graph snapshots from disk."""
    import pickle
    
    with open(path, 'rb') as f:
        serializable = pickle.load(f)
    
    graphs = []
    for item in serializable:
        graphs.append(GraphSnapshot(
            data=item['data'],
            player_ids=item['player_ids'],
            player_roles=item['player_roles'],
            team_ids=item['team_ids'],
            match_id=item['match_id'],
            frame_number=item['frame_number']
        ))
    
    return graphs


if __name__ == "__main__":
    # Test graph assembly
    from .data_preparation import SkillCornerDataLoader
    
    config = get_config()
    
    # Load some test data
    loader = SkillCornerDataLoader(config.data)
    match_id = loader.matches_info[0]['id']
    frames = loader.process_match(match_id)
    
    # Assemble graphs
    assembler = GraphAssembler(config.graph)
    graphs = assembler.frames_to_graphs(frames[:100])  # Test with first 100 frames
    
    print(f"\n=== Graph Statistics ===")
    print(f"Total graphs: {len(graphs)}")
    
    if graphs:
        sample = graphs[0]
        print(f"\n=== Sample Graph ===")
        print(f"Match: {sample.match_id}, Frame: {sample.frame_number}")
        print(f"Nodes: {sample.data.num_nodes}")
        print(f"Edges: {sample.data.edge_index.shape[1]}")
        print(f"Node features shape: {sample.data.x.shape}")
        print(f"Edge features shape: {sample.data.edge_attr.shape}")
        print(f"Target positions shape: {sample.data.y.shape}")
        
        print(f"\n=== Sample Node Features ===")
        print(f"Player 0: {sample.player_ids[0]} ({sample.player_roles[0]})")
        print(f"Features: {sample.data.x[0].tolist()}")
        
        print(f"\n=== Masking Test ===")
        masked_data, target = create_masked_graph(sample, mask_idx=0)
        print(f"Masked node features (x,y should be 0): {masked_data.x[0, :4].tolist()}")
        print(f"Target position: {target.tolist()}")
