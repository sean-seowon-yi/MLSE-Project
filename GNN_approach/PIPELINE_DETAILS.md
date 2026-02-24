# Pipeline Details: GNN Player Similarity System

This document provides a detailed explanation of each phase in the pipeline, including the logic, data flow, and rationale behind design decisions.

---

## Table of Contents

1. [Phase 1: Data Preparation](#phase-1-data-preparation)
2. [Phase 2: Graph Assembly](#phase-2-graph-assembly)
3. [Phase 3: Model Architecture](#phase-3-model-architecture)
4. [Phase 4: Training Strategy](#phase-4-training-strategy)
5. [Phase 5: Inference & Similarity Search](#phase-5-inference--similarity-search)
6. [Complete Data Flow](#complete-data-flow)
7. [Key Design Decisions](#key-design-decisions)

---

## Phase 1: Data Preparation

**File**: `data_preparation.py`

**Goal**: Convert raw, noisy tracking data into a standardized geometric format that preserves tactical roles.

### 1.1 Data Loading

**What happens:**
- Loads `matches.json` to get list of all match IDs
- For each match, loads:
  - `{match_id}_match.json` — Metadata (lineups, pitch dimensions, periods)
  - `{match_id}_tracking_extrapolated.jsonl` — Frame-by-frame tracking data (10 fps)

**Why this structure:**
- JSONL format allows streaming large files without loading everything into memory
- Metadata separated from tracking data for efficient access

**Code location:**
```python
class SkillCornerDataLoader:
    def _load_matches_index(self) -> List[Dict]:
        """Load the matches.json index file."""
        
    def _load_match_metadata(self, match_id: int) -> Dict:
        """Load detailed match metadata including player info and pitch dimensions."""
        
    def _load_tracking_data(self, match_id: int) -> Generator[Dict, None, None]:
        """Load tracking data frame by frame (memory efficient generator)."""
```

### 1.2 Frame Filtering

**What happens:**
- Filters frames to include only **active play**
- Requires exactly **22 players** (full teams)
- Skips frames outside match periods (pre-match, halftime)

**Why filter:**
- **Active play only**: Halftime warmups or stoppages don't represent real tactical situations
- **22 players**: Red cards create asymmetric situations that confuse the model
- **Match periods**: Only analyze actual game time

**Filtering logic:**
```python
# Check if frame is in active period
if frame_num < start_frame or frame_num > end_frame:
    continue  # Skip

# Check player count
valid_players = [p for p in frame['player_data'] if p.get('x') is not None]
if len(valid_players) != 22:
    continue  # Skip - not full teams
```

### 1.3 Team-Aware Normalization (CRITICAL)

**The Problem:**
Soccer fields are symmetric. Teams switch sides at halftime. If we don't normalize, a Left Back attacking left looks identical to a Right Back attacking right, and the model can't distinguish roles.

**The Solution:**
Rotate all coordinates so **every team appears to attack towards positive X (right side)**.

**How it works:**

```python
def _normalize_coordinates(self, x, y, is_home_team, home_attacking_direction, ...):
    # Determine if this player's team is attacking left
    if is_home_team:
        attacking_left = 'right_to_left' in home_attacking_direction
    else:
        # Away team attacks opposite direction
        attacking_left = 'left_to_right' in home_attacking_direction
    
    # Apply rotation if attacking left
    if attacking_left:
        x = -x  # Flip X
        y = -y  # Flip Y (crucial!)
    
    # Normalize to [-1, 1]
    x = x / (pitch_length / 2)
    y = y / (pitch_width / 2)
    
    return x, y
```

**Why flip both X and Y:**
- Flipping only X would move a Left Back from top-left to top-right
- Flipping both X and Y moves them to bottom-left
- This ensures "Left" is always **negative Y**, "Right" is always **positive Y**

**Visual Example:**

```
Before normalization (Period 1):
Home team attacking left:
  Left Back at (x=-30, y=+20)  → top-left quadrant
  Right Back at (x=-30, y=-20) → bottom-left quadrant

After normalization (flip both x and y):
  Left Back at (x=+30, y=-20)  → bottom-right (now attacking right!)
  Right Back at (x=+30, y=+20) → top-right (consistent positioning!)
```

### 1.4 Feature Engineering

**For each player, compute:**

| Feature | Description | Computation |
|---------|-------------|-------------|
| Position (x, y) | Normalized coordinates | `x / (pitch_length/2)`, range [-1, 1] |
| Velocity (vx, vy) | Movement direction | `(current_pos - prev_pos) / dt` |
| Speed | Movement magnitude | `sqrt(vx² + vy²)` |
| Sprint flag | High-intensity indicator | `1 if speed > 7.0 m/s else 0` |
| Team indicator | Friend or foe | `0 = teammate, 1 = opponent` |

**Why these features:**
- **Position**: Where the player is (fundamental)
- **Velocity**: Where they're going (captures movement intent)
- **Sprint flag**: High-intensity movement (captures explosive actions)
- **Team indicator**: Critical for understanding relationships

**Output data structure:**
```python
@dataclass
class PlayerFrame:
    player_id: int      # Unique player identifier
    team_id: int        # Team identifier
    x: float            # Normalized position X [-1, 1]
    y: float            # Normalized position Y [-1, 1]
    vx: float           # Velocity X component
    vy: float           # Velocity Y component
    speed: float        # Speed magnitude (m/s)
    is_sprinting: bool  # True if speed > 7 m/s
    is_home_team: bool  # True if home team player
    role_name: str      # Position role (e.g., "Left Back")
    trackable_object: int  # Internal tracking ID
```

---

## Phase 2: Graph Assembly

**File**: `graph_assembly.py`

**Goal**: Convert frame data into graph structures that Graph Neural Networks can process.

### 2.1 Sampling Strategy

**What happens:**
- Creates a graph snapshot every **N frames** (default: 10 frames = 1 second)
- With 10 matches × 90 minutes × 60 seconds ≈ 54,000 potential graphs
- After filtering, expect ~50,000 valid graphs

**Why sample every second:**
- Captures tactical structure at that moment
- Not too sparse (loses context) or too dense (redundant)
- 1-second intervals preserve tactical relationships while reducing data volume

**Configuration:**
```python
@dataclass
class DataConfig:
    sampling_rate: int = 10  # Create graph every N frames (10 = 1 second at 10fps)
```

### 2.2 Node Definition

**Each graph contains exactly 22 nodes** (one per player).

**Node features** (6 dimensions):
```python
node_features = [
    x,           # Position X [-1, 1]
    y,           # Position Y [-1, 1]
    vx,          # Velocity X
    vy,          # Velocity Y
    sprint_flag, # 0 or 1
    team_flag    # 0 = teammate, 1 = opponent
]
```

**Node feature matrix shape**: `[22, 6]`

**Example node features:**
```python
# Node 0: Left Back (teammate, not sprinting)
[0.15, -0.30, 2.5, -1.2, 0.0, 0.0]

# Node 11: Opponent Right Winger (opponent, sprinting)
[-0.20, 0.25, -3.1, 1.8, 1.0, 1.0]
```

### 2.3 Edge Definition

**Connectivity: Fully Connected Graph**

Every player is connected to every other player (excluding self-loops).

**Why fully connected:**
- Don't want to manually decide that "30 meters is the cutoff for influence"
- A striker watching a goalkeeper 60m away is a valid tactical relationship
- The attention mechanism will learn which connections matter

**Edge Index construction:**
```python
def _build_edge_index(self, num_nodes: int) -> torch.Tensor:
    source = []
    target = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # No self-loops
                source.append(i)
                target.append(j)
    
    return torch.tensor([source, target], dtype=torch.long)
    # Shape: [2, 462] for 22 players (22 × 21 = 462 edges)
```

**Edge Features** (1 dimension):
```python
edge_features = [same_team_flag]  # 1 if same team, 0 if opponents
```

**Why edge features:**
- Tells message passing layers whether information is from friendly or hostile player
- Critical for learning defensive vs attacking patterns

### 2.4 Complete Graph Structure

```python
@dataclass
class GraphSnapshot:
    data: Data              # PyTorch Geometric Data object
    player_ids: List[int]   # Player IDs in node order
    player_roles: List[str] # Roles in node order
    team_ids: List[int]     # Team IDs in node order
    match_id: int           # Source match
    frame_number: int       # Source frame

# The PyG Data object contains:
Data(
    x=torch.tensor([22, 6]),           # Node features
    edge_index=torch.tensor([2, 462]), # Edge connectivity
    edge_attr=torch.tensor([462, 1]),  # Edge features
    y=torch.tensor([22, 2])            # Target positions (for training)
)
```

---

## Phase 3: Model Architecture

**File**: `model.py`

**Goal**: Design a neural network that learns meaningful player representations ("Player DNA").

### 3.1 Architecture Overview

```
Input Graph (22 nodes, 462 edges)
         ↓
┌─────────────────────────────────────┐
│         GATv2 ENCODER               │
│  ┌─────────────────────────────┐    │
│  │ Layer 1: Perception         │    │
│  │   Input: [22, 6]            │    │
│  │   GATv2 (4 heads, 64 dim)   │    │
│  │   Output: [22, 256]         │    │
│  └─────────────────────────────┘    │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │ Layer 2: Compression        │    │
│  │   Input: [22, 256]          │    │
│  │   GATv2 (1 head, 32 dim)    │    │
│  │   Output: [22, 32]          │    │
│  └─────────────────────────────┘    │
│                                     │
│  Output: Player Embeddings          │
│          ("Player DNA")             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│         MLP DECODER                 │
│  (Training only)                    │
│                                     │
│   Input: [32] embedding             │
│   Hidden: [64, 32]                  │
│   Output: [2] (x, y position)       │
└─────────────────────────────────────┘
```

### 3.2 GATv2 Encoder Details

**What is GATv2?**
- Graph Attention Network version 2
- Each node attends to all neighbors with learned attention weights
- GATv2 fixes the "static attention" problem of the original GAT
- Attention now depends on both query and key nodes dynamically

**Layer 1: Perception**
```python
self.gat1 = GATv2Conv(
    in_channels=6,        # Input: node features
    out_channels=64,      # Hidden dimension per head
    heads=4,              # 4 attention heads
    concat=True,          # Concatenate heads → 64 × 4 = 256
    dropout=0.1,
    edge_dim=1            # Use edge features
)
```

**What happens in Layer 1:**
- Each player looks at all 21 other players
- Computes attention score for each connection
- Aggregates information weighted by attention
- Output: `[22, 256]` (64 features × 4 heads, concatenated)

**Layer 2: Compression**
```python
self.gat2 = GATv2Conv(
    in_channels=256,      # From Layer 1
    out_channels=32,      # Latent dimension ("Player DNA")
    heads=1,              # Single head for final output
    concat=False,
    dropout=0.1,
    edge_dim=1
)
```

**Output**: `[22, 32]` — Each player represented as 32 numbers

**Attention Mechanism:**
```python
# For each edge (i → j):
# 1. Compute attention score
attention_score = LeakyReLU(W_query @ h_i + W_key @ h_j + W_edge @ e_ij)

# 2. Normalize with softmax
attention_weight = softmax(attention_score)  # Over all neighbors

# 3. Aggregate neighbor information
h_i_new = Σ (attention_weight[j] × W_value @ h_j)  # Sum over neighbors j
```

### 3.3 MLP Decoder

**Purpose:** Reconstructs player positions from embeddings (training only).

```python
class MLPDecoder(nn.Module):
    def __init__(self, config):
        layers = [
            nn.Linear(32, 64),   # Latent → Hidden
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),   # Hidden → Hidden
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)     # Hidden → Position (x, y)
        ]
```

**Why a decoder:**
- Forces the encoder to learn meaningful spatial information
- If embeddings can reconstruct positions, they capture tactical structure
- Only used during training; discarded for inference

### 3.4 Forward Pass

**Training mode:**
```python
def forward(self, x, edge_index, edge_attr, mask_idx):
    # Encode all players
    embeddings = self.encoder(x, edge_index, edge_attr)  # [22, 32]
    
    # Decode all positions
    all_predicted = self.decoder(embeddings)  # [22, 2]
    
    # Return prediction for masked node
    predicted_pos = all_predicted[mask_idx]  # [1, 2]
    
    return embeddings, predicted_pos, all_predicted
```

**Inference mode:**
```python
def get_all_embeddings(self, data):
    # Encode only (no decoder)
    embeddings = self.encoder(data.x, data.edge_index, data.edge_attr)
    return embeddings  # [22, 32] - These are "Player DNA"
```

---

## Phase 4: Training Strategy

**File**: `train.py`

**Goal**: Train the model using self-supervised learning—no role labels needed.

### 4.1 The Masking Task

**The "Fill in the Blank" Game:**

1. Take a graph of 22 players
2. Randomly select one player (e.g., the Right Back)
3. Set their position to (0, 0) in the input—effectively "hiding" them
4. Ask the model: "Where should this player be?"
5. Model predicts the position using context from other 21 players
6. Compare prediction to true position

**Masking implementation:**
```python
def create_masked_graph(graph: GraphSnapshot, mask_idx: int):
    data = graph.data.clone()
    
    # Store true position as target
    target_position = data.y[mask_idx].clone()
    
    # Mask position in node features (indices 0, 1 are x, y)
    data.x[mask_idx, 0] = 0.0  # x = 0
    data.x[mask_idx, 1] = 0.0  # y = 0
    
    # Also mask velocity (depends on position)
    data.x[mask_idx, 2] = 0.0  # vx = 0
    data.x[mask_idx, 3] = 0.0  # vy = 0
    
    return data, target_position
```

**Why this works:**
- To predict where a Right Back is, the model must understand:
  - Defensive line structure (from Center Backs)
  - Opponent positions (who they're marking)
  - Team shape (formation context)
- By forcing position prediction, model learns tactical relationships

### 4.2 Training Loop

**Per epoch:**
```python
def train_epoch(self):
    self.model.train()
    
    for batch in self.train_loader:
        # Move to device
        data = batch['data'].to(self.device)
        targets = batch['target'].to(self.device)
        mask_indices = batch['global_mask_idx']
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        embeddings, _, all_predictions = self.model(
            data.x, data.edge_index, data.edge_attr
        )
        
        # Get predictions for masked nodes
        predictions = all_predictions[mask_indices]
        
        # Compute loss
        loss = MSE(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
```

**Loss Function:**
```python
loss = MSE(predicted_position, true_position)
     = mean((pred_x - true_x)² + (pred_y - true_y)²)
```

**Why MSE:**
- Position is continuous (not classification)
- MSE penalizes large errors more than small ones
- Smooth gradient for optimization

### 4.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 64 | Balance between speed and gradient stability |
| Learning rate | 1e-3 | Standard starting point, with reduction on plateau |
| Epochs | 100 max | Early stopping prevents overfitting |
| Patience | 10 | Stop if no improvement for 10 epochs |
| Optimizer | Adam | Adaptive learning rates |
| Scheduler | ReduceLROnPlateau | Halve LR when validation plateaus |

### 4.4 What the Model Learns

Through the masking task, the model learns (without role labels):

1. **Spatial relationships**: "Right Backs are usually near Right Center Backs"
2. **Team structure**: "Defenders form a line at similar x-coordinates"
3. **Opponent awareness**: "Left Wingers often position opposite Right Backs"
4. **Movement patterns**: "Wingers move forward when team has possession"

### 4.5 Validation and Checkpointing

```python
# After each epoch:
val_loss = self.validate()

# Learning rate adjustment
self.scheduler.step(val_loss)

# Save best model
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    self.save_checkpoint('best_model.pt')
    
# Early stopping
if no_improvement_for_n_epochs:
    break
```

---

## Phase 5: Inference & Similarity Search

**File**: `inference.py`

**Goal**: Generate player embeddings and find similar players.

### 5.1 Embedding Generation

**Process:**
```python
class EmbeddingGenerator:
    @torch.no_grad()
    def generate_embeddings(self, graphs, batch_size=64):
        player_embeddings = defaultdict(list)
        
        for batch in dataloader:
            # Get embeddings for all players
            embeddings = self.model.get_all_embeddings(batch['data'])
            
            # Store per player
            for player_id, embedding in zip(player_ids, embeddings):
                player_embeddings[player_id].append({
                    'embedding': embedding,
                    'role': role,
                    'frame': frame_number
                })
        
        return player_embeddings
```

**Output structure:**
```python
# Player 38673 appears in 3600 frames (60 minutes × 60 seconds)
player_embeddings[38673] = [
    {'embedding': [0.12, -0.45, ..., 0.23], 'role': 'Center Forward', 'frame': 100},
    {'embedding': [0.15, -0.42, ..., 0.21], 'role': 'Center Forward', 'frame': 110},
    ...  # 3600 embeddings total
]
```

### 5.2 Player Profile Building

**Aggregation strategy:**
```python
class PlayerProfileBuilder:
    def build_profiles(self, player_embeddings, min_samples=100):
        profiles = []
        
        for player_id, records in player_embeddings.items():
            if len(records) < min_samples:
                continue  # Need enough data for reliable profile
            
            # Average all embeddings
            embeddings = np.array([r['embedding'] for r in records])
            profile_embedding = embeddings.mean(axis=0)
            
            # Get most common role
            roles = [r['role'] for r in records]
            most_common_role = max(set(roles), key=roles.count)
            
            profiles.append({
                'player_id': player_id,
                'role': most_common_role,
                'embedding': profile_embedding,
                'num_samples': len(records)
            })
        
        return pd.DataFrame(profiles)
```

**Why averaging:**
- Individual frames are noisy (specific tactical situations)
- Average captures consistent patterns—the player's "habitual" behavior
- More reliable than any single-frame embedding

### 5.3 Similarity Computation

**Method: Cosine Similarity**

```python
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Implementation:**
```python
class SimilaritySearcher:
    def compute_similarity_matrix(self, profiles):
        embeddings = np.stack(profiles['embedding'].values)
        similarity = cosine_similarity(embeddings)  # [N, N]
        return similarity
```

**Why cosine similarity:**
- Measures angle between vectors (direction, not magnitude)
- Normalized to range [-1, 1]
- Good for high-dimensional embeddings
- Robust to different embedding scales

### 5.4 Finding Similar Players

```python
def find_similar_players(self, target_player_id, profiles, similarity_matrix, top_k=10):
    # Find target index
    target_idx = np.where(player_ids == target_player_id)[0][0]
    
    # Get similarities
    similarities = similarity_matrix[target_idx]
    
    # Sort and get top-k (excluding self)
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = [i for i in sorted_indices if i != target_idx][:top_k]
    
    # Build results
    results = []
    for idx in sorted_indices:
        results.append({
            'player_id': profiles.iloc[idx]['player_id'],
            'role': profiles.iloc[idx]['role'],
            'similarity': similarities[idx]
        })
    
    return pd.DataFrame(results)
```

**Example output:**
```
Target: Player 38673 (Center Forward)

Similar Players:
  player_id | role           | similarity
  50951     | Center Forward | 0.92
  163972    | Center Forward | 0.88
  43829     | Right Winger   | 0.75
  133501    | Right Winger   | 0.71
  795506    | Center Forward | 0.68
```

### 5.5 Role Validation

**The Ultimate Test:**
- Model never saw role labels during training
- If similar players share roles, model learned from geometry alone

**Validation metrics:**
```python
class RoleValidator:
    def validate_role_consistency(self, profiles, similarity_matrix, top_k=5):
        exact_matches = []
        group_matches = []
        
        for i in range(len(profiles)):
            target_role = profiles.iloc[i]['role']
            
            # Get top-k similar (excluding self)
            top_indices = get_top_k_similar(similarity_matrix[i], exclude=i)
            
            # Count role matches
            similar_roles = [profiles.iloc[j]['role'] for j in top_indices]
            
            exact_match = sum(r == target_role for r in similar_roles) / top_k
            group_match = sum(same_position_group(r, target_role) for r in similar_roles) / top_k
            
            exact_matches.append(exact_match)
            group_matches.append(group_match)
        
        return {
            'exact_match_rate': np.mean(exact_matches),
            'group_match_rate': np.mean(group_matches)
        }
```

**Position groups:**
```python
position_groups = {
    'Goalkeeper': ['Goalkeeper'],
    'Defender': ['Left Back', 'Right Back', 'Center Back', 'Left Center Back', 'Right Center Back'],
    'Midfield': ['Defensive Midfield', 'Central Midfield', 'Attacking Midfield', ...],
    'Forward': ['Left Winger', 'Right Winger', 'Center Forward', 'Striker']
}
```

**Success criteria:**
- Exact match rate > 60%: Model distinguishes specific roles
- Group match rate > 80%: Model understands position groups

---

## Complete Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    RAW SKILLCORNER DATA                        │
│  matches.json, {id}_match.json, {id}_tracking.jsonl            │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                 PHASE 1: DATA PREPARATION                      │
│  • Filter active play frames                                   │
│  • Require 22 players                                          │
│  • Team-aware normalization (flip for attacking direction)     │
│  • Compute velocity, sprint flags                              │
│                                                                │
│  Output: FrameData (22 PlayerFrame objects per frame)          │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                 PHASE 2: GRAPH ASSEMBLY                        │
│  • Sample every 10 frames (1 second)                           │
│  • Build node features [22, 6]                                 │
│  • Build fully connected edges [2, 462]                        │
│  • Build edge features [462, 1]                                │
│                                                                │
│  Output: GraphSnapshot (~50,000 graphs)                        │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                 PHASE 3: MODEL ARCHITECTURE                    │
│  • GATv2 Encoder (2 layers, 4 heads)                           │
│  • MLP Decoder (for training)                                  │
│                                                                │
│  Output: Model ready for training                              │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                 PHASE 4: TRAINING                              │
│  • Masking task: hide one player, predict position             │
│  • Loss: MSE(predicted, true)                                  │
│  • Early stopping, checkpointing                               │
│                                                                │
│  Output: Trained model weights                                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                 PHASE 5: INFERENCE                             │
│  • Generate embeddings for all players in all frames           │
│  • Aggregate to player profiles (mean)                         │
│  • Compute similarity matrix                                   │
│  • Find similar players                                        │
│  • Validate role consistency                                   │
│                                                                │
│  Output: Player profiles, similarity rankings                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Alternative | Why This Choice |
|----------|-------------|-----------------|
| **GATv2 over GCN** | Graph Convolutional Network | Attention learns which connections matter; GCN treats all neighbors equally |
| **Fully connected graphs** | Distance-based edges (e.g., < 30m) | Let model learn important edges; a striker-goalkeeper relationship is valid |
| **Self-supervised (masking)** | Supervised with role labels | No labels needed; learns from structure alone; more generalizable |
| **Team-aware normalization** | Raw coordinates | Preserves left/right sidedness; critical for role distinction |
| **Cosine similarity** | Euclidean distance | Direction-based; normalized; robust for high-dimensional embeddings |
| **Average aggregation** | Max pooling, attention | Captures consistent patterns; reduces noise from single-frame situations |
| **1-second sampling** | Every frame or 30-second averages | Preserves tactical structure; not too sparse or redundant |

---

## Expected Results

After successful training:

1. **Low Reconstruction Error**
   - Model accurately predicts masked player positions
   - Indicates understanding of tactical structure

2. **Role Clustering**
   - t-SNE visualization shows players clustered by role
   - Similar embeddings for similar positions

3. **High Validation Scores**
   - Exact role match > 60%
   - Position group match > 80%

4. **Meaningful Attention**
   - Defenders attend to nearby defenders and opposing attackers
   - Attackers attend to defensive line and goalkeeper

The embeddings ("Player DNA") capture:
- **Tactical role**: Position on field
- **Movement patterns**: Velocity, sprinting behavior
- **Relationships**: Teammate and opponent interactions

All learned purely from geometric data—no role labels used.
