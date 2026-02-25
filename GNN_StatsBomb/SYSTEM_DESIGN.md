# GNN-Based Player Similarity System — System Design

## Problem Statement

Given a query soccer player, find the most similar players in the dataset — where **similar** means: *if placed in the exact same game situation, both players would choose similar actions*.

This goes beyond surface-level statistics (goals, assists, pass completion). Two players are similar if, given the same pitch location, the same surrounding teammate/opponent positions, the same pressure state, and the same game context, they would make the same decisions — pass in the same direction, carry forward, attempt a dribble, or shoot.

### Input / Output

| | Description |
|---|---|
| **Input** | A query player ID |
| **Output** | Ranked list of similar players with cosine similarity scores, plus per-situation counterfactual comparisons showing predicted action distributions for the query and each candidate in identical game states |

---

## Dataset

### Source: StatsBomb Open Data with 360 Frames

The system uses [StatsBomb's open-data repository](https://github.com/statsbomb/open-data), which provides:

- **Event data**: Every on-ball action in a match (passes, shots, carries, duels, etc.) with spatial coordinates, timestamps, actor identity, and action outcomes.
- **360 freeze-frame data**: For supported matches, the positions of all visible players (teammates and opponents) at the moment of each on-ball event. This is the spatial context that defines the "situation."

StatsBomb coordinates place the acting team attacking left-to-right (x ∈ [0, 120], y ∈ [0, 80]).

### Event Types Retained (14)

Ball Recovery, Block, Carry, Clearance, Dribble, Duel, Foul Committed, Foul Won, Goal Keeper, Interception, Miscontrol, Pass, Pressure, Shot.

### Scale

From the available competitions/seasons with 360 data: ~51,000 possessions, ~737,000 events, covering hundreds of players across multiple leagues.

---

## Pipeline Overview

The system is a 7-phase pipeline:

```
Phase 1: Data Preparation & Feature Encoding
    ↓
Phase 2: Possession Construction
    ↓
Phase 3: Heterogeneous Graph Construction
    ↓
Phase 4: Model Architecture (defined here, instantiated at training)
    ↓
Phase 5: Training (masked imitation + contrastive learning)
    ↓
Phase 6: Inference (embedding generation + similarity search)
    ↓
Phase 7: Analysis (situation-level counterfactual comparison)
```

---

## Phase 1: Data Preparation & Feature Encoding

**Goal**: Convert raw StatsBomb JSON events into a fixed-length numeric vector per event.

Each event is encoded into a **122-dimensional** feature vector:

| Index Range | Group | Dims | Description |
|---|---|---|---|
| 0–13 | Event type | 14 | One-hot encoding of the action type |
| 14–15 | Location | 2 | Normalised (x/120, y/80) ball position |
| 16–18 | End location | 3 | (end_x, end_y, has_end_flag) |
| 19–20 | Delta | 2 | (dx, dy) displacement |
| 21–22 | Distance & angle | 2 | Euclidean distance and signed angle of delta |
| 23–31 | Play pattern | 9 | One-hot (Regular Play, From Corner, etc.) |
| 32–57 | Position | 26 | One-hot player position (Left Wing, Center Back, etc.) |
| 58–64 | Body part | 7 | One-hot (Left Foot, Right Foot, Head, etc.) |
| 65–94 | Outcomes | 30 | One-hot pass/shot/dribble outcomes + pass type/height + shot type |
| 95–103 | Scalars | 9 | Duration, under_pressure, counterpress, pass_length, pass_angle, etc. |
| 104–112 | Pitch zone | 9 | 3×3 grid one-hot (thirds × lanes) |
| 113–121 | Spatial 360 | 9 | From 360 freeze frames: teammate count, opponent count, keeper flag, mean teammate position (x, y), mean opponent position (x, y), min distance to teammate, min distance to opponent |

Left and right positions are kept distinct (not mirrored) so that preferred foot and tactical side are preserved.

**Outputs**: `event_features.npy` (N × 122), `event_metadata.parquet`, `freeze_frames.pkl`.

### Data Processing Reasoning

- **Why 360 data**: Standard event data tells you *what* happened but not *where everyone else was*. Two passes from the same location can be radically different decisions depending on whether a defender is blocking the lane. The 360 freeze frames provide this spatial context, which is essential for defining "same situation" — without it, the model cannot distinguish a creative through-ball under heavy pressure from a routine square pass in open space.
- **Why these 14 event types**: These are the on-ball actions that reflect decision-making. Events like "Starting XI", "Half Start", or "Substitution" are administrative and carry no spatial or tactical information. Pressure events are retained because applying a press is an active defensive decision with spatial context.
- **Why fixed-length vectors over learned tokenisation**: Each event mixes categorical fields (event type, position) with continuous fields (coordinates, xG) and structural fields (360 counts). A fixed encoding makes the feature space transparent and debuggable — we can inspect exactly which dimensions correspond to which information and mask specific groups during training.
- **Why coordinates are normalised to [0, 1]**: StatsBomb already standardises pitch orientation (acting team always attacks left-to-right), so raw coordinates are directly comparable across events. Dividing by pitch dimensions (120 × 80) brings all spatial features onto the same scale as the binary/one-hot features.
- **Why left/right positions are kept distinct**: A Left Wing and a Right Wing make systematically different decisions (preferred foot, cut-inside vs. go-wide, crossing angle). Mirroring would merge these into one role, destroying signal the model needs for per-player behavioural differentiation.
- **Why Spatial_360 is included in the event vector (then zeroed for the GNN)**: The 9-dim summary (counts and distances) is useful for non-GNN consumers of the feature matrix. For the GNN, these summary stats are redundant with the explicit player nodes and `context_for` edges — zeroing them forces the GNN to learn spatial reasoning from the graph structure rather than leaking aggregate statistics through the event features.

---

## Phase 2: Possession Construction

**Goal**: Group events into possession sequences — continuous spells where one team controls the ball.

Each possession is defined by StatsBomb's `possession_number` and `possession_team_id` within a match. Events within a possession are sorted temporally. Possession-level labels are computed:

- `ends_in_shot` (bool)
- `ends_in_goal` (bool)
- `total_xg` (float)

Possessions with fewer than 2 events or more than 200 are discarded.

Each possession also stores per-event timestamps (seconds from match start, computed as `minute × 60 + second`), used downstream in Phase 3 for computing time-delta edge attributes on temporal edges.

**Output**: `possessions.pkl` — list of `Possession` objects, each containing global event indices, per-event metadata, and per-event timestamps.

### Data Processing Reasoning

- **Why possession as the unit of analysis**: A possession is the natural unit of team play — a continuous spell of ball control during which a team makes a sequence of decisions. Splitting events by possession (rather than by match or by fixed time windows) ensures that every graph represents a coherent tactical situation with a beginning, a series of decisions, and an end. The player making each decision does so *within* the context of what happened earlier in that possession.
- **Why min 2 events**: A single-event possession has no temporal structure — there's no sequence of decisions to learn from, and no temporal edges to build.
- **Why max 200 events**: Extremely long possessions (>200 events) are statistical outliers — likely data artefacts, prolonged keeper back-passes, or edge cases that would dominate graph memory without adding proportionate learning signal. The cap ensures consistent batch sizes during training.
- **Why per-possession labels (shot, goal, xG)**: These provide a secondary training signal. The outcome prediction head learns to recognise which game states lead to scoring opportunities, adding a task-relevant inductive bias to the event representations.

---

## Phase 3: Heterogeneous Graph Construction

**Goal**: Convert each possession into a heterogeneous graph that encodes the full game situation — the sequence of actions, who performed them, and where every player was standing.

### Node Types

| Type | Features | Description |
|---|---|---|
| `event` | 122-D vector (Spatial_360 block zeroed) | One node per on-ball action in the possession. The 360 summary stats are zeroed so the GNN must learn spatial context from explicit player nodes. |
| `player` | [position_idx, is_possession_team, dx, dy] | One node per distinct player. **Actor** players (who performed events) get their real position and player_id. **Off-ball** players (from 360 freeze frames) get their spatial offset from the ball. |

### Edge Types (Relations)

| Relation | Edge Attr | Description |
|---|---|---|
| `(event, next, event)` | 1-D time delta | Temporal ordering: event t → event t+1 |
| `(event, prev, event)` | 1-D time delta | Reverse temporal: event t+1 → event t |
| `(player, acts_in, event)` | — | Actor performed this event |
| `(event, performed_by, player)` | — | Reverse of acts_in |
| `(player, context_for, event)` | — | Off-ball player visible during this event (from 360 data) |

**Temporal edge attributes**: Each `next` and `prev` edge carries a 1-D time-delta attribute representing the elapsed time between the two connected events, normalised as `min(Δt / 30, 1.0)`. This gives the GNN tempo information: a rapid pass-carry sequence (0.5s gaps) produces edge attributes near 0, while a slow buildup (15s gaps) produces attributes near 0.5, and filtered-out intermediate events (30s+ gaps) saturate at 1.0.

The 360 freeze frames are critical: they create `context_for` edges that tell the GNN about teammate and opponent positioning. This is the spatial context that makes a "situation" well-defined.

**Output**: `possession_graphs.pkl` — list of `HeteroData` objects.

### Data Processing Reasoning

- **Why heterogeneous graphs (not sequences)**: A possession is more than a sequence of events. Each event involves an actor, and that event occurs in the context of where all other players are standing. A flat sequence model (LSTM, Transformer over events) would need the 360 spatial context injected as auxiliary features on each event — losing the explicit structure of "player X is 5 metres to the left." A heterogeneous graph natively encodes: temporal ordering (event → event edges), who did what (player ↔ event edges), and who was where (context player → event edges). The GNN can then reason about all of these simultaneously.
- **Why separate node types for events and players**: Events and players are fundamentally different entities. An event has 122 features describing what happened; a player node has 4 features describing who they are and where they stand. Heterogeneous typing lets the model learn separate projection and attention parameters for each, rather than forcing both into a single feature space.
- **Why off-ball players get `Unknown` position**: Off-ball players from 360 freeze frames don't carry position labels in the StatsBomb data — only the actor's position is known. Rather than guessing or omitting them, they receive the `Unknown` position embedding (learned from data), and their spatial offset (dx, dy) from the ball carries the crucial information about where they are.
- **Why actor players get dx=0, dy=0**: The actor is at the ball by definition. Their spatial information is already encoded in the event node's location. The actor node's value comes from the position embedding and team flag, not spatial offset.
- **Why bidirectional temporal edges**: The `next` edges propagate information forward in time (what happened earlier informs the current state). The `prev` edges propagate backward (the eventual outcome of the possession influences the interpretation of earlier events). Together they let every event node attend to the full temporal context of the possession.
- **Why time-delta edge attributes on temporal edges**: The graph structure encodes *ordering* but not *tempo*. A pass-carry-pass in 2 seconds (counterattack) and the same sequence over 15 seconds (slow buildup) produce identical graph structures but represent fundamentally different situations requiring different player decisions. The time delta gives the GNN's attention mechanism a direct signal about urgency and rhythm. Normalising by 30 seconds and clipping at 1.0 gives good resolution in the 0-30s range while clearly flagging gaps where intermediate events were filtered out.

---

## Phase 4: Model Architecture

### Overview

The model learns a **player trait embedding** `z_p` (64-D vector) for each player such that players with similar decision-making behavior in identical situations get similar embeddings. It does this through a masked imitation objective: given a game state, predict what action the player took.

### Components

#### 1. Feature Projections

- **EventProjection**: 122-D → 64-D via MLP (Linear → ReLU → Linear → LayerNorm)
- **PlayerProjection**: Position embedding (26 positions → 16-D) concatenated with 3 continuous features (team flag, dx, dy) → MLP → 64-D

#### 2. Heterogeneous GNN Encoder (2-layer GATv2)

A 2-layer heterogeneous Graph Attention Network (GATv2) processes the possession graph. Each layer uses `HeteroConv` with per-relation `GATv2Conv` — every edge type has its own learned attention weights. Temporal edge types (`next`, `prev`) use `edge_dim=1` so that GATv2's attention mechanism conditions on the time-delta edge attribute; non-temporal edge types have no edge attributes.

- **Layer 1**: 64-D → 512-D (4 attention heads × 128-D per head, concatenated)
- **Layer 2**: 512-D → 64-D (single head, no concatenation)
- Learned linear skip connections from input to output are added after Layer 2, preventing over-smoothing by preserving the original signal

After encoding, both `event` and `player` nodes live in the same 64-D latent space:

- `h_event` (E × 64): the **situation encoding** — what is happening in this game state
- `h_player` (P × 64): the **player-in-context encoding** — this player in this specific possession

#### 3. Attention Pooling (per-player aggregation)

Within each training batch, the same player may appear across multiple possessions. The attention pooling module aggregates their per-possession `h_player` embeddings into a single `z_p`:

```
score(h) = Linear(Tanh(Linear(h)))      →  scalar attention score
α_i = softmax(score(h_i))               →  attention weight per possession
z_p = Σ α_i · h_i                       →  pooled player trait embedding
```

This is trained end-to-end — gradients flow from action prediction through `z_p` back into the GNN. The learned scoring function can up-weight high-leverage possessions (shots, line-breaking plays) over routine recycling.

At inference time, the same pooling aggregates across all of a player's possessions to produce a single global `z_p`.

#### 4. FiLM Conditioning (player → action prediction)

To predict what action a player would take in a given situation, the model uses **Feature-wise Linear Modulation (FiLM)**:

```
γ = Linear(z_p)          →  per-dimension scale
β = Linear(z_p)          →  per-dimension shift
h_conditioned = γ ⊙ h_event + β
```

This is then passed to the action prediction heads (action type, direction, length).

**Why FiLM instead of concatenation**: With simple concatenation `[h_event; z_p]` → Linear, the model can learn to ignore `z_p` by assigning it near-zero weights, relying on `h_event` alone. FiLM creates **multiplicative dependence**: if `z_p` is the same for two players, they get *identical predictions in every situation*. The model is forced to differentiate `z_p` to explain any player-level behavioral variation. This makes z_p truly load-bearing.

#### 5. Prediction Heads

| Head | Input | Output | Purpose |
|---|---|---|---|
| Action type | FiLM-conditioned h (64-D) | 14 logits | Which action type? |
| Angle bin | FiLM-conditioned h (64-D) | 9 logits | Which direction (8 sectors + no-angle)? |
| Length bin | FiLM-conditioned h (64-D) | 5 logits | How far (very short to very long)? |
| Outcome | h_event (64-D) | 2 logits | Does this possession end in shot/goal? |

The outcome head uses `h_event` only (no `z_p`) because possession outcomes depend more on game state than individual player tendencies.

---

## Phase 5: Training

### Masked Imitation Objective

The core training signal is **masked action prediction**: given a game state with the action and player identity zeroed out, predict what the player actually did. The masking is applied at runtime — by the Phase 5 dataset class during training and by the Phase 6 embedding generator during inference (Phase 3 separately zeroes Spatial_360 during graph construction). The mask zeros out:

- Event type (indices 0–13) — this IS the prediction target
- End location, delta, distance/angle (16–22) — reveals the action
- Position (32–57) — player identity, not situation (see below)
- Body part (58–64) — reveals modality (foot vs head)
- All outcome fields (65–94) — reveals what happened
- Action-specific scalars (95, 98–103) — duration, pass length, etc.

What survives masking (the "situational state"):

- Location (14–15): where on the pitch
- Play pattern (23–31): open play, corner, free kick, etc.
- Under pressure / counterpress (96–97)
- Pitch zone (104–112): 3×3 grid zone
- Spatial 360 is zeroed at the event-feature level (handled by GNN via player nodes)

**Why this masking scheme**: The model must predict the action from the pre-action situational state — if the action type or its consequences are visible in the input, the model can trivially copy the answer. Everything that reveals *what happened* (event type, body part, outcomes, displacement, duration) is zeroed. Everything that describes *the state before the action* (location, play pattern, pressure, pitch zone) is preserved. The boundary is strict: even `duration` is masked because a 0.1s event (touch) vs. a 5s event (long carry) reveals the action type.

**Why position is masked from event features**: Position is a *player-level* attribute, not a situational one. Two events at the same location with the same surrounding players are the "same situation" regardless of whether the actor is a Left Wing or a Center Back. If position remains in the event features, the model can learn "Left Wings usually pass forward" directly from `h_event`, reducing its dependence on `z_p`. Despite FiLM, the `γ`/`β` parameters could converge toward identity if `h_event` already carries most of the predictive signal through position. Masking position forces it to flow through the correct channel: player node → `h_player` → `z_p` → FiLM, making `z_p` truly encode player identity including role.

### Loss Function

The total loss is:

```
L = L_action + 0.5 · L_outcome + 1.0 · L_contrastive
```

#### L_action: Focal Loss with Class Weights

Standard cross-entropy would be dominated by Pass (~38%) and Carry (~33%). The system uses **Focal Loss** to down-weight easy/frequent classes:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)    where γ = 2.0
```

Class weights (`1/√(count)`, normalised to mean 1) are applied to action type and length bin predictions, giving rare classes like Shot (~1.1%) and Dribble (~1.1%) significantly higher weight compared to Pass (~38%) and Carry (~33%). Angle bin predictions use Focal Loss without explicit class weights (only the focal modulation γ=2.0).

#### L_outcome: Binary Cross-Entropy

BCE on the shot/goal prediction head. Weight: 0.5.

#### L_contrastive: InfoNCE on h_player

InfoNCE contrastive loss applied to actor-only `h_player` embeddings (off-ball 360 players are excluded since they lack player IDs needed to form positive/negative pairs):

```
sim(i,j) = normalize(h_i) · normalize(h_j) / τ       where τ = 0.05
```

- **Positives**: Same player_id appearing across different possessions within the batch
- **Negatives**: Different player_ids

This serves a complementary role to the action loss:

- **Contrastive loss**: Ensures same-player GNN representations are consistent across different possessions and different-player representations are separated. Shapes the quality of `h_player`.
- **Action loss via FiLM**: Forces `z_p` (the pooled, global embedding) to capture behavioral differences between players, because predictions are multiplicatively dependent on it.

### Training Details

- Optimiser: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Batch size: 64 possession graphs
- Data split: 70/15/15 by match_id (prevents leakage — all possessions from a given match go to the same split, so the model cannot memorise match-specific patterns and leak them across train/val/test)
- Early stopping: patience=15, min_delta=1e-4

---

## Phase 6: Inference & Similarity Search

### 6A: Embedding Generation

For each possession graph in the dataset:

1. Apply future-info masking (same as training)
2. Run the GNN encoder to get `h_player` for every player node
3. Collect each player's `h_player` vectors across all their possessions
4. Attention-pool into a single global `z_p` per player (minimum 50 possessions required)

Processing uses mini-batch inference with PyG's `Batch.from_data_list()` for efficiency.

**Output**: `player_embeddings.npy` (n_players × 64), `player_info.parquet`.

### 6B: Similarity Search

Given a query player_id:

1. Compute pairwise cosine similarity between all `z_p` vectors
2. Rank by similarity, with optional filters (position group, minimum possessions, exclude same team)
3. Return top-k matches

---

## Phase 7: Analysis — Situation-Level Comparison

Phase 7 validates the similarity by answering: "In concrete game situations, would the query player and candidates actually make similar decisions?"

For each query player:

1. Find top-5 nearest neighbours from Phase 6
2. Sample 5 real game events from the query player's possessions
3. For each event, extract `h_event` (the situation encoding) from the GNN
4. Substitute each player's global `z_p` and run through FiLM + action heads
5. Compare the predicted action distributions

This produces:

- **Text reports**: Per-situation predicted action type, direction, and length distributions for all players
- **Bar charts**: Grouped bar charts comparing action probabilities across players for each situation
- **PCA plots**: 2D visualisation of the embedding space with highlighted neighbourhoods

---

## Rationale: Why This Architecture?

### Why a GNN?

A possession is naturally a graph: events are connected temporally, players are connected to the events they perform, and 360 freeze-frame players are spatially connected to events. A heterogeneous GNN natively represents all of these relationships and can learn from the full spatial context.

### Why masked imitation?

We don't have direct labels for "player similarity." Instead, we learn player embeddings indirectly: if the model can predict what action a player takes in a given situation, then the `z_p` vector must encode that player's decision-making tendencies. Players who make similar decisions get similar `z_p` vectors as a consequence of minimising the action prediction loss.

### Why FiLM conditioning?

The model must be forced to actually use `z_p` for prediction. With concatenation, the game state alone carries enough signal for a reasonable prediction — the model can collapse `z_p` to near-identical vectors for all players. FiLM makes `z_p` multiplicatively gate the state, so the model cannot ignore it.

### Why attention pooling?

Simple mean-pooling over all possessions would over-weight routine plays (backwards passes in one's own half). Attention pooling learns which possessions are most informative about a player's unique style and up-weights them.

### Why contrastive + action loss?

They address different aspects: the contrastive loss provides a direct embedding-space signal (same player = similar representations, different players = different), while the action loss provides a behavioural signal (z_p must explain what this specific player does differently). Together they shape both the representation quality and the behavioural meaningfulness of the embedding space.

---

## Key Configuration

| Parameter | Value | Description |
|---|---|---|
| `latent_dim` | 64 | Embedding dimensionality |
| `num_layers` | 2 | GNN depth |
| `num_heads` | 4 | Attention heads in layer 1 |
| `hidden_dim` | 128 | Per-head hidden dimension |
| `n_action_types` | 14 | Action type classes |
| `n_angle_bins` | 9 | Direction bins (8 sectors + no-angle) |
| `n_length_bins` | 5 | Displacement magnitude bins |
| `lambda_contrast` | 1.0 | Contrastive loss weight |
| `lambda_outcome` | 0.5 | Outcome loss weight |
| `temperature` | 0.05 | InfoNCE temperature |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `min_samples_per_player` | 50 | Minimum possessions for embedding |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-3 | Adam learning rate |

---

## Audit & Fixes Applied

The following issues were identified during a full code audit and corrected:

### 1. Validation loss now includes contrastive component

**Problem**: `_validate()` in `trainer.py` computed only `L_action + λ_outcome · L_outcome`, while `_train_epoch()` computed the full `L_action + λ_outcome · L_outcome + λ_contrast · L_contrastive`. This meant `ReduceLROnPlateau` and early stopping operated on a different objective than training — the scheduler could reduce LR or trigger early stopping even while the contrastive loss was still improving the embedding space.

**Fix**: Added actor-only `h_player` extraction and contrastive loss computation to `_validate()`, mirroring the training loop. Validation loss is now computed with the same formula as training loss.

### 2. Device selection standardised to CPU across all phases

**Problem**: Training (`trainer.py`) explicitly preferred CPU over MPS (because PyG heterogeneous-graph workloads with many small kernel launches run slower on Apple MPS), but inference (`main.py`) and analysis (`report_builder.py`) still attempted MPS before falling back to CPU. Since inference runs the same GNN encoder with the same workload characteristics, MPS would be equally slow.

**Fix**: Removed MPS preference from `run_inference()` in `main.py` and from `ReportBuilder.run()` in `report_builder.py`. All phases now use CUDA if available, otherwise CPU.

### 3. Removed dead `event_player_ids` from dataset collate

**Problem**: `collate_fn` in `dataset.py` manually concatenated `event_player_ids` from each graph and stored it in the batch dict. The trainer never accessed `batch["event_player_ids"]` — it instead used `data["player"].player_node_pids`, which PyG's `Batch.from_data_list()` automatically concatenates from the per-graph HeteroData objects.

**Fix**: Removed the redundant `event_player_ids` concatenation from `collate_fn` and the corresponding field from `__getitem__`.

### 4. Player position uses most-frequent assignment

**Problem**: `EmbeddingGenerator` in `embedding_generator.py` assigned each player's position from their first appearance in the metadata (`rows.iloc[0]`). A player who played multiple positions across matches would get an arbitrary (first-encountered) position label.

**Fix**: Changed to `rows["position_name"].mode().iloc[0]`, which selects the most frequently occurring position for each player.

### 5. Clarified `ends_in_shot` semantics

**Problem**: `ends_in_shot` in `possession_builder.py` was computed as `"Shot" in event_types` — checking whether *any* event in the possession is a Shot, not specifically the last event. The name suggests "ends in" but the logic is "contains."

**Resolution**: In StatsBomb data, shots almost always terminate the possession (a save, block, or goal resets play), so "contains shot" is a close approximation of "ends in shot" in practice. Added a clarifying comment to the code explaining this equivalence.

### 6. Fixed contrastive loss docstring (factual error)

**Problem**: The `ContrastiveLoss` docstring in `losses.py` claimed the loss was "state-aware" with positives defined as "same player in similar pitch zones." This was factually incorrect — the code forms positive pairs purely by matching `player_id` with no zone or state filtering.

**Fix**: Corrected the docstring to accurately state: positives are same `player_id` across different possessions within the batch; negatives are different `player_id`s.

### 7. Fixed training/inference pooling asymmetry

**Problem**: During training, `_gather_and_pool_actor_embeddings` in `model.py` operated over all `acts_in` edges. Since each event has one `acts_in` edge, a player who performs 10 events in one possession and 2 in another would contribute 10 copies of the *same* `h_player` vector from the first possession and 2 from the second. The attention pooling softmax then implicitly weighted possessions by event count — not purely by the learned attention scores. During inference, `EmbeddingGenerator` correctly deduplicates to one `h_player` per possession per player before pooling. This meant `z_p` was computed differently during training and inference.

**Fix**: Added deduplication by unique player node index (`torch.unique(src)`) before the attention pooling step. Since all `acts_in` edges from the same player within a single graph share the same player node (and thus the same `h_player`), this collapses to one embedding per possession per player. The attention pooling now operates on the same structure during training as during inference: one `h_player` per possession, weighted purely by the learned attention scores.

### 8. Position masked from event features to remove role bias

**Motivation**: The player's tactical position (indices 32-57) was present in event features, allowing the model to predict actions from role alone (e.g., "Center Backs clear the ball") without relying on `z_p`. This made `z_p` less load-bearing and caused player similarity to correlate with positional similarity rather than individual behavioral style. Position information still flows through player node → `h_player` → `z_p`, which is the correct channel for player identity.

**Changes**: Added `range(32, 58)` to the masked ranges in `masking.py`. Updated the docstring and variable naming (`_FUTURE_RANGES` → `_MASKED_RANGES`) to reflect that the mask now covers both future/action info and player identity.

### 9. Added time-delta edge attributes to temporal edges

**Motivation**: The graph structure encoded event ordering but not tempo. Two possessions with identical event sequences but vastly different pacing (2-second counterattack vs. 15-second buildup) produced identical graphs — the GNN had no way to distinguish them. Since tempo directly affects player decisions (time pressure forces different choices), this was a missing signal for achieving the system's goal.

**Changes**:
- `Possession` dataclass: added `timestamps_sec` field (seconds from match start, computed from `minute * 60 + second`).
- `PossessionGraphBuilder`: computes per-edge time deltas between consecutive events, normalises as `min(Δt / 30, 1.0)`, and stores as `edge_attr` (shape `(n_edges, 1)`) on both `next` and `prev` edge types.
- `PossessionGNNEncoder`: temporal `GATv2Conv` layers now use `edge_dim=1`. The `forward` method accepts an optional `edge_attr_dict` and passes it to `HeteroConv` via the `edge_attr_dict` kwarg pattern.
- `PlayerSimilarityModel`: `forward` and `encode_possession` extract `edge_attr` from `HeteroData` and route it to the GNN.

**Note**: This is an architectural change — existing checkpoints are incompatible and the full pipeline (Phase 2 → Phase 5) must be re-run.

---

## File Structure

```
GNN_StatsBomb/
├── main.py                          # CLI entry point for all phases
├── src/
│   ├── config.py                    # All configuration and vocabularies
│   ├── data_preparation.py          # Phase 1A: StatsBomb JSON loading
│   ├── feature_encoder.py           # Phase 1B: 122-D feature encoding
│   ├── phase2_possession/
│   │   └── possession_builder.py    # Phase 2: possession grouping
│   ├── phase3_graph/
│   │   ├── graph_builder.py         # Phase 3: HeteroData construction
│   │   └── masking.py               # Future-info and spatial masking
│   ├── phase4_model/
│   │   ├── model.py                 # Full model (FiLM + heads)
│   │   ├── gnn_encoder.py           # 2-layer heterogeneous GATv2
│   │   ├── projections.py           # Event and player feature projections
│   │   └── pooling.py               # Attention-weighted pooling
│   ├── phase5_training/
│   │   ├── trainer.py               # Training loop with early stopping
│   │   ├── losses.py                # Focal, contrastive, combined losses
│   │   ├── dataset.py               # PyG dataset with masking + targets
│   │   └── action_targets.py        # Discretise actions into bins
│   ├── phase6_inference/
│   │   ├── embedding_generator.py   # Batched z_p generation
│   │   └── similarity_search.py     # Cosine similarity top-k search
│   └── phase7_analysis/
│       ├── report_builder.py        # Orchestrates full analysis
│       ├── situation_comparison.py   # Counterfactual action prediction
│       └── embedding_viz.py         # PCA visualisations
├── processed_data/                  # Phase 1–3 outputs
├── checkpoints/                     # Trained model weights
└── embeddings/
    ├── player_embeddings.npy        # (n_players, 64)
    ├── player_info.parquet          # Player metadata
    └── analysis/                    # Phase 7 reports and plots
```
