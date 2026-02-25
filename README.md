# GNN Pass Recipient Prediction --Jacky Wang
## Using StatsBomb Open Data

---

## Project Overview

**Goal**: Build a Graph Neural Network (GNN) to predict which teammate will receive a pass in soccer, given only the spatial positions of visible players at the moment of the pass.

**Data Sources**:
- **Event files** — contain all match events including passes with origin/end locations and recipient IDs
- **360 files** — contain freeze frames with player positions at the moment of each event

Only matches with both event data **and** 360 data are used (326 matches).

**Key Constraint**: The model does NOT see the pass end location — it must learn passing patterns purely from spatial context.

---

## Problem Formulation

Given a pass event, the model sees:
- **Positions** of all visible players (from 360 freeze frame)
- Which players are **teammates** vs **opponents**
- Who is the **passer** (actor)
- Who is a **goalkeeper**
- Each player's **distance and angle** to the ball

**Task**: Select which teammate the passer will pass to.

**Why it matters**: Understanding passing decision-making enables players similarity analysis.

---

## Data Pipeline

1. **Identify matches with 360 data** (326 matches across multiple competitions)
2. **Extract pass events** from match event files
3. **Filter to completed passes only** (incomplete passes don't have reliable end locations)
4. **Link each pass** to its 360 freeze frame snapshot
5. **Build a graph** for each valid pass event
6. **Split** into Train (70%) / Val (15%) / Test (15%)

**Dataset Statistics**:
- 227,513 graphs (completed passes with 360 data)
- 43,455 incomplete passes filtered out
- 3–22 nodes per graph (mean ~15.3)
- 1–10 candidate recipients (teammates) per graph (mean ~6.6)

---

## Graph Construction — Overview

Each graph represents a **single completed pass event** combined with its **360 freeze frame**.

```
Pass Event + 360 Freeze Frame → Graph (Nodes + Edges + Target)
```

**Key Design Decisions**:
- Every visible player becomes a **node** (3–22 per graph)
- The graph is **fully connected** (every player pair has edges)
- Target = index of the teammate node closest to `pass.end_location`
- `pass.end_location` is **excluded** from node features (prevents leakage)

---

## Graph Construction — Node Features (7 per node)

| Feature         | Description                                         | Range      |
|-----------------|-----------------------------------------------------|------------|
| `x_norm`        | Player x-position, normalized                       | [0, 1]     |
| `y_norm`        | Player y-position, normalized                      | [0, 1]     |
| `is_teammate`   | 1.0 if same team as passer, 0.0 otherwise            | {0, 1}     |
| `is_actor`      | 1.0 if this player IS the passer, 0.0 otherwise      | {0, 1}     |
| `is_keeper`     | 1.0 if goalkeeper, 0.0 otherwise                      | {0, 1}     |
| `dist_to_ball`  | Euclidean distance to pass origin (normalized coords) | ≥ 0        |
| `angle_to_ball` | Angle from pass origin to player (radians / π)        | [-1, 1]    |

**Coordinate normalization**: StatsBomb's 120×80 yard pitch coordinates → [0, 1] range via `x / 120`, `y / 80`.

---

## Graph Construction — Edge Construction

**Fully connected** graph: every pair of players has a directed edge in both directions (no self-loops).

For a graph with *n* nodes → *n(n-1)* directed edges.

```python
src = [i for i in range(n) for j in range(n) if i != j]
dst = [j for i in range(n) for j in range(n) if i != j]
```

**Rationale**: Every player on the pitch can potentially influence every other player's decision. The GAT's attention mechanism learns **which** relationships actually matter.

---

## Graph Construction — Candidate Mask & Target

**Candidate Mask** (boolean tensor per node):
- A node is a valid prediction target if and only if:
  - `is_teammate == True` AND `is_actor == False`
- Only these nodes can be selected as the predicted recipient

**Target Label**:
- The index of the teammate node **closest to `pass.end_location`**
- Since only **completed passes** are used, the end location reliably corresponds to the actual recipient
- This acts as a proxy for "who received the pass"

---

## Graph Construction — Output Data Object

Each graph is a `torch_geometric.data.Data` object:

| Attribute          | Type / Shape        | Description                              |
|--------------------|---------------------|------------------------------------------|
| `x`                | Float `[n, 7]`      | Node features                            |
| `edge_index`       | Long `[2, n(n-1)]`  | Fully connected edges                    |
| `y`                | Long scalar          | Index of the recipient node              |
| `candidate_mask`   | Bool `[n]`           | Which nodes are valid targets            |
| `num_candidates`   | Int                  | Count of candidate nodes                 |

---

## GAT Model — Architecture Overview

**PassRecipientGAT** — ~36,800 trainable parameters

```
Raw Features (7d)
  → Linear Projection (7 → 64) 
  → GAT Block 1 (4 heads, 64d, residual + BN + dropout)
  → GAT Block 2 (4 heads, 64d, residual + BN + dropout)
  → Score Head (64 → 32 → 1) per node
  → Scalar score per player
```

**Key Architectural Choices**:
- 2 GAT layers with **residual connections** (prevents over-smoothing)
- **Batch normalization** after each GAT layer
- MLP scoring head produces a **single scalar** per node

---

## GAT Model — Input Projection

```python
self.input_proj = nn.Linear(in_channels, hidden_channels)  # 7 → 64
```

- Maps 7-dimensional raw node features to the hidden dimension (64)
- Output goes through activation

---

## GAT Model — GAT Blocks (×2)

```python
self.gat1 = GATConv(64, 64, heads=4, concat=False, dropout=0.2)
self.bn1  = nn.BatchNorm1d(64)
```

**Inside each GAT block**:
1. **4 attention heads** independently compute attention coefficients for every edge
2. Each head: a learnable vector applied to concatenated neighbor features → LeakyReLU → softmax
3. Weighted sum of neighbor features using attention weights
4. **BatchNorm** → **Dropout** (p=0.2)
5. **Residual connection**: `h = h + dropout(bn(gat(h)))`

---

## GAT Model — Scoring Head

```python
self.score_head = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1),
)
```

- A 2-layer MLP that maps each node's 64-dimensional learned representation to a **single scalar score**
- The score represents "how likely this player is to be the pass recipient"
- Applied **independently** to each node

---



## Results

**Baseline**: Random guessing with ~6.6 candidates → ~15% top-1 accuracy

**GNN Performance**: After training, the model achieves **~42.5% top-1 accuracy** on validation, significantly outperforming random baseline.

**Key Observations**:
- The model learns meaningful spatial passing patterns
- Accuracy is higher when there are fewer candidates (easier decision)
- The model consistently outperforms the random baseline across all candidate set sizes




## Future Work — Player Similarity from Pass Prediction

The trained GAT produces a **64-dim embedding** per node that captures tactical context. This can be leveraged for player similarity analysis.


**Approach — Learnable Player Identity Embeddings** (requires lineup data):
- Link freeze frame positions to StatsBomb **lineup data** to identify players
- Add a **per-player embedding** trained end-to-end with pass prediction
- Directly compute **player-to-player similarity** from learned embeddings.

---

