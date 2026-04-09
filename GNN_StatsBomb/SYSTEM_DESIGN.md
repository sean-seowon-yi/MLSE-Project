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

With the default StatsBomb open-data snapshot (all competition–seasons that ship 360 files, no extra filters), a full Phase 1–2 run on a typical checkout yields on the order of **737,020** on-ball events (with 360 frames), **51,778** possessions after the min/max event filters, **323** matches, and **7** distinct `(competition_id, season_id)` pairs. Counts change if StatsBomb adds matches, you pass `--competition` / `--season`, or you set `use_360=False`. After Phase 5’s minimum-possession filter, about **1,633** players receive embeddings.

---

## Pipeline Overview

The system is a 7-phase pipeline, exposed as CLI modes via `main.py`:

```
Phase 1: Data Preparation & Feature Encoding          --mode prepare
    ↓
Phase 2: Possession Construction                       --mode build_possessions
    ↓
Phase 3: Heterogeneous Graph Construction              --mode build_graphs
    ↓
Phase 4: Model Architecture (defined here, instantiated at training)
    ↓
Phase 5: Training (masked imitation + contrastive)     --mode train
         Test-set evaluation                           --mode evaluate
    ↓
Phase 6: Embedding generation                          --mode inference
         Similarity search                             --mode search
         Pseudo ground-truth evaluation                --mode ground_truth
         Self-consistency evaluation                   --mode self_consistency
         Policy diagnostic                             --mode policy_diagnostic
         Embedding-only metrics                        --mode position_retrieval | split_half | qualitative_neighbors
    ↓
Phase 7: Analysis (situation-level comparison)         --mode analyze
         FIFA stat comparison                          --mode fifa_comparison
         Possession animation (MP4/GIF visualisation)   --mode possession_animation
         Empirical behavioral fidelity (standalone)    --mode empirical_behavioral
         Full evaluation (single pipeline)             --mode full_eval
         Full evaluation (all variants)                --mode full_eval_all
         Heuristics + same eval layout                  --mode generate_heuristics | eval_heuristics | full_eval_all_with_heuristics
```

`--mode full_pipeline` runs Phases 1→2→3→5→6A end-to-end. Search and evaluation modes require embeddings to already exist.

### Experiment Tagging

The `--tag` flag namespaces **checkpoints and embeddings** while sharing Phase 1–2 data and (per `graphs_filename`) the shared graph pickle(s). Multiple tags can reuse the same `possession_graphs.pkl` or `possession_graphs_split_ctx.pkl` when their graph topology matches. The `--split_context_edges` flag must be passed consistently across all phases of the same experiment (it changes graph structure and model architecture). Combined workflow:

```
python main.py --mode build_graphs   --tag split_ctx --split_context_edges
python main.py --mode train          --tag split_ctx --split_context_edges
python main.py --mode evaluate       --tag split_ctx --split_context_edges
python main.py --mode inference      --tag split_ctx --split_context_edges
python main.py --mode ground_truth   --tag split_ctx --split_context_edges
python main.py --mode analyze        --tag split_ctx --split_context_edges
```

---

## Phase 1: Data Preparation & Feature Encoding

**Goal**: Convert raw StatsBomb JSON events into a fixed-length numeric vector per event.

Each event is encoded into a **126-dimensional** feature vector:

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
| 122–125 | Period | 4 | One-hot match period (Period 1, Period 2, Extra Time 1, Extra Time 2) |

Left and right positions are kept distinct (not mirrored) so that preferred foot and tactical side are preserved.

**Outputs**: `event_features.npy` (N × 126), `event_metadata.parquet`, `freeze_frames.pkl`.

### Data Processing Reasoning

- **Why 360 data**: Standard event data tells you *what* happened but not *where everyone else was*. Two passes from the same location can be radically different decisions depending on whether a defender is blocking the lane. The 360 freeze frames provide this spatial context, which is essential for defining "same situation" — without it, the model cannot distinguish a creative through-ball under heavy pressure from a routine square pass in open space.
- **Why these 14 event types**: These are the on-ball actions that reflect decision-making. Events like "Starting XI", "Half Start", or "Substitution" are administrative and carry no spatial or tactical information. Pressure events are retained because applying a press is an active defensive decision with spatial context.
- **Why fixed-length vectors over learned tokenisation**: Each event mixes categorical fields (event type, position) with continuous fields (coordinates, xG) and structural fields (360 counts). A fixed encoding makes the feature space transparent and debuggable — we can inspect exactly which dimensions correspond to which information and mask specific groups during training.
- **Why coordinates are normalised to [0, 1]**: StatsBomb already standardises pitch orientation (acting team always attacks left-to-right), so raw coordinates are directly comparable across events. Dividing by pitch dimensions (120 × 80) brings all spatial features onto the same scale as the binary/one-hot features.
- **Why left/right positions are kept distinct**: A Left Wing and a Right Wing make systematically different decisions (preferred foot, cut-inside vs. go-wide, crossing angle). Mirroring would merge these into one role, destroying signal the model needs for per-player behavioural differentiation.
- **Why Spatial_360 is included in the event vector (then zeroed for the GNN)**: The 9-dim summary (counts and distances) is useful for non-GNN consumers of the feature matrix. For the GNN, these summary stats are redundant with the explicit player nodes and `context_for_tm` / `context_for_opp` edges — zeroing them forces the GNN to learn spatial reasoning from the graph structure rather than leaking aggregate statistics through the event features.

---

## Phase 2: Possession Construction

**Goal**: Group events into possession sequences — continuous spells where one team controls the ball.

Each possession is defined by StatsBomb's `possession_number` and `possession_team_id` within a match (see `possession_builder.py`: `groupby` on `match_id`, `possession_number`, `possession_team_id` — **`period` is not part of the key**). Events within a group are sorted by `(period, minute, second, original row order)`. Possession-level labels are computed from **those rows only** (Phase 1 may have dropped events without 360 or outside the allowed event types):

- `ends_in_shot` (bool)
- `ends_in_goal` (bool)
- `total_xg` (float)

Possessions with fewer than 2 events or more than 200 are discarded.

Each possession also stores per-event timestamps with millisecond precision (parsed from StatsBomb's period-relative `timestamp` field, e.g. `"00:23:15.432"` → `1395.432` seconds), used downstream in Phase 3 for computing time-delta edge attributes on temporal edges. **Caveat:** In rare cases StatsBomb can attach the same `(possession_number, possession_team_id)` across a period boundary (e.g. events straddling half-time). Because the grouping key omits `period`, those rows can land in one `Possession`; consecutive-event Δt can then be large or reflect a stoppage. For almost all open-play sequences, all events in a group share one period and deltas behave as intended.

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
| `event` | 126-D vector (Spatial_360 block zeroed) | One node per on-ball action in the possession. The 360 summary stats are zeroed so the GNN must learn spatial context from explicit player nodes. |
| `player` | [position_idx, is_possession_team, dx, dy] | **Actors:** one node per distinct `player_id` in the possession (`("actor", player_id)` in code), with real position, `dx=dy=0` at the ball. **Off-ball (360):** one node per **freeze-frame slot at each event** — keyed as `("tm", event_index, k)` or `("opp", event_index, k)` in `graph_builder.py`, not merged across events or stable StatsBomb player IDs; each carries `Unknown` position and `(dx, dy)` offset from the ball at that event. |

### Edge Types (Relations)

| Relation | Edge Attr | Description |
|---|---|---|
| `(event, next, event)` | 1-D time delta | Temporal ordering: event t → event t+1 |
| `(event, prev, event)` | 1-D time delta | Reverse temporal: event t+1 → event t |
| `(player, acts_in, event)` | — | Actor performed this event |
| `(event, performed_by, player)` | — | Reverse of acts_in |
| `(player, context_for, event)` | — | Off-ball player visible during this event (from 360 data). **Default** edge type when `split_context_edges=False`. |
| `(player, context_for_tm, event)` | — | **Teammate** off-ball player (passing options, support runs). Used when `--split_context_edges` is set. |
| `(player, context_for_opp, event)` | — | **Opponent** off-ball player (defensive pressure, blocks). Used when `--split_context_edges` is set. |

> **Ablation flag** (`GraphConfig.split_context_edges`, default `False`): When set to `True` via `--split_context_edges`, the single `context_for` relation is split into `context_for_tm` and `context_for_opp`, giving `HeteroConv` separate projection/attention weights for offensive options vs defensive pressure. The default (`False`) preserves backward compatibility with existing checkpoints.

**Temporal edge attributes**: Each `next` and `prev` edge carries a 1-D time-delta attribute representing the elapsed time between the two connected events, normalised as `min(Δt / 30, 1.0)`. This gives the GNN tempo information: a rapid pass-carry sequence (0.5s gaps) produces edge attributes near 0, while a slow buildup (15s gaps) produces attributes near 0.5, and filtered-out intermediate events (30s+ gaps) saturate at 1.0.

The 360 freeze frames are critical: they create context edges that tell the GNN about off-ball player positioning. When `--split_context_edges` is used, these are split into `context_for_tm` (teammate) and `context_for_opp` (opponent), allowing `HeteroConv` to learn separate projection matrices and attention weights for offensive options vs defensive pressure, rather than forcing the attention mechanism to distinguish teams using only the `is_possession_team` node feature. This is the spatial context that makes a "situation" well-defined.

**Output**: Graphs are written to `processed_data/` using `Config.graphs_filename`: `possession_graphs.pkl` when `split_context_edges=False`, or `possession_graphs_split_ctx.pkl` when `split_context_edges=True`. The filename keys off **graph topology**, not `--tag` — tagged experiments share the same graph file when their split-context setting matches (see `apply_tag` in `config.py`). Checkpoints and embeddings are still namespaced by tag.

### Data Processing Reasoning

- **Why heterogeneous graphs (not sequences)**: A possession is more than a sequence of events. Each event involves an actor, and that event occurs in the context of where all other players are standing. A flat sequence model (LSTM, Transformer over events) would need the 360 spatial context injected as auxiliary features on each event — losing the explicit structure of "player X is 5 metres to the left." A heterogeneous graph natively encodes: temporal ordering (event → event edges), who did what (player ↔ event edges), and who was where (teammate / opponent context → event edges). The GNN can then reason about all of these simultaneously.
- **Why separate node types for events and players**: Events and players are fundamentally different entities. An event has 126 features describing what happened; a player node has 4 features describing who they are and where they stand. Heterogeneous typing lets the model learn separate projection and attention parameters for each, rather than forcing both into a single feature space.
- **Why off-ball context is one node per (event, slot)**: A single physical opponent could appear at different offsets at successive events; reusing one node per player would overwrite geometry. The builder therefore instantiates **separate** context nodes per event and per teammate/opponent slot in the freeze frame (see `_process_freeze_frame` in `graph_builder.py`). They use the `Unknown` position embedding (freeze-frame entries are not given tactical positions like the actor), and `(dx, dy)` from the ball encodes where they stand for that snapshot.
- **Why actor players get dx=0, dy=0**: The actor is at the ball by definition. Their spatial information is already encoded in the event node's location. The actor node's value comes from the position embedding and team flag, not spatial offset.
- **Why bidirectional temporal edges**: The `next` edges propagate information forward in time (what happened earlier informs the current state). The `prev` edges propagate backward (the eventual outcome of the possession influences the interpretation of earlier events). Together they let every event node attend to the full temporal context of the possession.
- **Why time-delta edge attributes on temporal edges**: The graph structure encodes *ordering* but not *tempo*. A pass-carry-pass in 2 seconds (counterattack) and the same sequence over 15 seconds (slow buildup) produce identical graph structures but represent fundamentally different situations requiring different player decisions. The time delta gives the GNN's attention mechanism a direct signal about urgency and rhythm. Normalising by 30 seconds and clipping at 1.0 gives good resolution in the 0-30s range while clearly flagging gaps where intermediate events were filtered out.

---

## Phase 4: Model Architecture

### Overview

The model learns a **player trait embedding** `z_p` (64-D vector) for each player such that players with similar decision-making behavior in identical situations get similar embeddings. It does this through a masked imitation objective: given a game state, predict what action the player took.

### Components

#### 1. Feature Projections

- **EventProjection**: 126-D → 64-D via MLP (Linear → ReLU → Linear → LayerNorm)
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

#### 4. FiLM Conditioning (player + position → action prediction)

To predict what action a player would take in a given situation, the model uses **Feature-wise Linear Modulation (FiLM)** with a **dual-channel position design**:

```
pos_emb = Embedding(position_idx)        →  16-D position embedding
cond    = [z_p ; pos_emb]               →  (64 + 16 = 80)-D conditioning vector
γ       = Linear(cond)                   →  per-dimension scale
β       = Linear(cond)                   →  per-dimension shift
h_conditioned = (1 + γ) ⊙ h_event + β
```

This is then passed to the action prediction heads (action type, direction, length).

**Why FiLM instead of concatenation**: With simple concatenation `[h_event; z_p]` → Linear, the model can learn to ignore `z_p` by assigning it near-zero weights, relying on `h_event` alone. FiLM creates **multiplicative dependence**: if `z_p` is the same for two players, they get *identical predictions in every situation*. The model is forced to differentiate `z_p` to explain any player-level behavioral variation. This makes z_p truly load-bearing.

**Dual-channel position design**: Position enters `z_p` through the player-node path (`PlayerProjection` position embedding → GNN → `h_player` → attention pool → `z_p`), retaining coarse role structure (e.g. GK far from outfield). It *also* enters FiLM through a separate, dedicated embedding, which sharpens within-role prediction without requiring `z_p` to carry the full positional burden. The pooled uniformity loss provides a direct gradient signal on `z_p` that prevents the model from collapsing all embeddings despite the FiLM shortcut.

#### 5. Prediction Heads

| Head | Input | Output | Purpose |
|---|---|---|---|
| Action type | FiLM-conditioned h (64-D) | 14 logits | Which action type? |
| Angle bin | FiLM-conditioned h (64-D) | 9 logits | Which direction (8 sectors + no-angle)? |
| Length bin | FiLM-conditioned h (64-D) | 5 logits | How far (very short to very long)? |
| Outcome | h_event (64-D) | 2 logits | Does this possession end in shot/goal? |

The outcome head uses `h_event` only (no `z_p`) because possession outcomes depend more on game state than individual player tendencies.

#### 6. Position Ablation (optional)

When `ModelConfig.ablate_position = True` (CLI: `--ablate_position`), all explicit position information is removed from the learned representations:

- **PlayerProjection**: The position embedding table is not used; only the three continuous features (team flag, dx, dy) are passed through the MLP.
- **FiLM**: With ablation, `film_pos_embed` is `None` and `film_condition` uses **`z_p` alone** (64-D) for `film_gamma` / `film_beta` — there is no concatenated zero padding; the linear layers are sized for 64-D input in this mode.

Position indices remain in graph metadata for contrastive hard-negative mining. This forces the model to learn behavioral representations without positional shortcuts, allowing `z_p` to encode purely stylistic traits rather than role assignment.

The `pos_ablated_split_ctx` family combines this with split-context edges and increased uniformity pressure (λ_pooled=1.0, t=4.0), yielding **among the strongest** retrieval and cross-competition metrics in the suite; the top weighted composite score in `EVALUATION_RESULTS.md` is `acts_in_dropout_pos_gu`, which builds on the same split-context + ablation stack with additional regularizers.

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
- Period (122–125): match stage (first half, second half, extra time)
- Spatial 360 is zeroed at the event-feature level (handled by GNN via player nodes)

**Why this masking scheme**: The model must predict the action from the pre-action situational state — if the action type or its consequences are visible in the input, the model can trivially copy the answer. Everything that reveals *what happened* (event type, body part, outcomes, displacement, duration) is zeroed. Everything that describes *the state before the action* (location, play pattern, pressure, pitch zone) is preserved. The boundary is strict: even `duration` is masked because a 0.1s event (touch) vs. a 5s event (long carry) reveals the action type.

**Why position is masked from event features**: Position is a *player-level* attribute, not a situational one. Two events at the same location with the same surrounding players are the "same situation" regardless of whether the actor is a Left Wing or a Center Back. If position remains in the event features, the model can learn "Left Wings usually pass forward" directly from `h_event`, reducing its dependence on `z_p`. Despite FiLM, the `γ`/`β` parameters could converge toward identity if `h_event` already carries most of the predictive signal through position. Masking position forces it to flow through the correct channel: player node → `h_player` → `z_p` → FiLM, making `z_p` truly encode player identity including role.

### Loss Function

The total loss is:

```
L = L_action + λ_outcome · L_outcome + λ_contrast · L_contrastive + λ_pooled · L_pooled_uniformity [+ λ_alignment · L_alignment] [+ λ_pos · L_pos_group]
```

Default training weights (see `TrainingConfig` in `config.py`): λ_outcome = 0.5, λ_contrast = 0.5, λ_pooled = 0.3 (`lambda_pooled_contrast`), λ_alignment = 0.3 when EMA alignment is enabled, λ_pos = 0.0 unless using a pipeline that sets `lambda_pos` (e.g. `acts_in_dropout_pos`). The alignment term is optional (`ema_alignment=True`). All weights are configurable.

#### L_action: Focal Loss with Class Weights

Standard cross-entropy would be dominated by Pass (~38%) and Carry (~33%). The system uses **Focal Loss** to down-weight easy/frequent classes:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)    where γ = 2.0
```

Class weights (`1/√(count)`, normalised to mean 1) are applied to action type and length bin predictions, giving rare classes like Shot (~1.1%) and Dribble (~1.1%) significantly higher weight compared to Pass (~38%) and Carry (~33%). Angle bin predictions use Focal Loss without explicit class weights (only the focal modulation γ=2.0).

#### L_outcome: Binary Cross-Entropy

BCE on the shot/goal prediction head. Weight: 0.5.

#### L_contrastive: InfoNCE on actor-only h_player with hard negatives

InfoNCE contrastive loss applied to **actor-only `h_player` embeddings** with **same-position-group hard negatives**:

```
sim(i,j) = normalize(z_i) · normalize(z_j) / τ       where τ = 0.05
```

- **Positives**: Same `player_id` appearing across different events / possessions within the batch (multiple `h_player` rows for the same player).
- **Negatives**: Different `player_id`s **in the same coarse position group** (GK / Defender / Midfielder / Forward / Unknown), using the `POSITION_IDX_TO_GROUP` mapping derived from `POSITIONS`. When no same-group negatives exist for an anchor, all different-player pairs are used as fallback.

Hard negatives prevent the loss from being satisfied by merely encoding positional role. The model must learn fine-grained individual differences within each role.

The contrastive softmax denominator includes **positive pairs and hard negatives** (standard supervised contrastive / InfoNCE form), not negatives alone.

#### L_pooled_uniformity: Gaussian-potential uniformity on pooled z_p

Directly penalises pooled `z_p` vectors (the embeddings used for similarity search) that are too close on the unit hypersphere:

```
L_uniform = log E[ exp(-t · ||z_i - z_j||^2) ]     where t = 2.0
```

The expectation is over all pairs of distinct pooled players in the batch. This loss pushes all `z_p` vectors apart, widening cosine similarity gaps so that "similar" and "dissimilar" players occupy meaningfully different regions. Without it, pooled `z_p` can collapse into a narrow cone (cosine > 0.99 for all pairs) because the attention pooling averaging effect concentrates vectors toward the population mean.

Weight: 0.3.

#### L_alignment: EMA Cross-Batch Alignment (optional)

When `ema_alignment=True`, an additional loss term encourages a player's pooled `z_p` to remain consistent across batches:

```
L_alignment = 1 - cosine_similarity(z_p_current, z_p_ema)
```

An **EMA memory bank** maintains a smoothed historical embedding for each player (`momentum=0.999`). Before computing the loss, the bank is queried for each player in the batch; the alignment loss is averaged over players with valid history (i.e., seen in a previous batch). After the optimizer step, the bank is updated with the current batch's `z_p` (detached). The bank state is saved in checkpoints for resume compatibility.

This addresses a gap in the standard training: without this term, a player's `z_p` can differ across batches (where different subsets of possessions are sampled) with no explicit penalty. The EMA alignment acts as a temporal regularizer, promoting embedding stability.

Weight: 0.3 (configurable via `lambda_alignment`).

#### How the loss terms complement each other

- **Contrastive (h_player)**: Within-batch alignment (same player → close) and within-role separation (different players in same position group → apart). Operates on pre-pooling embeddings where positive pairs exist.
- **Pooled uniformity (z_p)**: Spreads pooled embeddings uniformly on the hypersphere, directly improving the cosine similarity space used for downstream search. Re-establishes cross-role separation (e.g., GK far from outfield).
- **Action loss via FiLM**: Forces `z_p` to capture behavioral differences between players, because predictions are multiplicatively dependent on it.
- **Outcome loss**: Provides a secondary signal to shape event representations toward possession-level outcomes.
- **EMA alignment (z_p, optional)**: Promotes cross-batch self-consistency in pooled embeddings. Without it, `z_p` can drift when different possession subsets are sampled. Complements uniformity (which spreads embeddings apart) by anchoring each player's position on the hypersphere across training iterations.

### Training Details

- **Optimiser**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5) — monitors `val_supervised` (action + λ_outcome × outcome), which is immune to schedule-induced changes in contrastive/uniformity weights
- **Gradient clipping**: max_norm=1.0
- **Batch size**: 96 possession graphs (default random shuffle)
- **Data split**: 70/15/15 by match_id (prevents leakage — all possessions from a given match go to the same split, so the model cannot memorise match-specific patterns and leak them across train/val/test)
- **Early stopping**: patience=15, min_delta=1e-4, on `val_supervised`

#### Dual Checkpoint Strategy

The trainer maintains two "best" checkpoints targeting different objectives:

| Checkpoint | Tracks | Purpose |
|---|---|---|
| `best_model.pt` | Lowest `val_supervised` (action + λ_outcome × outcome) | Primary checkpoint for inference — optimises supervised accuracy |
| `best_total_model.pt` | Lowest `val_total` (full multi-objective loss including contrastive + uniformity) | Useful for embedding-space diagnostics where auxiliary losses matter |

Additionally: periodic checkpoints every N epochs (`checkpoint_epoch_{n}.pt`), a `final_model.pt` at the end of training, and a `training_history.json` with per-epoch loss breakdowns (train/val for each loss component + learning rate).

#### Resume from Checkpoint

Training can resume from any checkpoint via `--resume <filename>`, which restores model weights, optimizer state, scheduler state, best-loss tracking, and patience counter. History is also restored so the full training curve is preserved.

#### Player-Aware Batch Sampling (optional)

When `--player_sampling` is set, the standard random-shuffle DataLoader is replaced with `PlayerAwareBatchSampler`, which constructs each batch as K players × M possessions:

- **K** (`players_per_batch`, default 16): distinct players per batch
- **M** (`possessions_per_player`, default 6): possessions sampled per player

This guarantees that every batch contains multiple possessions per player, ensuring the InfoNCE contrastive loss always has positive pairs. Without this, random shuffling produces batches where most players appear only once, starving the contrastive loss of signal. When the in-epoch player pool runs low, it is **extended** with a fresh shuffle so leftover players are not dropped.

When `--player_sampling` is active, the CLI also applies a tuned hyperparameter preset: temperature 0.15, λ_contrast 0.15, λ_pooled 0.5, with cosine annealing enabled for both temperature and pooled uniformity weight.

#### Schedule Annealing (optional)

When enabled (via `--player_sampling` or manual config), the trainer applies cosine annealing per epoch:

- **Temperature**: anneals from `temperature_start` (0.15) to `temperature_end` (0.02) — sharpens the contrastive loss over training
- **Pooled uniformity weight**: anneals from `pooled_weight_start` (0.10) to `pooled_weight_end` (0.50) — gradually increases the uniformity pressure on pooled embeddings

The `val_supervised` metric used for early stopping and LR scheduling is deliberately immune to these annealing changes, preventing schedule-induced artifacts in convergence decisions.

#### Test-Set Evaluation

After training, the best checkpoint can be evaluated on the held-out test set (`--mode evaluate`, same split, seed=42). Metrics: action type / angle bin / length bin accuracy and macro F1, plus outcome accuracy, BCE, and AUC-ROC for ends_in_shot and ends_in_goal. Default output directory: **`evaluations/{tag}/test_metrics/`** (`test_metrics.json`, confusion matrices, ROC plots). The same folder is used when `evaluate` runs as step 2 of `full_eval`.

---

## Model Variation Components & Pipeline Registry

### Overview

The system supports eight modular architectural and training components that can be independently toggled to form distinct pipeline variants. Each component addresses a specific hypothesis about what makes a good player-similarity embedding. The `PIPELINE_REGISTRY` in `main.py` defines 11 named combinations of these components (plus one archived run), and `--mode full_eval_all` evaluates all of them under a unified evaluation suite.

The components are presented in roughly the order they were developed, each motivated by a limitation observed in previous variants.

### Component 1: Split-Context Edges

**Config**: `GraphConfig.split_context_edges` (default `False`). CLI: `--split_context_edges`.

**What it does**: Splits the single `(player, context_for, event)` edge type into two relation types:
- `(player, context_for_tm, event)` — teammate 360-frame context
- `(player, context_for_opp, event)` — opponent 360-frame context

Each relation gets its own learned projection matrix and attention weights inside `HeteroConv`.

**Why it exists**: A teammate 3 metres ahead is a passing option; an opponent 3 metres ahead is a pressing threat. With a single edge type, the GNN must learn to distinguish these from a single scalar feature (`is_possession_team`). Splitting gives the attention mechanism structurally different pathways for offensive support and defensive pressure.

**Effect on results**: `split_ctx` alone (without position ablation) produces the best pseudo-GT mean rank among all models (56.1) and the best hit@50 when combined with EMA alignment (0.700 for `pos_ablated_split_ctx_ema`). However, split-context edges alone do not improve behavioral fidelity — `split_ctx` has the worst Spearman rho (0.710) and worst absolute top-k JS (0.004611) among non-collapsed models, indicating its embeddings retrieve positional peers rather than behavioral matches. The component's value is unlocked when combined with position ablation (Component 2), which forces the model to use the teammate/opponent distinction for behavioral reasoning rather than positional shortcuts.

### Component 2: Position Ablation

**Config**: `ModelConfig.ablate_position` (default `False`). CLI: `--ablate_position`.

**What it does**: Removes all explicit position information from the learned representation:
- **PlayerProjection**: The 16-D position embedding is skipped; player nodes are projected from only 3 continuous features (team flag, dx, dy).
- **FiLM conditioning**: The dedicated `pos_emb` channel is zeroed; scale/shift parameters are computed from `z_p` alone (64-D input instead of 80-D).

Position indices remain in graph metadata (`player.x[:, 0]`) for hard-negative mining in the contrastive loss.

**Why it exists**: When position is available, the model can satisfy the action-prediction loss with a simple rule — "Left Wings cross, Center Backs clear" — without differentiating individual players. FiLM gamma/beta converge toward values that mostly encode positional role, leaving `z_p` with little capacity for individual style. Removing position forces `z_p` to learn the behavioral repertoire of each player from raw event patterns and spatial context, making the embedding space capture style rather than role assignment.

**Effect on results**: `pos_ablated` (position ablation alone, without split-context) achieves the highest Spearman rho (0.974) across all models, meaning embedding proximity correlates strongly with behavioral similarity. However, this high rho is an artifact of embedding compression — all players look similar (mean cosine distance 0.085) — which makes retrieval noisy (GT mean rank 69.6, competition-split mean rank 81.3). Position ablation requires strong uniformity pressure (Component 4) to spread embeddings apart and produce useful retrieval.

### Component 3: Player-Aware Batch Sampling

**Config**: `TrainingConfig.player_sampling` (default `False`). CLI: `--player_sampling`.

**What it does**: Replaces the default random-shuffle DataLoader with `PlayerAwareBatchSampler`, which constructs each batch as K players × M possessions (default K=16, M=6, effective batch size 96). This guarantees every batch contains multiple possessions per player, ensuring the InfoNCE contrastive loss always has positive pairs.

When active, the system also applies a tuned hyperparameter preset:
- Contrastive temperature: 0.15 (warmer than default 0.05)
- λ_contrast: 0.15 (reduced)
- λ_pooled: 0.5 (increased)
- Cosine annealing: temperature 0.15→0.02, pooled weight 0.10→0.50

**Why it exists**: With random shuffling at batch size 96, most players appear only once per batch, starving the contrastive loss of positive pairs. Player-aware sampling guarantees 6 possessions per player per batch, producing (6 choose 2)=15 positive pairs per player for InfoNCE. The hypothesis was that richer contrastive signal would produce tighter per-player clusters and better retrieval.

**Effect on results**: Player-sampling models (`player_samp`, `split_ctx_ps`) achieve the best random-half self-consistency (mean rank 2.7–2.9, hit@10 0.94–0.95) and highest supervised F1 (0.700 for `player_samp`). However, they fail at the substitute-finding task: tier-1 hit@10 = 0.0 (cannot place any of the five strongest pairs in the top 10), competition-split mean rank > 89, and Spearman rho < 0.79. The training signal "same player = close" does not teach the model what makes two *different* players functionally similar. The strong within-context consistency is a direct consequence of the sampling mechanism, not emergent generalization — the model excels at player re-identification but underperforms at cross-player similarity retrieval. This is the "player-sampling paradox" discussed in `EVALUATION_RESULTS.md` §9.4.

### Component 4: Uniformity Sensitivity (t) and Weight (λ_pooled)

**Config**: `TrainingConfig.uniformity_t` (default 2.0), `TrainingConfig.lambda_pooled_contrast` (default 0.3). CLI: `--uniformity_t`, `--lambda_pooled`.

**What it does**: Controls the Gaussian-potential uniformity loss applied to pooled `z_p` vectors:

```
L_uniform = log E[ exp(-t · ||z_i - z_j||²) ]
```

Parameter `t` controls sensitivity — higher values concentrate the gradient on the closest pairs, penalising embedding compression more aggressively. Parameter `λ_pooled` controls the loss weight in the total objective.

**Why it exists**: Without uniformity pressure, attention-pooled `z_p` vectors collapse into a narrow cone where all players have cosine similarity > 0.99. This happens because averaging over many possessions pulls every player's embedding toward the population mean. The uniformity loss pushes all `z_p` apart on the unit hypersphere, widening cosine-similarity gaps so that "similar" and "dissimilar" players occupy meaningfully different regions.

Position ablation (Component 2) creates an especially strong dependency on uniformity: without positional shortcuts, the action loss alone provides weaker gradient signal for separating `z_p` vectors. The `pos_ablated_split_ctx` family uses elevated settings (t=4.0, λ_pooled=1.0) — roughly 3× the default weight — to counteract this.

**Effect on results**: The `pos_ablated_split_ctx` family (t=4.0, λ_pooled=1.0) achieves the best competition-split mean ranks (67.3–67.7) and strong behavioral fidelity (rho 0.929–0.944). Critically, `pos_ablated_split_ctx_v2` reduced λ_pooled from 1.0 to 0.7 (a 30% reduction) while keeping t=4.0, and the result was catastrophic: GT mean rank 198.8, competition-split mean rank 200.8, random-half hit@10 0.094. This confirms that position ablation creates a hard dependency on uniformity pressure — even a modest reduction causes representational collapse. The threshold effect suggests that λ_pooled=1.0 is near the minimum viable weight for the position-ablated architecture.

### Component 5: EMA Cross-Batch Alignment

**Config**: `TrainingConfig.ema_alignment` (default `False`), `TrainingConfig.lambda_alignment` (default 0.3), `TrainingConfig.ema_momentum` (default 0.999). CLI: `--ema_alignment`, `--lambda_alignment`, `--ema_momentum`.

**What it does**: Adds a cosine-alignment loss that encourages each player's pooled `z_p` to remain consistent across training batches:

```
L_alignment = 1 - cosine_similarity(z_p_current, z_p_ema)
```

An EMA memory bank (`EMAPlayerMemoryBank`) maintains a momentum-updated historical embedding for each player (decay 0.999). Before the loss computation, the bank is queried for each player in the current batch; players with stored history contribute to the alignment loss. After the optimizer step, the bank is updated with current `z_p` values (detached from the computation graph). Bank state is saved in checkpoints for resume compatibility.

**Why it exists**: Standard training with random batching means a player's `z_p` is computed from a different subset of possessions each epoch. Without an explicit consistency signal, `z_p` can drift — the model might assign subtly different embeddings depending on which 6 (out of 200) possessions happen to be in the batch. EMA alignment acts as a temporal regularizer that anchors each player's position on the hypersphere. It complements the uniformity loss (which pushes embeddings apart globally) by adding a per-player stability constraint.

**Effect on results**: Two variants were tested:
- `pos_ablated_split_ctx_ema` (λ_alignment=0.3): Best pseudo-GT hit@50 (0.700), best GT mean rank (62.6), and highest rho among fully-evaluated models (0.944). However, it trades supervised F1 (0.650, the lowest among non-collapsed split-ctx runs) and random-half consistency (mean rank 9.3, worse than its non-EMA counterpart at 6.3). The alignment loss regularizes the embedding space at the cost of slightly loosening within-context tightness.
- `pos_ablated_split_ctx_ema_v2` (λ_alignment=0.1): Partially recovers F1 to 0.674 and random-half to 6.1, while retaining strong competition-split performance (67.5). However, this variant is archived (not in `PIPELINE_REGISTRY`) since the acts-in dropout family superseded it.

EMA alignment provides the most value for pseudo-GT pair retrieval (the metric it most directly optimizes for: stable, pair-consistent embeddings). Its weakness is reduced supervised prediction quality, because the alignment loss trades some task-specific gradient signal for embedding stability.

### Component 6: Acts-In Edge Dropout

**Config**: `TrainingConfig.acts_in_dropout` (default 0.0). CLI: `--acts_in_dropout`.

**What it does**: With probability `p` (default 0.3 when enabled), the `(player, acts_in, event)` edges are entirely dropped from the GNN message-passing graph during a training forward pass. The reverse edge `(event, performed_by, player)` and all context edges are retained. This is a stochastic regularizer applied per-batch during training only; inference always uses the full graph.

Implementation in `PlayerSimilarityModel.forward()`:
```python
drop_acts_in = (
    self.training
    and self.acts_in_dropout > 0
    and random.random() < self.acts_in_dropout
)
edge_index_dict = {
    et: data[et].edge_index
    for et in data.edge_types
    if not (drop_acts_in and et == ("player", "acts_in", "event"))
}
```

**Why it exists**: The `acts_in` edges carry actor identity into the event representation — they tell the GNN *who* performed each action. When these edges are always present, `h_event` can encode player-specific information, reducing FiLM's need to use `z_p` for personalization. By stochastically removing actor identity from the graph, the model must learn to predict actions in two regimes: (1) with full actor context (70% of batches), where actor-specific h_event patterns help, and (2) without actor identity (30% of batches), where FiLM and `z_p` are the *only* pathway for player-dependent prediction. This forces `z_p` to carry more behavioral information and makes the embedding space more discriminative for substitute retrieval.

The mechanism is related to dropout in spirit but operates on the graph topology rather than on node features or weights. It is a form of structural noise injection that prevents the model from over-relying on actor-event message passing.

**Effect on results**: `acts_in_dropout` (p=0.3, with split-context + position ablation + t=4.0/λ=1.0) achieves the best competition-split hit@10 (0.351), second-best absolute top-k JS (0.001031), and strong behavioral ordering (rho 0.928). It sits between the base `pos_ablated_split_ctx` and the EMA variant on most metrics, with the advantage of simpler training (no memory bank, no extra hyperparameters beyond the dropout probability). The stochastic masking slightly loosens within-context tightness (random-half mean rank 8.7 vs 6.3 for non-EMA base) but improves cross-context robustness and external validation (FIFA main-6 diff 7.9, second-best among GNN models).

### Component 7: Position-Group Prediction Loss (λ_pos)

**Config**: `TrainingConfig.lambda_pos` (default 0.0). CLI: `--lambda_pos`.

**What it does**: Adds a cross-entropy loss that predicts each player's coarse position group (Goalkeeper / Defender / Midfielder / Forward / Unknown) from their pooled `z_p`:

```
L_pos = CrossEntropy(pos_head(z_p), position_group_label)
```

A small linear head (`nn.Linear(d, NUM_POSITION_GROUPS)`) on top of pooled `z_p` provides the logits. The loss weight `λ_pos` controls how much this auxiliary objective contributes to the total loss.

**Why it exists**: Position ablation (Component 2) removes position from the input features, which is necessary for behavioral (not positional) embeddings. However, a practical substitute must typically play the same broad position. Without any positional signal, the model may produce embeddings where a goalkeeper and a forward are close because they both rarely touch the ball in midfield. The position-prediction loss provides a mild regularizer that preserves coarse positional structure in `z_p` without reintroducing the explicit positional shortcuts that ablation removed.

The key distinction from the original position embedding is that position enters as a *soft prediction target* (the model must learn to infer position from behavior) rather than a *hard input feature* (the model is told the position directly). This encourages position-aware representations without permitting the "Left Wings cross" shortcut.

**Effect on results**: `acts_in_dropout_pos` (λ_pos=0.3) was tested but has no `evaluations/` folder yet, so direct metrics are unavailable. The component's contribution is observed through `acts_in_dropout_pos_gu`, which combines it with group-weighted uniformity (Component 8).

### Component 8: Group-Weighted Uniformity

**Config**: `TrainingConfig.uniformity_group_weight` (default 1.0). CLI: `--uniformity_group_weight`.

**What it does**: Modifies the Gaussian-potential uniformity loss so that same-position-group player pairs receive extra repulsive weight:

```python
if position_groups is not None and self.group_weight != 1.0:
    same_group = pos_groups.unsqueeze(0) == pos_groups.unsqueeze(1)
    weights = torch.where(same_group, self.group_weight, 1.0)
    loss = log((exp_vals * weights).sum() / weights.sum())
```

With `uniformity_group_weight=3.0`, same-group pairs (e.g., midfielder vs midfielder) receive 3× the repulsive force of cross-group pairs (e.g., midfielder vs goalkeeper). Cross-group separation is already strong because different position groups exhibit very different action distributions; within-group separation requires more gradient pressure because players in the same role share many behavioral patterns.

**Why it exists**: Standard uniformity loss distributes its finite gradient budget uniformly across all player pairs. But cross-group pairs (GK vs. FWD) are already well-separated by behavioral differences, so they consume gradient signal that would be more useful for separating similar players within the same role. Group-weighted uniformity reallocates the repulsive budget toward within-group pairs, sharpening the embedding space precisely where scouting queries operate — "find me another right-back like Trent Alexander-Arnold" requires distinguishing among right-backs, not between right-backs and goalkeepers.

**Effect on results**: `acts_in_dropout_pos_gu` (group_weight=3.0, with acts-in dropout + λ_pos + split-context + position ablation) is the top-ranked model in the weighted evaluation (score 0.879). Its within-position discrimination produces neighbors that are not only behaviorally similar but also externally validated:
- **Best FIFA main-6 diff** (7.7) among all GNN models — retrieved neighbors have the closest FIFA skill profiles
- **Best qualitative head-coach score** (3.5/5) — produces Rúben Dias for Van Dijk and Bruno Fernandes for De Bruyne, picks no other model surfaces
- **Strong behavioral fidelity** (rho 0.922, top-k JS 0.001051) without compression artifacts

The trade-off is a slightly higher competition-split mean rank (74.0, vs 67.3 for `pos_ablated_split_ctx`) — the group-uniformity term trades some cross-context stability for sharper within-position boundaries. This is the expected cost: concentrating repulsive force within groups loosens the constraints between groups, allowing some cross-context drift for players whose competition context differs substantially.

### Pipeline Registry

All 11 named pipeline variants in `PIPELINE_REGISTRY`, listed in registry order. Each row shows which components are active and the key hyperparameter overrides from the defaults.

| # | Pipeline Tag | Split Ctx | Pos Ablated | Player Samp | t | λ_pooled | EMA | λ_align | Acts-In Drop | λ_pos | Group Wt |
|---|---|:-:|:-:|:-:|--:|--:|:-:|--:|--:|--:|--:|
| 1 | `baseline` | — | — | — | 2.0 | 0.3 | — | — | — | — | 1.0 |
| 2 | `player_samp` | — | — | Yes | 2.0 | 0.5 | — | — | — | — | 1.0 |
| 3 | `split_ctx` | Yes | — | — | 2.0 | 0.3 | — | — | — | — | 1.0 |
| 4 | `split_ctx_ps` | Yes | — | Yes | 2.0 | 0.5 | — | — | — | — | 1.0 |
| 5 | `pos_ablated` | — | Yes | — | 2.0 | 0.3 | — | — | — | — | 1.0 |
| 6 | `pos_ablated_split_ctx` | Yes | Yes | — | 4.0 | 1.0 | — | — | — | — | 1.0 |
| 7 | `pos_ablated_split_ctx_v2` | Yes | Yes | — | 4.0 | 0.7 | — | — | — | — | 1.0 |
| 8 | `pos_ablated_split_ctx_ema` | Yes | Yes | — | 4.0 | 1.0 | Yes | 0.3 | — | — | 1.0 |
| 9 | `acts_in_dropout` | Yes | Yes | — | 4.0 | 1.0 | — | — | 0.3 | — | 1.0 |
| 10 | `acts_in_dropout_pos` | Yes | Yes | — | 4.0 | 1.0 | — | — | 0.3 | 0.3 | 1.0 |
| 11 | `acts_in_dropout_pos_gu` | Yes | Yes | — | 4.0 | 1.0 | — | — | 0.3 | 0.3 | 3.0 |

Additionally, `pos_ablated_split_ctx_ema_v2` (λ_alignment=0.1) exists under `evaluations/` but is **not** in `PIPELINE_REGISTRY` — it is an archived run kept for comparison.

Pipeline 10 (`acts_in_dropout_pos`) is registered but has **no evaluation run** yet.

### Pipeline Progression: Design Rationale

The 11 pipelines represent a progressive exploration. Each stage was motivated by a specific limitation of the previous best model:

**Stage 1 — Baselines (pipelines 1–4):**

`baseline` establishes the minimal system: heterogeneous GNN with unified context edges, position in input features, random batching, default uniformity (t=2.0, λ=0.3). `player_samp` tests whether guaranteeing contrastive positive pairs via structured batching improves embeddings. `split_ctx` tests whether separating teammate/opponent context helps. `split_ctx_ps` combines both.

**Observation**: `split_ctx` achieves the best pseudo-GT retrieval (mean rank 56.1) but the worst behavioral fidelity (rho 0.710). Player-sampling models excel at self-consistency but fail at cross-player similarity. The baseline is surprisingly competitive with more complex models. The core issue: position information dominates `z_p`, producing positional-peer retrieval rather than behavioral-match retrieval.

**Stage 2 — Position ablation (pipelines 5–8):**

`pos_ablated` removes position from input features, forcing behavioral representations. `pos_ablated_split_ctx` combines this with split-context and elevated uniformity (t=4.0, λ=1.0) to prevent collapse. `pos_ablated_split_ctx_v2` tests whether λ=0.7 suffices — it does not (catastrophic collapse). `pos_ablated_split_ctx_ema` adds EMA alignment for cross-batch stability.

**Observation**: The `pos_ablated_split_ctx` family dominates competition-split self-consistency (mean rank 67.3–67.7) and behavioral fidelity (rho 0.929–0.944). The collapsed v2 variant confirms the hard dependency on uniformity weight. EMA alignment further improves pseudo-GT retrieval (hit@50 0.700) but at the cost of supervised F1 (0.650). The remaining gap: within-position discrimination is moderate — the model treats all center-backs somewhat similarly.

**Stage 3 — Acts-in dropout + position/group regularization (pipelines 9–11):**

`acts_in_dropout` introduces stochastic actor-identity removal to force `z_p` to carry more behavioral signal. `acts_in_dropout_pos` adds a soft position-prediction loss to maintain coarse positional structure. `acts_in_dropout_pos_gu` adds group-weighted uniformity to sharpen within-position discrimination.

**Observation**: `acts_in_dropout_pos_gu` is the top-ranked model (weighted score 0.879), winning the external validation categories (FIFA, qualitative) while maintaining strong behavioral metrics. The group-uniformity term produces discriminative within-position embeddings that yield practically useful substitutes — not just behaviorally similar players, but players a head coach would actually consider (Rúben Dias for VVD, Bruno Fernandes for KDB).

### Component Interaction Summary

| Component pair | Interaction |
|---|---|
| Split-context + Position ablation | Synergistic. Split-context provides the structural distinction (tm/opp) that the model needs when position labels are removed. Without split-context, position ablation must infer teammate/opponent from a single scalar. |
| Position ablation + High uniformity | Required. Position ablation weakens the per-player gradient signal (no role shortcuts), so stronger uniformity (t=4.0, λ=1.0) is needed to maintain embedding spread. Reducing to λ=0.7 causes collapse. |
| Acts-in dropout + FiLM | Complementary. Stochastically removing actor identity forces FiLM (and therefore `z_p`) to carry the full behavioral prediction burden. Without FiLM's multiplicative dependence, dropout of acts-in edges would simply degrade predictions. |
| Group-uniformity + Position ablation | Complementary. Position ablation removes role shortcuts; group-weighted uniformity re-introduces *soft* position awareness by focusing repulsive force within position groups. The net effect is embeddings that are position-aware (center-backs cluster away from forwards) but differentiated within position (this center-back plays differently from that one). |
| Player sampling + Uniformity | Partially redundant. Player sampling increases within-player cohesion but the resulting tight clusters are already well-separated by the uniformity loss. The combination produces the highest random-k JS in the substitute-quality diagnostic (`player_samp` / `split_ctx_ps` ≈ 0.01337–0.01344 vs ~0.008–0.009 for strong split-ctx models), inflating the substitute *ratio* via the denominator rather than improving absolute top-k neighbor quality. |
| EMA alignment + Acts-in dropout | Alternative approaches. Both improve embedding stability, but EMA alignment does so via explicit regularization (memory bank) while acts-in dropout does so via implicit regularization (noise injection). The acts-in dropout family achieves comparable or better results with simpler training infrastructure. |

### Heuristic Baselines

Three non-learned baselines are evaluated under the same framework for calibration:

| Tag | Method | Description |
|---|---|---|
| `h_mean_features` | Mean Features | Per-player average of all 126-D raw StatsBomb event features. Tests whether simple feature statistics suffice. |
| `h_action_profile` | Action Profile | Per-player histogram of the 14 action types, normalized to a probability distribution. Tests whether action-type frequency alone captures player style. |
| `h_fifa_attributes` | FIFA Attributes | Raw FIFA video-game attribute vectors (pace, shooting, passing, dribbling, defending, physic + sub-attributes). Tests an external human-curated skill profile. |

Heuristic embeddings use the full player corpus (1,965 players for mean-features/action-profile; 1,450 for FIFA attributes, limited by FIFA data coverage) rather than the GNN's 1,633 (50-possession minimum). No self-consistency or policy diagnostics are computed for heuristics since they have no learned model to probe.

`h_mean_features` is surprisingly competitive on pseudo-GT (mean rank 108.7, hit@50 0.633) — simple feature averages capture some behavioral patterns. `h_action_profile` performs worst (mean rank 287.2), confirming that action-type frequency alone is insufficient. `h_fifa_attributes` provides a calibration ceiling for external validation (FIFA main-6 diff 5.0, position match 75%) since it uses the FIFA data directly.

---

## Phase 6: Inference, Evaluation & Similarity Search

### 6A: Embedding Generation

For each possession graph in the dataset:

1. Apply future-info masking (same as training)
2. Run the GNN encoder to get `h_player` for every player node
3. Collect each player's `h_player` vectors across all their possessions
4. Attention-pool into a single global `z_p` per player (minimum 50 possessions required)

Processing uses mini-batch inference with PyG's `Batch.from_data_list()` for efficiency.

The `--inference_split` flag controls which graphs are used: `all` (default) uses the full corpus, while `train`/`val`/`test` uses the corresponding match-level split (same seed=42 as training). This allows generating embeddings from specific subsets for controlled evaluation.

**Output**: `player_embeddings.npy` (n_players × 64), `player_info.parquet`, `embedding_manifest.json`.

#### Provenance System

Every embedding generation writes an `embedding_manifest.json` alongside the embedding files, recording:

- Checkpoint path and modification timestamp
- Graphs filename and `split_context_edges` flag
- Experiment tag and inference split

Downstream modes (`search`, `ground_truth`, `policy_diagnostic`) validate the manifest before running, catching stale embeddings (checkpoint retrained but embeddings not regenerated), graph/checkpoint architecture mismatches, and cross-tag contamination. Staleness is detected by comparing checkpoint and embedding file modification timestamps.

### 6B: Similarity Search

Given a query player_id:

1. Validate embedding provenance against the current checkpoint
2. Compute pairwise cosine similarity between all `z_p` vectors
3. Rank by similarity, with optional filters (position group, minimum possessions, exclude same team)
4. Return top-k matches

**Gender filtering.** All similarity-based modes (search, ground_truth, self_consistency, policy_diagnostic) automatically filter candidates by gender. Gender is derived from competition metadata (`competitions.json` → `competition_gender`). Male query players only see male candidates; female queries only see female candidates. The `build_gender_map()` utility in `similarity_search.py` constructs this mapping dynamically.

### 6C: Pseudo Ground-Truth Evaluation

Evaluates the embedding space against **15 LLM-compiled** player-similarity pairs (5 tier-1, 7 tier-2, 3 tier-3) with citations to public analytics/media sources (see `docs/pseudo_ground_truth.md` and `GROUND_TRUTH_PAIRS` in `src/phase6_inference/ground_truth.py`). These are **directional sanity checks, not expert-validated ground truth**. Each pair has a tier reflecting how strong the directional expectation is:

| Tier | Description | Example (from `ground_truth.py`) |
|---|---|---|
| 1 | Strong directional expectation | Modrić ↔ Kroos; Van Dijk ↔ Dias; Bonmatí ↔ Putellas |
| 2 | Good directional expectation | Kroos ↔ Enzo Fernández (StatsBomb raw top-5); Kane ↔ Lewandowski |
| 3 | Weaker / conditional expectation | Musiala ↔ Foden (noted stylistic differences); Neuer ↔ Donnarumma |

For each pair (A, B):

1. Compute cosine similarity between `z_A` and `z_B`
2. Find B's rank in A's nearest-neighbour list (and vice versa)

Aggregate metrics: mean/median rank, hit@5/10/20/50, per-tier breakdowns, and bootstrap 95% confidence intervals on mean rank and hit@10.

**Output**: `ground_truth_report.txt`, `ground_truth_results.json` (with run provenance appended).

### 6D: Self-Consistency Evaluation

Tests whether the model assigns stable player identity by checking if the same player, observed in different data subsets, retrieves themselves as the closest match. Two complementary tests:

**Competition-split test**: For players appearing in ≥2 competitions with sufficient possessions in each, pool `h_player` embeddings per competition independently, and check self-retrieval rank. Non-testable players whose total possessions meet the threshold are added to the retrieval gallery as distractors, so that ranks reflect the full inference-time search population.

**Random-half test**: For all players with ≥100 possessions (regardless of competition count), randomly split possessions 50/50, pool each half, and check self-retrieval rank. This tests embedding stability without the confound of competition context shift.

Both tests report: self-cosine (mean/median/std), cross-cosine, cosine margin, self-retrieval ranks (mean/median, hit@1/5/10/20/50), and per-position-group breakdowns.

**Output**: `self_consistency_report.txt`, `self_consistency_results.json`.

### 6E: Policy Diagnostic

Three diagnostics testing whether cosine similarity corresponds to actual behavioral similarity:

**1. Policy Distance Correlation**: Sample canonical game situations (stratified by actor position group), compute predicted action distributions for all players via FiLM conditioning, measure pairwise Jensen-Shannon divergence, and correlate with pairwise cosine distance using within-group Spearman ρ. Statistical significance is assessed via Mantel-style row-permutation tests (not parametric, because distance-matrix pairs share players and violate i.i.d. assumptions), with Holm-Bonferroni correction across group-level tests.

**2. Substitute Quality**: For a set of query players (mixed: ~half top-possession, ~half random per position group), compare the behavioral similarity (mean JS divergence of predicted action distributions) of their top-K cosine neighbours versus K random same-position-group players. A ratio > 1 means cosine retrieval finds better behavioral matches than chance. Reports bootstrap 95% CI on the aggregate ratio.

**3. FiLM Sensitivity**: For each player, replace their `z_p` with a random same-group player's `z_p` (guaranteed derangement — no identity permutations) and measure JS divergence against the correct predictions. High JS means FiLM is load-bearing and `z_p` genuinely modulates predictions.

**Output**: `policy_diagnostic_report.txt`, `policy_diagnostic_results.json`.

### 6F: FIFA Stat Comparison (`--mode fifa_comparison`)

Validates the GNN similarity system against external FIFA/EA Sports FC player attributes. Requires pre-matched FIFA CSVs in `FIFA_data/` (generated by `match_fifa_players.py`).

1. Sample 10 male + 10 female players (stratified by position group, random each run), plus famous players when available.
2. For each sampled player, find the top-1 same-gender substitute via GNN cosine similarity (must also have FIFA data).
3. Compare main stats and detailed sub-attributes (radar charts are GK-aware).
4. Correlation / agreement summaries (e.g. similarity vs mean stat distance).
5. Multiple PNGs: radar grid, similarity vs stat distance, breakdowns, summary-style figures (e.g. dumbbell / heatmap), category views, sub-attribute detail heatmaps.

**Output** (e.g. under `evaluations/{tag}/fifa_comparison/`): text report plus the PNGs above (exact filenames may evolve; see `test_fifa_comparison.py`).

### Unified Evaluation Pipeline (`--mode full_eval`, `--mode full_eval_all`)

A `PIPELINE_REGISTRY` in `main.py` defines all named GNN pipeline variants. The current registry contains **11 pipelines**: `baseline`, `player_samp`, `split_ctx`, `split_ctx_ps`, `pos_ablated`, `pos_ablated_split_ctx`, `pos_ablated_split_ctx_v2`, `pos_ablated_split_ctx_ema`, `acts_in_dropout`, `acts_in_dropout_pos`, and `acts_in_dropout_pos_gu`. Each entry specifies all architectural and training flags (split-context edges, player sampling, position ablation, uniformity parameters, EMA alignment, acts-in dropout, optional position/group-uniformity terms). The `full_eval` mode runs the complete evaluation suite for the current pipeline (**11 steps**):

1. Inference (generate embeddings)
2. Test-set evaluation (accuracy, F1, confusion matrices)
3. Ground-truth pair evaluation
4. Position-group retrieval precision (`embedding_eval.py`)
5. Split-half embedding stability (skipped for GNN; used for event-feature heuristics in `eval_heuristics`)
6. Qualitative nearest-neighbour table
7. Self-consistency evaluation
8. Policy diagnostic
9. Empirical behavioral fidelity (`empirical_behavioral.py` — observed-action agreement vs random peers)
10. Phase 7 analysis (`analyze`)
11. FIFA stat comparison

`full_eval_all` iterates through all registered pipelines. **`generate_heuristics`**, **`eval_heuristics`**, and **`full_eval_all_with_heuristics`** add or evaluate **mean-features**, **action-profile**, and **FIFA-attributes** heuristics under the same `evaluations/{tag}/…` layout as `full_eval` (`config.tag`, or `baseline` when empty). Configurable root: `--eval_output_dir` (default `./evaluations`).

**Name → ID:** `find_player.py` uses accent folding and optional fuzzy matching (`rapidfuzz`) to obtain `player_id` for `--mode search`.

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
- **PCA and t-SNE plots**: 2D visualisation of the embedding space (global + per-query neighbourhoods); t-SNE is omitted when there are too few players for a stable perplexity

### Possession animation (`--mode possession_animation`)

Optional **MP4/GIF** export of a single possession (or chained segments): ball trail from event locations, freeze-frame player layout, and **counterfactual** predicted actions (two chosen players or dynamic top-1 cosine substitute vs on-ball actor). Uses `src/phase7_analysis/possession_animation.py`; requires `--pipeline` (checkpoint + embeddings), graphs, and model. Not part of `full_eval`. See `main.py` docstring for examples (`--list-possessions`, `--graph-index`, `--players`, `--dynamic-substitute`, `--output`).

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

## Design Discussion: Match Time and Score as Features

### Match time (period one-hot) — implemented

Players behave differently depending on match stage. A center-back at minute 5 plays a safe short pass; at minute 88 when chasing a goal, the same player might launch a long ball forward. Since the system defines "same situation" as "same state → same action," the match period is part of the situational state that shapes what a reasonable action is.

**Implementation: period is encoded as a 4-D one-hot (indices 122–125), appended after Spatial_360.**

- Vocabulary: Period 1, Period 2, Extra Time 1, Extra Time 2 (from StatsBomb `event["period"]`, where 1=first half, 2=second half, 3=ET1, 4=ET2). Period 5 (penalty shootout) events are **excluded** in `data_preparation.py` — shootout penalties are not open play and carry no useful decision-making signal.
- The feature vector is now **126-D** (was 122-D).
- The feature is **not masked** at training — it is pre-action situational context (like location, play pattern, pitch zone).
- 4 dimensions out of 126 is small enough that it will not dominate the feature space; it is comparable in scale to the 2-D location or 2-D distance/angle groups.
- FiLM conditioning means `z_p` must still explain player-level variation given the same period context — the model cannot collapse player traits by attributing all behavioral differences to "what period it is."
- The period field is already present on every StatsBomb event (`event["period"]`), so no new data source is needed.

**Why period one-hot rather than normalised minute:**

- Period boundaries (half-time, extra time) are categorical discontinuities — minute 45 in period 1 and minute 45 in period 2 are very different contexts. A one-hot captures this cleanly.
- Normalised minute within each period could be added as a further refinement, but the coarse period signal is sufficient for a first iteration and avoids over-weighting time information.

**Note:** This is a feature-dimension change — existing checkpoints and processed data are incompatible. Phase 1 must be re-run to produce 126-D features, followed by Phases 2–5.

### Score differential — considered and rejected

Score state affects behavior: a team losing 0-2 at minute 80 presses high and plays direct. However, incorporating score differential was rejected for the following reasons:

1. **Confounds player trait with team strength.** The system compares players across different matches and teams. If player A's team is usually winning and player B's team is usually losing, their embeddings could diverge because of the team's match context — not because the players themselves decide differently. This would cause similarity to correlate with "plays for a strong/weak team" rather than individual style.
2. **Partially redundant with behavior.** A team losing will naturally produce more long balls, more shots, more aggressive pressing — patterns the model already observes through the event sequence and graph structure.
3. **Data complexity.** StatsBomb events do not carry a running score field; computing it requires tracking goals throughout the match, adding a preprocessing step with edge cases (own goals, VAR reversals).

For these reasons, score is not included. If future analysis suggests score context materially improves embedding quality without introducing the team-strength confound, it can be revisited with controls (e.g., conditioning on score differential as a separate input to the outcome head only, not to the action prediction path).

---

## Key Configuration

All configuration is defined in `src/config.py` as nested dataclasses (`DataConfig`, `FeatureConfig`, `PossessionConfig`, `GraphConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`) wrapped in a master `Config`. There is no external YAML/JSON config file — runtime overrides are passed via CLI flags.

### Model & Loss

| Parameter | Value | Description |
|---|---|---|
| `latent_dim` | 64 | Embedding dimensionality |
| `num_layers` | 2 | GNN depth |
| `num_heads` | 4 | Attention heads in layer 1 |
| `hidden_dim` | 128 | Per-head hidden dimension |
| `position_embed_dim` | 16 | Position embedding dimension (player node + FiLM) |
| `n_action_types` | 14 | Action type classes |
| `n_angle_bins` | 9 | Direction bins (8 sectors + no-angle) |
| `n_length_bins` | 5 | Displacement magnitude bins |
| `dropout` | 0.1 | GNN layer dropout |
| `ablate_position` | False | Zero out position in PlayerProjection and FiLM |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |

### Training

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 96 | Training batch size (random shuffle mode) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | Adam weight decay |
| `lambda_outcome` | 0.5 | Outcome loss weight |
| `lambda_contrast` | 0.5 | Contrastive loss weight (h_player InfoNCE) |
| `lambda_pooled_contrast` | 0.3 | Pooled uniformity loss weight (z_p) |
| `contrastive_temperature` | 0.05 | InfoNCE temperature |
| `uniformity_t` | 2.0 | Gaussian-potential uniformity sensitivity |
| `ema_alignment` | False | Enable EMA cross-batch alignment loss |
| `lambda_alignment` | 0.3 | EMA alignment loss weight |
| `ema_momentum` | 0.999 | EMA memory bank decay rate |
| `patience` | 15 | Early stopping patience |
| `save_every_n_epochs` | 10 | Periodic checkpoint interval |

### Player-Aware Sampling (when `--player_sampling`)

| Parameter | Default | Description |
|---|---|---|
| `players_per_batch` | 16 | Distinct players per batch (K) |
| `possessions_per_player` | 6 | Possessions per player per batch (M) |
| `contrastive_temperature` | 0.15 | Override: warmer start for sampling mode |
| `lambda_contrast` | 0.15 | Override: reduced contrastive weight |
| `lambda_pooled_contrast` | 0.5 | Override: increased uniformity weight |
| `temperature_start → end` | 0.15 → 0.02 | Cosine-annealed temperature |
| `pooled_weight_start → end` | 0.10 → 0.50 | Cosine-annealed uniformity weight |

### Inference

| Parameter | Value | Description |
|---|---|---|
| `min_samples_per_player` | 50 | Minimum possessions for embedding |
| `top_k` | 10 | Default number of neighbours returned |
| `similarity_metric` | cosine | Similarity metric for search |

---

## Audit & Fixes Applied

The following issues were identified during a full code audit and corrected:

### 1. Validation loss now includes contrastive component

**Problem**: `_validate()` in `trainer.py` computed only `L_action + λ_outcome · L_outcome`, while `_train_epoch()` computed the full `L_action + λ_outcome · L_outcome + λ_contrast · L_contrastive`. This meant `ReduceLROnPlateau` and early stopping operated on a different objective than training — the scheduler could reduce LR or trigger early stopping even while the contrastive loss was still improving the embedding space.

**Fix**: Added actor-only `h_player` extraction and contrastive loss computation to `_validate()`, mirroring the training loop. Validation loss is now computed with the same formula as training loss.

### 2. Device selection standardised across all phases

**Problem**: Training (`trainer.py`) explicitly preferred CPU over MPS (because PyG heterogeneous-graph workloads with many small kernel launches run slower on Apple MPS), but inference (`main.py`) and analysis (`report_builder.py`) still attempted MPS before falling back to CPU. Since inference runs the same GNN encoder with the same workload characteristics, MPS would be equally slow, and the overall device-selection logic was inconsistent across phases.

**Fix**: Removed MPS preference from `run_inference()` in `main.py` and from `ReportBuilder.run()` in `report_builder.py`, and standardised device selection so that all phases now use **CUDA if available, otherwise CPU**.

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
- `Possession` dataclass: added `timestamps_sec` field (millisecond-precision seconds, parsed from StatsBomb's period-relative `timestamp` string; falls back to `minute * 60 + second` if the column is absent).
- `PossessionGraphBuilder`: computes per-edge time deltas between consecutive events, normalises as `min(Δt / 30, 1.0)`, and stores as `edge_attr` (shape `(n_edges, 1)`) on both `next` and `prev` edge types.
- `PossessionGNNEncoder`: temporal `GATv2Conv` layers now use `edge_dim=1`. The `forward` method accepts an optional `edge_attr_dict` and passes it to `HeteroConv` via the `edge_attr_dict` kwarg pattern.
- `PlayerSimilarityModel`: `forward` and `encode_possession` extract `edge_attr` from `HeteroData` and route it to the GNN.

**Note**: This is an architectural change — existing checkpoints are incompatible and the full pipeline (Phase 2 → Phase 5) must be re-run.

### 10. Decoupled position from z_p via separate FiLM channel

**Problem**: Position is masked from event features (fix #8) so it only enters through the player node → `h_player` → `z_p` → FiLM path. This forces `z_p` to spend most of its 64 dimensions encoding positional role ("I'm a Right Centre Back") with only residual capacity for individual playing style. As a result, nearest-neighbour cosine similarities within the same position were extremely high (0.998–0.9997) and predicted action distributions across neighbours differed by only a few percentage points.

**Fix**: Added a **separate position embedding** to the FiLM conditioning layer. The FiLM scale/shift parameters are now computed from `[z_p ; pos_embed(position_idx)]` (80-D = 64 + 16) instead of `z_p` alone (64-D). The model can learn "CBs clear the ball" from the position embedding, freeing `z_p` to capture "but *this* CB prefers long diagonal switches."

**Changes**:
- `PlayerSimilarityModel.__init__`: added `self.film_pos_embed = nn.Embedding(n_positions, d_pos)`. `film_gamma` and `film_beta` now take `d + d_pos` input.
- `film_condition`: accepts optional `pos_idx` tensor; concatenates `[z_p, pos_embed(pos_idx)]` before computing gamma/beta.
- `forward`: extracts per-event position index from `(player, acts_in, event)` edges and passes it to `film_condition`.
- `situation_comparison.py`: `compare()` now passes each player's position index when calling `film_condition`.

### 11. Hard-negative contrastive loss (same-position-group negatives)

**Problem**: The original InfoNCE loss treated all different-player embeddings in a batch as negatives. With 64 possessions per batch, many negatives came from completely different positions (e.g. goalkeeper vs. forward) — "easy" negatives that the model satisfied by merely encoding positional role. This reduced the loss's ability to push apart players who share the same tactical role.

**Fix**: The contrastive loss now restricts negatives to players **in the same coarse position group** (Goalkeeper / Defender / Midfielder / Forward / Unknown). This forces the model to learn fine-grained individual differences within each role. When an anchor has no valid same-group negatives in a batch, the loss falls back to all-player negatives for that row.

**Changes**:
- `config.py`: added `POSITION_IDX_TO_GROUP` (list mapping each position index to a group 0–4) and `NUM_POSITION_GROUPS`.
- `ContrastiveLoss.forward`: accepts optional `position_groups` tensor; builds a same-group negative mask instead of an all-player mask.
- `CombinedLoss.forward`: forwards the new `contrastive_position_groups` argument.
- `PlayerSimilarityModel.forward`: computes and returns `pooled_z_p_pos_groups` for each unique pooled player.
- `Trainer._train_epoch` / `_validate`: pass pooled `z_p`, player IDs, and position groups from model outputs to the loss.

**Note**: This is an architectural change — existing checkpoints are incompatible and the model must be retrained (Phase 5). Phases 1–3 processed data remain compatible.

### 12. Contrastive on h_player (not pooled z_p) and pooled uniformity loss

**Problem (contrastive):** The trainer was updated to pass **pooled `z_p`** to the contrastive loss. With pooling, each player appears at most once per batch, so there are no positive pairs (same player, different rows). The contrastive loss therefore always returned 0 and had no effect.

**Fix (contrastive):** Reverted to using **pre-pooling `h_player`** (and `player_node_pids`, per-node position groups) for the contrastive loss. The same player can appear in multiple possessions in a batch, providing valid positive pairs. Position groups for hard negatives are derived per-node from `data["player"].x[:, 0]`.

**Addition (pooled uniformity):** To directly widen cosine similarity gaps in the space used for search, a **Gaussian-potential uniformity loss** was added on **pooled `z_p`** (all pairs in the batch pushed apart on the unit hypersphere). Weight `lambda_pooled_contrast = 0.3`. Contrastive weight reduced to `lambda_contrast = 0.5` (configurable). Validation uses the full loss including both terms.

**Dual-channel position:** Position continues to enter `z_p` through the player-node path (PlayerProjection → GNN → h_player → pool). The separate FiLM position embedding (audit #10) is retained. This keeps coarse role structure (e.g. GK separate) while the uniformity loss prevents embedding collapse.

### 13. Split context edges: teammate vs opponent

**Problem:** Phase 3 used a single edge type `(player, context_for, event)` for all 360 spatial context players. A teammate 3m ahead represents a passing option; an opponent 3m ahead represents a block. With one edge type, the GNN must rely only on the player node's `is_possession_team` feature (a single scalar) to distinguish them, forcing the attention mechanism to work harder to learn the fundamentally different roles of teammates vs opponents in the spatial context.

**Fix:** Split into two edge types:
- `(player, context_for_tm, event)` — teammate context (offensive options, support runs)
- `(player, context_for_opp, event)` — opponent context (defensive pressure, blocking threats)

`HeteroConv` now learns separate projection matrices and attention weights for each, letting the model naturally treat offensive and defensive spatial information differently. The routing is determined by the `is_possession_team` flag (already correctly derived in the graph builder, accounting for the XOR between StatsBomb's actor-relative `teammate` field and whether the actor is on the possession team).

**Ablation support:** `GraphConfig.split_context_edges` (default `False` for backward compatibility). Enable via `--split_context_edges` CLI flag for new experiments. Setting to `True` activates the split; `False` retains the original single `context_for` edge type. The GNN encoder's `get_edge_types()` function returns the appropriate relation list based on this flag.

**Changes:**
- `config.py`: Added `split_context_edges: bool = False` to `GraphConfig` (default `False` for backward compat). Added `validate_graph_config_match()` to catch graph/config mismatches at load time.
- `graph_builder.py`: Routes freeze-frame context edges into `ctx_tm_src/dst` or `ctx_opp_src/dst` when `split_context_edges=True`; writes two separate relation types to `HeteroData`.
- `gnn_encoder.py`: `ALL_EDGE_TYPES` replaced by `get_edge_types(split_context)`, computed at init from `GraphConfig`. Each `GATv2Conv` layer allocates separate parameters per context relation.
- `model.py`: `PlayerSimilarityModel.__init__` accepts optional `graph_config` and forwards it to the GNN encoder.
- `main.py` + `report_builder.py`: All model instantiation sites pass `graph_config=config.graph`. All graph load sites call `validate_graph_config_match()`.

**Note**: When `--split_context_edges` is used, this is an architectural change — existing checkpoints are incompatible and the full pipeline (Phase 3 → Phase 5) must be re-run with the flag. Graphs built with a different edge schema will be caught at load time by `validate_graph_config_match()`, which raises `ValueError` on mismatch.

**Ablation workflow** (`--tag` + `--split_context_edges`): Use `python main.py --mode build_graphs --tag split_ctx --split_context_edges` (then `train`, `evaluate`, etc. with the same flags). The tag namespaces **checkpoints and embeddings** (`checkpoints/{tag}/`, `embeddings/{tag}/`); graphs are saved as `possession_graphs_split_ctx.pkl` (topology key), shared across all pipelines that use split-context edges. Phase 1–2 data remain shared. The baseline run (no tag, no split-context flag) is never overwritten.

---

## File Structure

```
GNN_StatsBomb/
├── main.py                          # CLI entry point for all phases
├── find_player.py                   # CLI: name → player_id (accent + fuzzy)
├── match_fifa_players.py            # FIFA-StatsBomb player matcher
├── test_fifa_comparison.py          # FIFA stat comparison & visualizations
├── requirements.txt                 # Python dependencies (torch, torch-geometric, rapidfuzz, etc.)
├── SYSTEM_DESIGN.md                 # This document
├── src/
│   ├── config.py                    # All configuration, vocabularies, tag/graph validation
│   ├── data_preparation.py          # Phase 1A: StatsBomb JSON loading
│   ├── feature_encoder.py           # Phase 1B: 126-D feature encoding
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
│   │   ├── trainer.py               # Training loop with early stopping + dual checkpoints
│   │   ├── losses.py                # Focal, contrastive, uniformity, EMA alignment, combined losses
│   │   ├── dataset.py               # PyG dataset with masking + targets + match-level split
│   │   ├── action_targets.py        # Discretise actions into bins
│   │   ├── evaluator.py             # Test-set evaluation (metrics + plots)
│   │   └── sampler.py               # PlayerAwareBatchSampler for contrastive batching
│   ├── heuristics.py                # Mean-features / action-profile / FIFA-attribute embeddings
│   ├── phase6_inference/
│   │   ├── embedding_generator.py   # Batched z_p generation
│   │   ├── similarity_search.py     # Cosine similarity top-k search
│   │   ├── ground_truth.py          # Pseudo ground-truth pair evaluation
│   │   ├── self_consistency.py      # Competition-split + random-half self-retrieval
│   │   ├── policy_diagnostic.py     # JS correlation, substitute quality, FiLM sensitivity
│   │   ├── embedding_eval.py        # Position retrieval, split-half, qualitative neighbours
│   │   └── provenance.py            # Embedding manifest build/validate
│   └── phase7_analysis/
│       ├── report_builder.py        # Orchestrates full analysis
│       ├── situation_comparison.py   # Counterfactual action prediction
│       └── embedding_viz.py         # PCA + t-SNE visualisations
├── docs/                            # Per-phase documentation
│   ├── README.md                    # Docs index
│   ├── PHASE1.md … PHASE7.md       # One file per phase
│   ├── DATA_QUALITY.md              # Data quality notes
│   ├── EVALUATION_RESULTS.md        # Baseline evaluation results and analysis
│   ├── pseudo_ground_truth.md      # Curated pairs and rationale (no metric results)
│   ├── FUTURE_IMPROVEMENTS.md       # SOTA assessment and improvement roadmap
│   └── PLAYER_SIMILARITY_FINAL_PLAN.md  # Original design plan
├── processed_data/                  # Phase 1–3 outputs
│   ├── event_features.npy           # (N, 126) feature matrix
│   ├── event_metadata.parquet       # Per-event metadata
│   ├── freeze_frames.pkl            # 360 freeze frames for graph construction
│   ├── feature_names.json           # Human-readable feature dimension names
│   ├── data_stats.json              # Dataset statistics
│   ├── possessions.pkl              # Phase 2 output
│   ├── possession_graphs.pkl              # Phase 3: unified context edges
│   └── possession_graphs_split_ctx.pkl    # Phase 3: split teammate/opponent context
├── checkpoints/                     # Trained model weights
│   ├── baseline/                    # Baseline pipeline checkpoints
│   │   ├── best_model.pt            # Best val_supervised checkpoint
│   │   ├── best_total_model.pt      # Best val_total checkpoint
│   │   ├── final_model.pt           # End-of-training checkpoint
│   │   └── training_history.json    # Per-epoch loss curves
│   └── {tag}/                       # Tagged experiment checkpoints (same structure)
├── embeddings/                      # Embedding outputs
│   ├── baseline/                    # Baseline pipeline embeddings
│   │   ├── player_embeddings.npy    # (n_players, 64)
│   │   ├── player_info.parquet      # Player metadata
│   │   ├── embedding_manifest.json  # Provenance (checkpoint, graphs, config)
│   │   └── analysis/                # Phase 7 reports & plots (when analyze run standalone)
│   └── {tag}/                       # Tagged experiment embeddings (same structure)
└── evaluations/                     # Unified evaluation outputs (generated)
    └── {tag}/                       # Per-pipeline results: tag from config, or baseline if unset
        ├── test_metrics/            # Accuracy, F1, confusion matrices
        ├── ground_truth/            # Ground-truth pair evaluation
        ├── position_retrieval/      # Position-group retrieval precision
        ├── split_half/              # Split-half stability (heuristics / skipped for GNN)
        ├── qualitative_neighbors/   # Notable-player neighbour tables
        ├── self_consistency/        # Self-consistency evaluation
        ├── policy_diagnostic/       # Policy diagnostic results
        ├── empirical_behavioral/    # Observed-action JS fidelity (full_eval step 9)
        ├── analysis/                # Phase 7 analysis (when run via full_eval)
        └── fifa_comparison/         # FIFA stat comparison + visualizations
```
