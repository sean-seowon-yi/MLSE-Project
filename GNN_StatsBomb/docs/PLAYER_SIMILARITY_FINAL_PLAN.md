## Player Similarity System – Final Design (StatsBomb 360)

This document is the **final, detailed plan** for building a system that finds players who would **act similarly in the same situation**, using **StatsBomb events + StatsBomb 360** as the primary data source. It is intended to be precise enough to implement.

> **Note**: This plan predates several implementation-time design changes documented in [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md). Key differences: the system now has **7 phases** (Phase 7 = situation-level analysis); position (32–57) is **masked** from event features; the contrastive loss is **player-ID-based with same-position-group hard negatives** (not "state-aware"); FiLM uses **dual-channel position** (`(1 + γ) ⊙ h + β` with `cond = [z_p ; pos_emb]`); and training includes **pooled uniformity loss** on `z_p` to widen cosine similarity gaps. See SYSTEM_DESIGN.md for the current, authoritative design.

The system has six main phases (now seven — see note above):

1. Phase 1 – Event-level encoding (existing).
2. Phase 2 – Possession construction.
3. Phase 3 – Event+Player graph construction (per possession).
4. Phase 4 – GNN encoder and player embeddings.
5. Phase 5 – Training objectives (imitation + value + auxiliary contrastive).
6. Phase 6 – Inference and similarity search.

Throughout, we distinguish:

- **State \(s\)** – the situation: location, context, teammates, opponents, sequence state.
- **Trait \(z_p\)** – the player’s style/decision bias, learned as a global embedding.

The core idea is to train a model that predicts **what action** a player tends to take given a state and their trait, and then use the learned trait embedding `z_p` for player–player similarity.

---

## 1. Goal and definition of “similar player”

### 1.1 Goal

Given a player \(p\) and a large corpus of matches, learn a representation `z_p` such that:

- Players with similar `z` are those who **tend to choose similar actions in similar contexts**:
  - Similar choice of **pass vs carry vs shot vs dribble**.
  - Similar **directionality** (e.g. progressive vs safe passes).
  - Similar **risk profile** (e.g. high xG/line-breaking attempts vs low-risk recycling).
  - Similar behaviour under **pressure** and with specific **off-ball options**.

### 1.2 Working definition

Two players \(A\) and \(B\) are **similar** if:

1. For many comparable states \(s\) (same approximate location, pressure, options),  
2. Their action distributions \(p(y \mid s,z_A)\) and \(p(y \mid s,z_B)\) are close:
   - i.e., they pick the same or similar bins for action type, direction, and length.

We operationalise this by:

- Learning `z_p` via a **masked action prediction** (imitation) objective:
  - \( \hat{y_t} = f_\theta(s_t, z_{p_t}) \).
- Regularising with value-based heads and (optionally) state-aware contrastive loss.
- Using `z_p` for **k‑NN similarity search**.

---

## 2. Data sources and assumptions

### 2.1 StatsBomb events

From `StatsBomb/data`:

- `events/{match_id}.json`:
  - Core fields:
    - `id`, `index`, `period`, `timestamp`, `minute`, `second`.
    - `type.name` (event type), `play_pattern.name`, `team`, `player`, `position`.
    - `location` \([x,y]\) in standard 0–120 × 0–80 pitch.
    - Event-specific fields: `pass`, `shot`, `carry`, `dribble`, etc.
  - We derive:
    - `event_type`, `play_pattern`, `position_name`.
    - `location_x`, `location_y`.
    - `end_location_x`, `end_location_y` (from pass/shot/carry).
    - `duration`, `under_pressure`, `counterpress`.
    - Pass/shot/dribble attributes and outcomes (length, angle, outcome, xG, etc.).
    - `possession_number`, `possession_team_id`, `possession_team_name`.

### 2.2 StatsBomb 360

From `three-sixty/{match_id}.json`:

- For selected events:
  - `event_uuid`: matches `events[i]["id"]`.
  - `freeze_frame`: list of dicts with:
    - `teammate` (bool), `actor` (bool), `keeper` (bool).
    - `location` \([x,y]\) of the player at that instant.

We use 360 to model **off-ball teammates and opponents** around the ball.

### 2.3 Existing Phase 1 outputs (GNN_StatsBomb)

Phase 1 (already implemented) produces:

- `event_features.npy` – `float32` array `(n_events, 126)`:
  - 126‑D fixed-length representation per event (event type, location, play_pattern, position, scalars, pitch zone, **Spatial_360**, period, etc.).
- `event_metadata.parquet`:
  - Rows align with `event_features`.
  - Columns: `event_id`, `match_id`, `competition_id`, `season_id`, `player_id`, `player_name`, `team_id`, `team_name`, `position_name`, `event_type`, `period`, `minute`, `second`, (plus we will ensure `possession_number`, `possession_team_id` are available for Phase 2).
- `feature_names.json` – names of the 126 features.
- `data_stats.json` – summary including `feature_dim`, `use_360`, etc.

Phase 1 is **state-only**: it does not learn trait/style yet; it is a feature engineering stage.

---

## 3. Phase 1 – Event-level encoding (existing)

We retain the existing design with three notable choices:

1. **Use 360 by default**:
   - Only matches with 360 (`three-sixty/{match_id}.json`) are considered.
   - Only events that have a 360 frame are kept.

2. **Left/right roles are not mirrored**:
   - Left vs Right Wing, etc., stay distinct to capture real sidedness and footedness.
   - Optional `feature.mirror_sides` exists but defaults to `False`.

3. **Robust preprocessing**:
   - Clamp coordinates to legal pitch.
   - Clamp durations, pass lengths, xG, etc., to sensible ranges.
   - Use “Unknown” categories as safe defaults when needed.

We refer to `GNN_StatsBomb/docs/PHASE1.md` and `docs/DATA_QUALITY.md` for full details; the key point for later phases is that we can trust:

- **No NaNs / Infs** in `event_features.npy`.
- A consistent 126‑D layout with known indices for:
  - Spatial_360 block.
  - Action/outcome fields (end_location, pass_length, outcomes, etc.).

---

## 4. Phase 2 – Possession construction

### 4.1 Definition of a possession

We adopt StatsBomb’s `possession` definition:

- A **possession** is a continuous spell where one team is in control of the ball, from gaining it (recovery, interception, restart) until losing it (opponent gains it, ball out, foul, etc.).
- In the raw events:
  - `possession`: integer ID within a match.
  - `possession_team`: team with the ball.

We identify a possession by the key:

- `pos_key = (match_id, possession, possession_team_id)`.

**Implementation detail:** `period` is **not** part of this key in `possession_builder.py`. Sorting uses `period` so ordering is still valid if the same `(possession, possession_team_id)` ever appears across a period boundary (rare).

### 4.2 Construction procedure

Input: Phase 1 outputs `event_features.npy` and `event_metadata.parquet` (or an equivalent DataFrame with the same content).

Steps:

1. Ensure metadata includes:
   - `match_id`, `possession_number`, `possession_team_id`, `possession_team_name`.
   - `player_id`, `team_id`, `event_type`, `position_name`, `period`, `minute`, `second`.

2. Group events by `pos_key = (match_id, possession_number, possession_team_id)`:
   - For each group (possession), sort events by:
     - `(period, minute, second, original_index)` as tie-breaker.

3. For each possession:
   - Collect a list of event indices: `[i_1, i_2, …, i_T]` into `event_features`.
   - Maintain a parallel list of metadata for each event (e.g. player_id, event_type).
   - Optionally, record possession-level labels:
     - Whether it ends in a shot or goal.
     - Total xG, territory gained, etc.

Output: A list of possessions, each with:

- `event_indices`: indices into `event_features`.
- `event_meta`: relevant per-event metadata.
- Optional `possession_label` for downstream value heads.

---

## 5. Phase 3 – Event+Player graph construction

For each **possession**, we build a graph that captures:

- **Events** as nodes (on-ball actions, context).
- **Players** as nodes (actors + off-ball teammates and opponents).
- **Temporal** structure via event-to-event edges.
- **Options and pressure** via context edges from off-ball players to events.

### 5.1 Node types and features

For each possession:

#### 5.1.1 Event nodes

- One **event node** per event in the possession.
- Raw feature:
  - 126‑D vector from Phase 1: `x_raw ∈ R^126`.
- Attributes:
  - `player_id`, `team_id`, `event_type`, `period`, `minute`, `second`.
- These features will later be:
  - Masked (for imitation training) to remove action/outcome info.
  - Projected via an MLP to a latent dimension `d`.

#### 5.1.2 Player nodes

- **Actor nodes:** one node per distinct `player_id` who performs at least one retained event in that possession (key `("actor", player_id)` in `graph_builder.py`). Same actor across multiple events reuses that node; `(dx, dy)` stay at the ball.

- **Off-ball (360) context nodes:** for each event, for every `freeze_frame` entry that is not `actor`, the implementation creates **a new node** for that slot at **that event only** — keys `("tm", local_event_index, k)` and `("opp", local_event_index, k)` for the k-th teammate/opponent slot. This matches StatsBomb’s practice of anonymous freeze-frame players (`player_node_pids = -1`); geometry is correct per snapshot and not merged across time. Optional future design: deduplicate by stable ID and put `(dx, dy)` on `context_for` edge attributes instead.

- Player node features (for current version):
  - **Role / position embedding**:
    - Learn an embedding lookup `E_pos[position_name] ∈ R^{d_pos}` based on the primary or contextual position (“Left Wing”, “Right Back”, etc.).
  - **Team indicator**:
    - Binary or one-hot: `is_possession_team` vs `is_opponent_team`.
  - **Relative geometry**:
    - At the time of the focal event’s freeze frame, approximate:
      - Relative position `(dx, dy)` to the ball:
        - `dx = (player_x - ball_x) / pitch_length`, `dy = (player_y - ball_y) / pitch_width`.
      - (Velocity is **not** used at this stage; it is left as future work, since 360 only provides one snapshot per event.)
  - **Optional simple stats**:
    - E.g. count of touches in possession, boolean “in box”, etc., if desired.

- We **do not** use raw `player_id` as an input feature to avoid trivial identity learning.

### 5.2 Edge types

Within each possession graph:

1. **Temporal edges (`E_next`)**:
   - For events ordered as `e₁, e₂, …, e_T`:
     - Add edges `e_t → e_{t+1}` (and optionally `e_{t+1} → e_t` for undirected behaviour).
   - Encodes the **flow of the possession**.

2. **Actor edges (`E_actor`)**:
   - For each event `e` with `player_id = p`:
     - Connect `player_p ↔ event_e`.
   - **Actors:** one node per distinct `player_id`; multiple events attach to the same actor node.
   - This allows actor nodes to aggregate information from all their touches in the possession.

3. **Context edges (`E_context`)** (crucial):
   - For each event `e` with a 360 frame:
     - For each **off-ball teammate slot** in that frame (one node per slot at `e`, not shared across events):
       - Add edge `player_teammate_slot → event_e`.
     - For each **off-ball opponent slot** likewise:
       - Add edge `player_opponent_slot → event_e`.
   - These edges let the event embedding “see”:
     - Availability and position of passing options (teammates).
     - Local defensive pressure and blocking (opponents).

Additional edges we may add later (optional enhancements):

- **Pass edges**:
  - From a pass event to the recipient’s next touch (if identifiable).
- **Teammate edges (player–player)**:
  - Connect players on the same team (e.g. a clique or local neighbourhood).
- **Spatial proximity edges**:
  - Connect players or events within a certain distance on the pitch.

### 5.3 Representation choice: homogeneous vs heterogeneous

Two implementation strategies:

- **Homogeneous graph + type embeddings** (simpler to code):
  - Represent all nodes in a single `x` matrix, with:
    - Node-type one-hot or embedding (`is_event`, `is_player`).
  - A single `edge_index` and optional `edge_type` for relation types.

- **Heterogeneous graph** (cleaner semantics):
  - Use a library’s `HeteroData` with:
    - Node types: `{"event", "player"}`.
    - Edge types: `("event","next","event")`, `("player","acts_in","event")`, `("player","context_for","event")`, etc.
  - Use `HeteroConv` or per-relation GNN layers.

Either approach is acceptable; the design above is agnostic as long as the relations are respected.

### 5.4 Spatial_360 and redundancy

Phase 1’s 126‑D event vector includes a **Spatial_360** block (counts, mean positions, min distances) and a **period** one-hot (4‑D). Once we add explicit teammate and opponent nodes:

- If we feed both Spatial_360 and explicit player nodes, the model may rely on the easier **summary stats** and ignore the graph.

We therefore adopt the following policy:

- Keep the full 126‑D vectors on disk as Phase 1’s **canonical state representation**.
- For the **GNN encoder branch** (Phases 4–5):
  - Apply a `mask_spatial_360()` function to **zero out Spatial_360 dimensions** in event features before projection.
  - Rely on the **graph structure** (player nodes + `E_context` edges) for spatial context (options + pressure).

This forces the encoder to **use the graph** rather than the pre-aggregated spatial summary.

---

## 6. Phase 4 – GNN encoder and player embeddings

We now define how we turn possession graphs into:

- Event node embeddings `h_event`.
- Player node embeddings per possession `h_player^possession`.
- Global player embeddings `z_p`.

### 6.1 Per-event feature projection (mandatory)

The 126‑D event vectors mix:

- Many **sparse one-hots** (event type, position, outcomes, etc.), and
- Several **continuous scalars** (normalised coordinates, lengths, xG, etc.).

Feeding this directly into a GNN is suboptimal. We **must** first project to a well-behaved latent space.

Implementation:

1. For each event node:
   - Start from Phase 1 feature vector `x_raw ∈ R^126`.
   - Optionally apply:
     - `mask_future_info()` if this node is the imitation target (Phase 5).
     - `mask_spatial_360()` for the GNN branch to drop Spatial_360 dims.
2. Apply a small MLP with normalisation:
   - `h_0 = Linear(126 → d) → ReLU → Linear(d → d)`.
   - Apply `LayerNorm` (or similar) on the output.

This yields `x_event ∈ R^d` as the GNN input features for event nodes.

For player nodes:

- Construct raw feature vectors (role embedding + team indicator + relative position `(dx, dy)`).
- Optionally pass these through a separate MLP to map to the **same latent dimension** `d` for the GNN.

### 6.2 GNN encoder (per possession)

We use a **GATv2-based encoder** (or similar message-passing GNN) for each possession graph:

- Input:
  - Node features: `x_node` (either homogeneous or hetero).
  - Edge indices, and optionally relation types.
- Encoder:
  - 1–2 layers of GATv2 (or GraphSAGE/GCN) with:
    - Relation-aware weights if using hetero graphs.
  - Skip connections:
    - Project input features to latent space (`skip_proj`).
    - Add to GNN output (`out = h + skip`) to avoid over-smoothing.

Output per possession:

- Event node embeddings `h_event[i]`.
- Player node embeddings `h_player[p, possession]` for each player node.

### 6.3 Player-centric readout and global embedding `z_p`

We want **one global embedding** `z_p` per player that captures trait/style.

Procedure:

1. **Per-possession level:**
   - For each possession where player `p` has a node:
     - Extract the player node embedding `h_player^possession(p)`.

2. **Global player embedding:**
   - For each player `p`, collect the set `{h_player^possession(p)_i}` over all possessions they appear in.
   - Compute:
     - **Baseline**: simple mean  
       `z_p = mean_i(h_player^possession(p)_i)` (easy but can over-emphasise average location/role).
     - **Preferred**: attention-weighted pooling:
       - Learn a scoring function `score(h, context)` → `α_i` via a small MLP.
       - Normalise scores: `α_i = softmax_i(score_i)`.
       - `z_p = Σ_i α_i h_player^possession(p)_i`.
       - This lets the model emphasise **high-leverage possessions** (shots, line-breaking plays) and de-emphasise routine recycling.

`z_p` is used:

- In training as the trait vector in `f(s_t, z_p)`.
- In inference as the indexable embedding for player similarity.

---

## 7. Phase 5 – Training objectives

The encoder is trained so that:

- Given the same **state** \(s\), players with similar `z_p` act similarly.
- Embeddings also reflect **effectiveness**, not just style.

We use three kinds of objectives:

1. **Masked action / imitation loss** \(L_{\text{action}}\) – primary.
2. **Outcome / value loss** \(L_{\text{outcome}}\) – secondary.
3. **Contrastive loss** \(L_{\text{contrast}}\) – auxiliary (player-ID-based; see SYSTEM_DESIGN.md).

### 7.1 Masked action / imitation objective (primary)

For an event at time `t` by player `p`:

1. Build **state** \(s_t\):
   - Start from Phase 1 features + graph context **but**:
     - Apply `mask_future_info()` to **remove any fields that directly reveal the action or its consequences** from the event’s feature vector:
       - Allowed (examples): current location, play pattern, under_pressure / counterpress, pitch zone, period, team IDs, and graph-based context from teammate/opponent nodes. (**Note**: position is now **masked** — it flows through the player node channel instead; see SYSTEM_DESIGN.md.)
       - Forbidden (examples): `end_location`, delta, movement distance/angle, any pass/shot/dribble outcomes, `pass_type`, `pass_height`, `shot_type`, `pass_length`, `shot_xg`, `shot_first_time`, etc.
     - After masking, apply the per-event MLP + LayerNorm to get `x_event_t`.
   - Aggregate neighbourhood information via the GNN to get the final event embedding `h_event_t` (which encodes state `s_t`).

2. Define **action target** \(y_t\) via discretisation:
   - Because football actions are continuous and categorical, we **bin** continuous attributes:
     - **Action type**: categorical (pass, carry, shot, dribble, etc.).
     - **Subtypes**: cross, switch, through ball, cutback, etc.
     - **Direction bins**: partition the action angle into, say, 8–16 sectors.
     - **Length bins**: partition pass/carry length into around 5 bins (very short, short, medium, long, very long/switch).
     - Optionally: destination categories (into box vs outside, central vs wide).
   - Combine these into one or more discrete labels (multi-head classification).

3. Let **trait embedding** be `z_p`:
   - Computed via attention pooling over that player’s possession embeddings (or approximated within a batch).

4. Define the predictor:

\[
\hat{y_t} = f_\theta(h_{event,t}, z_p)
\]

5. Use an **action-prediction loss**:

\[
L_{\text{action}} = \mathbb{E}_{t,p}[\ell(\hat{y_t}, y_t)] \,,
\]

where \(\ell\) is typically a sum of **cross-entropy losses** over the discretised targets. We **avoid direct coordinate regression** at first, as it is harder to train and less stable.

Intuition:

- `h_event_t` encodes **state**, `z_p` encodes **trait**.
- If two players have similar `z_p`, they induce similar distributions over bins \(y_t\) for the same `h_event_t`.

### 7.2 Outcome / behaviour prediction (secondary)

To ensure embeddings reflect **impact**, not just style:

- Add one or more heads that predict:
  - Whether the possession ends in a shot or goal.
  - xG bucket for the possession.
  - Whether an action breaks lines or enters a dangerous zone.
- Define \(L_{\text{outcome}}\) as the sum of appropriate classification/regression losses for these heads.

This encourages the encoder to capture **value-relevant** patterns.

### 7.3 Contrastive objectives (auxiliary)

We regularise the embedding space with an **InfoNCE** contrastive loss on per-possession player embeddings `h_player`:

- **Positives**: Two `h_player` embeddings belonging to the **same player** from different possessions within the same batch.
- **Negatives**: All other `h_player` embeddings in the batch (different players).

The loss is \(L_{\text{contrast}}\) (InfoNCE) and is **auxiliary**, not primary. It mainly:

- Ensures same-player embeddings are **consistent** across possessions.
- Pushes different-player embeddings **apart** in the latent space.
- Smooths the embedding space for downstream similarity search.

### 7.4 Combined objective

Total loss:

\[
L = L_{\text{action}} + \lambda_{\text{outcome}} L_{\text{outcome}} + \lambda_{\text{contrast}} L_{\text{contrast}} \,,
\]

where \(\lambda_{\text{outcome}}\) and \(\lambda_{\text{contrast}}\) control the importance of the secondary and auxiliary tasks.

The encoder is shared across all tasks; only the heads differ.

---

## 8. Phase 6 – Inference and similarity search

### 8.1 Player embedding computation

At inference time:

1. Run (or reuse) Phase 1 to produce event features + metadata.
2. Build possession graphs (Phase 2–3) for the dataset of interest.
3. Run the trained GNN encoder:
   - Get event & player node embeddings per possession.
   - Aggregate per player to get `z_p` (attention pooling).
4. Store all `z_p` in:
   - A matrix `Z ∈ R^{n_players × d}`.
   - Optionally, an ANN index (e.g. FAISS) for fast k‑NN.

### 8.2 Similarity search and minimum sample size

To ensure stable embeddings:

- Impose a **minimum sample size** filter when building the search index:
  - e.g. require:
    - ≥ X minutes played, or
    - ≥ N on-ball events / possessions.
- This avoids including players whose embeddings are based on too little data.

For a query player `q`:

1. Retrieve `z_q`.
2. Compute similarity to all `z_p` (cosine similarity or Euclidean distance).
3. Return top‑k similar players, with optional filters:
   - Position group.
   - League, age, minutes thresholds.

Interpretation:

- Returned players have `z_p` close to `z_q`, i.e. they tend to make **similar action choices** in similar states.

---

## 9. Design choices and rationale (summary)

1. **StatsBomb 360** is used because it provides off-ball spatial context (teammates + opponents) needed to define “same situation”.
2. **No left/right mirroring** during training so sidedness and footedness are preserved; inference-time mirroring supports cross-sided queries.
3. **Possessions** are used as the natural unit of play; StatsBomb provides `possession` IDs.
4. **Event+Player graphs** are used to explicitly model who does what, in what sequence, with which options and pressure.
5. **Per-event projection** (126‑D → `d`) is mandatory to stabilise GNN training.
6. **Masked imitation** is the core loss, ensuring `z_p` captures **how** a player acts, not just where.
7. **Outcome heads** ensure representations are tied to value/impact.
8. **Attention pooling** ensures high-leverage actions dominate `z_p`, not routine recycling.
9. **Minimum sample size filters** ensure `z_p` is based on enough data to be meaningful.

Together, these choices make the system well-matched to the nature of StatsBomb + 360 data and to your goal of **finding the player who would act most similarly in the same situation**.

