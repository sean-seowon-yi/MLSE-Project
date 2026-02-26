# GNN_StatsBomb — Situation-Aware Player Similarity (StatsBomb 360)

### Purpose

`GNN_StatsBomb` builds a **player similarity system** from **StatsBomb 360** data.  
Unlike pure tracking, we have:

- Discrete **on-ball events** (passes, shots, carries, duels, etc.), and  
- **360 freeze frames** capturing where teammates and opponents were at the moment of each event.

The objective is:

> Given a query player, find players who would act **similarly in the same situations** — i.e. when placed into the same game state (location, context, spatial layout), they tend to choose similar actions.

This is achieved via:

- A 7‑phase pipeline (data → possessions → graphs → model → embeddings → search → analysis).  
- A heterogeneous GNN that reasons over **events + players + 360 context**.  
- A masked imitation objective that predicts **actions from pre-action state + player traits**.

---

## Phases & workflow

All phases are orchestrated by `main.py` via a `--mode` CLI.

### Phase 1 – Data preparation & feature encoding (`mode=prepare`)

**Files:** `src/data_preparation.py`, `src/feature_encoder.py`

1. Load StatsBomb events (and 360) from `../StatsBomb/data` (configurable).  
2. Filter to a core set of on-ball event types (passes, shots, carries, duels, etc.).  
3. Require a 360 freeze frame (when `data.use_360=True`):  
   - Only events with 360 context are kept.  
4. Encode each event into a **126‑D feature vector** including:
   - Event type, locations, displacements, play pattern, position, body part, outcomes, scalars (duration, pressure, xG, etc.), pitch zone, 9‑D 360 summary (counts & distances), and match period (4‑D one‑hot).
   - Coordinates are normalised to [0,1].  
   - Left and right positions are **not mirrored** (Left Wing vs Right Wing stay distinct).
5. Save:
   - `processed_data/event_features.npy`  
   - `processed_data/event_metadata.parquet`  
   - `processed_data/freeze_frames.pkl` (raw 360 lists, aligned by event index)  
   - `feature_names.json`, `data_stats.json`

### Phase 2 – Possession construction (`mode=build_possessions`)

**File:** `src/phase2_possession/possession_builder.py`

- Group events into **possessions** by `(match_id, possession_number, possession_team_id)`.  
- Sort within each possession by `(period, minute, second, original_index)`.  
- For each possession, store:
  - `event_indices`, `player_ids`, `team_ids`, `event_types`, `position_names`.  
  - Per‑event timestamps with millisecond precision (parsed from StatsBomb's period‑relative `timestamp` field; falls back to `minute*60 + second`) for tempo.  
  - Labels: `ends_in_shot`, `ends_in_goal`, `total_xg`.  
- Save a list of `Possession` objects to `processed_data/possessions.pkl`.

### Phase 3 – Heterogeneous graph construction (`mode=build_graphs`)

**Files:** `src/phase3_graph/graph_builder.py`, `src/phase3_graph/masking.py`

Build a **PyG `HeteroData` graph per possession**:

- **Node types:**
  - `event`: one node per on-ball event, features = 126‑D vector with the 9‑D Spatial_360 block **zeroed** (`mask_spatial_360`).  
  - `player`: one node per distinct player in the possession (actors + off-ball 360 players), features = `[position_idx, is_possession_team, dx, dy]`, where `dx, dy` are relative to the ball.

- **Edge types:**
  - `("event", "next", "event")`: temporal edges `e_t → e_{t+1}` with **1‑D time‑delta edge_attr**, `min(Δt/30,1.0)`.  
  - `("event", "prev", "event")`: optional reverse temporal edges (same edge_attr).  
  - `("player", "acts_in", "event")`: actor edges from player node to events they perform.  
  - `("event", "performed_by", "player")`: reverse actor edges.  
  - `("player", "context_for", "event")`: edges from off‑ball players (from 360) to the focal event.

- **Masking utilities:**
  - `mask_spatial_360`: zero out indices `[113..121]` so spatial context is learned via graph structure.  
  - `mask_future_info`: zero out any feature that reveals the action or its outcome, **including position one‑hot (32–57)**.  
    - After masking, only situational state remains: location, play pattern, under_pressure, counterpress, pitch zone, and period.

Graphs, with attached metadata, are saved to `processed_data/possession_graphs.pkl`.

### Phase 4 – Model architecture (`src/phase4_model/*`)

Core components:

- `EventProjection`: 126‑D → d (64) with `Linear → ReLU → Linear → LayerNorm`.  
- `PlayerProjection`: position embedding (26→16) + [team_flag, dx, dy] → d with MLP + LayerNorm.  
- `PossessionGNNEncoder`: 2‑layer **heterogeneous GATv2**:
  - Per‑relation `GATv2Conv` for each edge type.  
  - Temporal relations use `edge_dim=1` (time‑delta edge_attr).  
  - Learned linear skip connections for both node types (event/player).
- `AttentionPooling`: attention‑weighted pooling over per‑possession `h_player` embeddings to form per‑player trait embeddings.
- `PlayerSimilarityModel`:
  - Projects node features → runs GNN → pools actor embeddings per player → obtains `z_p`.  
  - Uses **FiLM conditioning** (`gamma(z_p), beta(z_p)`) to modulate `h_event` for action prediction.  
  - Heads:
    - Action: type (14 classes), angle bin (9), length bin (5) from FiLM‑conditioned `h_event`.  
    - Outcome: shot/goal head from `h_event` only.

### Phase 5 – Training (`mode=train`)

**Files:** `src/phase5_training/*`

- `PossessionGraphDataset`:
  - Loads graphs and `event_features.npy`.  
  - Uses **unmasked** features to build discrete targets:
    - Action type (14), angle bin (8 sectors + no‑angle), length bin (5).  
  - Applies `mask_future_info` to the graph’s event node features at sample time (strict pre‑action state).
- `CombinedLoss`:
  - `L_action` — Focal loss (γ=2.0) with class weights for type & length bins.  
  - `L_outcome` — BCE for possession shot/goal flags (weight λ_outcome=0.5).  
  - `L_contrastive` — InfoNCE on actor‑only `h_player` embeddings (same `player_id` positives, others negatives; temperature τ=0.05; weight λ_contrast=1.0).
- `Trainer`:
  - Adam + ReduceLROnPlateau (on full validation loss), gradient clipping, early stopping.  
  - Logs per‑epoch breakdown: total, action, outcome, contrastive, LR.

The masked imitation objective forces the model to answer:

> Given a situation (masked event features + graph context) and a player’s trait vector `z_p`, **what action will this player take?**

Similar `z_p` vectors then imply similar action distributions across situations.

### Phase 6 – Inference & similarity (`mode=inference`, `mode=search`)

**Files:** `src/phase6_inference/*`

- `EmbeddingGenerator`:
  - Loads best checkpoint.  
  - Runs the encoder over batched possession graphs (with runtime masking).  
  - For each player, collects per‑possession actor embeddings and applies the same attention pooling used during training.  
  - Applies a **minimum sample threshold** (e.g. 50 possessions) for stability.  
  - Saves:
    - `embeddings/player_embeddings.npy` — `Z ∈ ℝ^{n_players × d}`.  
    - `embeddings/player_info.parquet` — IDs, names, positions, n_possessions.

- `SimilaritySearcher`:
  - Computes cosine similarity over `Z`.  
  - `find_similar_players(query_player_id, top_k, filters)` returns a ranked DataFrame of neighbours.  
  - Filters support position groups, minimum possessions, etc.

### Phase 7 – Situation-level analysis (`mode=analyze`)

**Files:** `src/phase7_analysis/*`

- `ReportBuilder`:
  - Loads `Z`, `player_info`, possession graphs, and the trained model.  
  - Selects query players (either supplied or random).  
  - Finds nearest neighbours via cosine similarity.  
  - Samples real events from each query player and uses `SituationComparator` to:
    - Extract the event’s `h_event` (state embedding).  
    - Substitute each player’s global `z_p` (query + neighbours) into FiLM and run the action heads.  
    - Compare predicted action type / direction / length distributions in that **fixed state**.
  - Writes:
    - Text reports (per-query) with detailed descriptions of situations and per‑player predictions.  
    - Bar charts of action probabilities and angle distributions.  
    - PCA plots of the global embedding space with neighbourhoods highlighted.

This phase is how you **interpret and debug** whether similarity aligns with “would act similarly in the same situation.”

---

## CLI summary

From `GNN_StatsBomb/`:

```bash
# Install deps
pip install -r requirements.txt

# Phase 1
python main.py --mode prepare

# Phase 2
python main.py --mode build_possessions

# Phase 3
python main.py --mode build_graphs

# Phase 5 (train)
python main.py --mode train

# Phase 6A (embeddings)
python main.py --mode inference

# Phase 6B (search)
python main.py --mode search --player_id <STATS_BOMB_PLAYER_ID>

# Phase 7 (analysis)
python main.py --mode analyze

# Or run Phases 1–6A in one go:
python main.py --mode full_pipeline
```

---

## Further reading

- **System design & rationale**: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)  
  - Problem statement, data encoding, graph design, model, losses, masking, assumptions.  
  - Audit trail of fixes and architectural decisions.

- **Documentation index**: [docs/README.md](docs/README.md) — overview and links to all phase docs and other references.

- **Phase docs** (one per pipeline stage):
  - [docs/PHASE1.md](docs/PHASE1.md) — Data preparation & 126-D encoding
  - [docs/PHASE2.md](docs/PHASE2.md) — Possession construction
  - [docs/PHASE3.md](docs/PHASE3.md) — Heterogeneous graph construction
  - [docs/PHASE4.md](docs/PHASE4.md) — Model architecture
  - [docs/PHASE5.md](docs/PHASE5.md) — Training
  - [docs/PHASE6.md](docs/PHASE6.md) — Inference & similarity search
  - [docs/PHASE7.md](docs/PHASE7.md) — Situation-level analysis

- **Other docs**:
  - [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md) — Edge cases, clamping, missing 360, role labels.
  - [docs/PLAYER_SIMILARITY_FINAL_PLAN.md](docs/PLAYER_SIMILARITY_FINAL_PLAN.md) — High-level plan and design notes.

These documents are kept consistent with the current implementation and are the best reference when extending or reviewing the system.

