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

**Checkpoints & embeddings:** Training and inference write to **tagged subdirectories**, not the project root. The default **baseline** tag uses `checkpoints/baseline/` (e.g. `best_model.pt`, per-epoch checkpoints, `training_history.json`) and `embeddings/baseline/` (e.g. `player_embeddings.npy`, `player_info.parquet`). **Test-set metrics** from `--mode evaluate` default to `evaluations/baseline/test_metrics/` (or `evaluations/{tag}/test_metrics/`). Other pipeline variants use `checkpoints/{tag}/` and `embeddings/{tag}/` (see `--tag` and related flags). The legacy root-level `checkpoints/best_model.pt` and `embeddings/player_embeddings.npy` paths are no longer used.

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
  - `player`: **actors** — one node per distinct `player_id` in the possession; **360 context** — one node per teammate/opponent **slot per event** (not deduplicated across events). Features = `[position_idx, is_possession_team, dx, dy]` relative to the ball (`dx=dy=0` for actors).

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
- `PlayerProjection`: position embedding (26→16) + [team_flag, dx, dy] → d with MLP + LayerNorm. When `ablate_position=True`, the position embedding is zeroed.  
- `PossessionGNNEncoder`: 2‑layer **heterogeneous GATv2**:
  - Per‑relation `GATv2Conv` for each edge type.  
  - Temporal relations use `edge_dim=1` (time‑delta edge_attr).  
  - Learned linear skip connections for both node types (event/player).
- `AttentionPooling`: attention‑weighted pooling over per‑possession `h_player` embeddings to form per‑player trait embeddings.
- `PlayerSimilarityModel`:
  - Projects node features → runs GNN → pools actor embeddings per player → obtains `z_p`.  
  - Uses **FiLM conditioning** with a **dual-channel position design**: position enters `z_p` via the player-node path (retaining coarse role structure) AND via a dedicated FiLM embedding (`pos_emb = Embedding(position_idx)`, `cond = [z_p ; pos_emb]`) for within-role prediction sharpening. When `ablate_position=True`, both position channels output zeros, forcing purely behavioral embeddings.  
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
  - `L_contrastive` — InfoNCE on actor‑only `h_player` embeddings with **same-position-group hard negatives** (positives = same `player_id`, negatives = different `player_id`s in the same coarse position group). The softmax denominator includes **positives and negatives** (standard supervised contrastive form). Temperature τ=0.05 by default; weight λ_contrast=0.5.
  - `L_pooled_uniformity` — Gaussian‑potential uniformity loss on **pooled `z_p`** to spread embeddings on the unit hypersphere and widen cosine similarity gaps (weight λ_pooled=0.3, sensitivity `uniformity_t=2.0`).
  - `L_alignment` (optional) — EMA cross-batch alignment loss on pooled `z_p` to enforce self-consistency across batches. Uses an EMA memory bank (momentum=0.999) to maintain smoothed historical embeddings per player. Enabled via `--ema_alignment` (weight λ_alignment=0.3).
- `Trainer`:
  - Adam + ReduceLROnPlateau (on `val_supervised`), gradient clipping, early stopping.  
  - Logs per‑epoch breakdown: total, action, outcome, contrastive, uniform, alignment (when EMA enabled), LR.

The masked imitation objective forces the model to answer:

> Given a situation (masked event features + graph context) and a player’s trait vector `z_p`, **what action will this player take?**

Similar `z_p` vectors then imply similar action distributions across situations.

### Test-set evaluation (`mode=evaluate`)

After training, you can evaluate the **best checkpoint** on the held-out **test set** (same 70/15/15 split by match_id):

- **Action heads**: accuracy and macro F1 for action type, angle bin, and length bin; confusion matrices (saved as PNGs).
- **Outcome head**: accuracy, BCE, and AUC-ROC for `ends_in_shot` and `ends_in_goal`; ROC curves (saved as PNG).

Outputs default to **`evaluations/baseline/test_metrics/`** (or `evaluations/{tag}/test_metrics/`): `test_metrics.json` plus `confusion_matrix_*.png` and `outcome_roc.png`. Run with:

```bash
python main.py --mode evaluate
```

Requires Phase 1–3 outputs and a trained checkpoint (`checkpoints/baseline/best_model.pt`).

### Phase 6 – Inference & similarity (`mode=inference`, `mode=search`)

**Files:** `src/phase6_inference/*`

- `EmbeddingGenerator`:
  - Loads best checkpoint.  
  - Runs the encoder over batched possession graphs (with runtime masking).  
  - For each player, collects per‑possession actor embeddings and applies the same attention pooling used during training.  
  - Applies a **minimum sample threshold** (e.g. 50 possessions) for stability.  
  - Saves:
    - `embeddings/baseline/player_embeddings.npy` — `Z ∈ ℝ^{n_players × d}`.
    - `embeddings/baseline/player_info.parquet` — IDs, names, positions, n_possessions.

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
    - **PCA and t-SNE** plots of the global embedding space (three colour schemes: group, subgroup, position) and per-query neighbourhood plots for both methods.

This phase is how you **interpret and debug** whether similarity aligns with “would act similarly in the same situation.”

**Possession animation** (`--mode possession_animation`): exports an **MP4 or GIF** of one or more possession graphs — ball trail (discrete event locations), freeze-frame players, and **counterfactual** action arrows for two specified players or a **dynamic substitute** (top-1 cosine neighbour of the on-ball actor). Requires `--pipeline` (same as training), checkpoint, embeddings, and graphs. Examples:

```bash
python main.py --mode possession_animation --pipeline acts_in_dropout_pos_gu --list-possessions
python main.py --mode possession_animation --pipeline acts_in_dropout_pos_gu --graph-index 0 --players 123,456 --output out.mp4
python main.py --mode possession_animation --pipeline acts_in_dropout_pos_gu --random --dynamic-substitute --output clip.mp4
```

Implementation: `src/phase7_analysis/possession_animation.py`. Not run by `full_eval`.

**Empirical behavioral fidelity** (`--mode empirical_behavioral`): standalone run of the **observed-action** agreement metrics (also executed as step 9 of `full_eval`). Writes under `evaluations/{tag}/empirical_behavioral/`. Implementation: `src/phase6_inference/empirical_behavioral.py`.

### Named pipeline registry

`main.py` defines a `PIPELINE_REGISTRY` with **11 named GNN pipelines**, each with a distinct architectural and training configuration:

| Pipeline | Key features |
|----------|-------------|
| `baseline` | Default config, unified context edges |
| `player_samp` | Player-aware batch sampling (K=16, M=6) |
| `split_ctx` | Split teammate/opponent context edges |
| `split_ctx_ps` | Split context + player sampling |
| `pos_ablated` | Position ablation (zeroed position embeddings) |
| `pos_ablated_split_ctx` | Position ablation + split context + strong uniformity (l=1.0, t=4.0) |
| `pos_ablated_split_ctx_v2` | Same as above but l_pooled=0.7 (reduced uniformity) |
| `pos_ablated_split_ctx_ema` | pos_ablated_split_ctx + EMA alignment loss (l=0.3) |
| `acts_in_dropout` | pos_ablated_split_ctx stack + action-stream dropout (p=0.3) |
| `acts_in_dropout_pos` | acts_in_dropout + position regularizer (`lambda_pos=0.3`) |
| `acts_in_dropout_pos_gu` | `acts_in_dropout_pos` + stronger group uniformity (`uniformity_group_weight=3.0`) |

Use `--mode full_eval_all` to evaluate all registered pipelines. Results are saved under `evaluations/{tag}/` per pipeline (see `full_eval` above). See [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) for comprehensive cross-model analysis.

### Gender-aware evaluation

All similarity search and evaluation modes automatically filter candidates by gender — male query players only receive male candidates, female queries only receive female candidates. Gender is derived from competition metadata (e.g. Women's World Cup → female). This applies to:

- Similarity search (`--mode search`)
- Ground-truth evaluation (`--mode ground_truth`)
- Self-consistency evaluation (`--mode self_consistency`)
- Policy diagnostic (`--mode policy_diagnostic`)
- Phase 7 analysis (`--mode analyze`)
- FIFA stat comparison (`--mode fifa_comparison`)

No special flag is needed; gender filtering is always active.

### FIFA stat comparison (`mode=fifa_comparison`)

Validates the GNN similarity system against external FIFA/EA Sports FC player attributes. Requires matched FIFA CSVs in `FIFA_data/`; build them with `match_fifa_players.py` (name normalization, country aliases, year-based FIFA version mapping).

- Samples 10 male + 10 female players (stratified by position group, random each run), plus **one or two famous players** when present in the pool.
- For each, finds the top-1 same-gender substitute via GNN cosine similarity that also has FIFA data.
- Compares main stats and detailed sub-attributes (goalkeeper-specific radar layout when relevant).
- Generates multiple PNGs: radar grid, similarity vs stat distance, breakdowns, summary-style figures (e.g. dumbbell / heatmap), category views, and sub-attribute detail heatmaps.

Standalone script (same logic): `test_fifa_comparison.py`.

### Unified evaluation pipeline (`mode=full_eval`, `mode=full_eval_all`)

Run all post-training evaluations for one or all pipeline variants in a single command:

```bash
# Evaluate the baseline pipeline
python main.py --mode full_eval

# Evaluate a specific tagged pipeline
python main.py --mode full_eval --tag split_ctx --split_context_edges

# Evaluate ALL registered pipelines (11 GNN variants)
python main.py --mode full_eval_all --eval_output_dir ./evaluations
```

`full_eval` runs **11** steps, in order: **inference** → **test_metrics** → **ground_truth** → **position_retrieval** → **split_half** (skipped for GNN; runs for heuristics with an event-based `embed_fn`) → **qualitative_neighbors** → **self_consistency** → **policy_diagnostic** → **empirical_behavioral** → **analysis** → **fifa_comparison**. Outputs are saved under `evaluations/{tag}/{step}/`, where `{tag}` is `config.tag` or `baseline` if the tag is empty (matches each registry entry’s folder name in practice). Configurable root: `--eval_output_dir`.

Related modes: **`generate_heuristics`**, **`eval_heuristics`**, **`full_eval_all_with_heuristics`** (orchestrates heuristic embedding generation and the same evaluation layout).

**Name lookup:** `python find_player.py "<partial name>"` — accent folding + fuzzy match; use printed `player_id` with `--mode search`.

---

## CLI summary

From `GNN_StatsBomb/`:

```bash
# Install deps (includes rapidfuzz for find_player.py)
pip install -r requirements.txt

# Phase 1
python main.py --mode prepare

# Phase 2
python main.py --mode build_possessions

# Phase 3
python main.py --mode build_graphs

# Phase 5 (train)
python main.py --mode train

# Test-set evaluation (optional)
python main.py --mode evaluate

# Phase 6A (embeddings)
python main.py --mode inference

# Phase 6B (search — requires numeric StatsBomb player_id)
python main.py --mode search --player_id <STATS_BOMB_PLAYER_ID>

# Optional: resolve name → id (accent + fuzzy; requires rapidfuzz)
python find_player.py "mbappe"
python find_player.py --tag split_ctx de bruyne

# Phase 7 (analysis)
python main.py --mode analyze

# Empirical behavioral fidelity (standalone; also inside full_eval)
python main.py --mode empirical_behavioral

# Possession animation (requires --pipeline and graph/player args)
python main.py --mode possession_animation --pipeline acts_in_dropout_pos_gu --list-possessions

# FIFA stat comparison
python main.py --mode fifa_comparison

# Run all evaluations for current pipeline
python main.py --mode full_eval

# Run all evaluations for ALL 11 registered pipeline variants
python main.py --mode full_eval_all --eval_output_dir ./evaluations

# Heuristics + evaluate them like GNN (after full_eval_all or on demand)
python main.py --mode generate_heuristics
python main.py --mode eval_heuristics --eval_output_dir ./evaluations
python main.py --mode full_eval_all_with_heuristics --eval_output_dir ./evaluations

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
  - [docs/FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md) — SOTA assessment and improvement roadmap.
  - [docs/PLAYER_SIMILARITY_FINAL_PLAN.md](docs/PLAYER_SIMILARITY_FINAL_PLAN.md) — High-level plan and design notes.
  - [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) — Cross-model evaluation results, policy diagnostics, empirical behavioral metrics.
  - [docs/models/VIEW_CONSISTENCY.md](docs/models/VIEW_CONSISTENCY.md) — View-consistency variant (design proposal; **not implemented** in code).

These documents are kept consistent with the current implementation and are the best reference when extending or reviewing the system.

