# Phase 4: Model Architecture

Phase 4 defines the **model**: feature projections, heterogeneous GNN encoder, attention pooling, FiLM conditioning, and prediction heads. The model is instantiated at training time (Phase 5) and used for inference (Phase 6) and analysis (Phase 7).

---

## Goal

Learn a **player trait embedding** `z_p` (64-D) per player such that players who would act similarly in the same situation get similar embeddings. The main learning signal is **masked imitation**: given a game state (with action and identity masked), predict what action the player took.

---

## Components

### 1. Feature projections

- **EventProjection**: 126-D → 64-D (Linear → ReLU → Linear → LayerNorm).
- **PlayerProjection**: Position embedding (26 → 16-D) concat with [is_possession_team, dx, dy] → MLP → 64-D.

### 2. Heterogeneous GNN encoder (2-layer GATv2)

- **HeteroConv** with per-relation **GATv2Conv**.
- **Temporal** edge types (`next`, `prev`) use `edge_dim=1` (time-delta); other edge types have no edge attributes.
- Layer 1: 64-D → 512-D (4 heads × 128-D, concatenated).
- Layer 2: 512-D → 64-D (single head).
- Learned linear skip connections from input to output after Layer 2.

After encoding:

- **h_event** (E × 64): situation encoding.
- **h_player** (P × 64): player-in-context encoding for this possession.

### 3. Attention pooling (per-player)

Within a batch (and at inference across all possessions for a player):

- Gather `h_player` per player (one per possession after deduplication by player node).
- Score: `Linear(Tanh(Linear(h)))` → scalar; softmax over possessions → weights α_i.
- **z_p** = Σ α_i · h_i.

Trained end-to-end so that high-leverage possessions can be up-weighted.

### 4. FiLM conditioning (player + position → action prediction)

- **Dual-channel position design**: position enters `z_p` through the player-node path (PlayerProjection → GNN → h_player → pool), retaining coarse role structure. It also enters FiLM through a dedicated embedding for within-role prediction.
- FiLM position embedding: `pos_emb = Embedding(position_idx) ∈ R^{16}`.
- Conditioning vector: `cond = [z_p ; pos_emb]` (64 + 16 = 80-D).
- γ = Linear(cond), β = Linear(cond).
- h_conditioned = (1 + γ) ⊙ h_event + β.
- Action heads (type, angle bin, length bin) take **h_conditioned** so that predictions depend multiplicatively on both `z_p` and the actor’s positional role.

### 5. Prediction heads

| Head | Input | Output |
|------|--------|--------|
| Action type | FiLM-conditioned h | 14 logits |
| Angle bin | FiLM-conditioned h | 9 logits |
| Length bin | FiLM-conditioned h | 5 logits |
| Outcome (shot/goal) | h_event only | 2 logits |

Outcome uses only h_event (no z_p) because possession outcomes are driven by game state.

---

## Position ablation (`--ablate_position`)

When `ModelConfig.ablate_position = True`, all explicit position information is removed from the model:

- **PlayerProjection**: The 16-D position embedding is replaced with a zero vector, so player nodes carry only the team flag and spatial offset (dx, dy). The model must learn player role structure purely from behavioral patterns.
- **FiLM conditioning**: The dedicated `pos_emb` embedding is replaced with zeros, so the conditioning vector is effectively `[z_p ; 0]`. Action prediction cannot shortcut through positional role.

Position indices remain available in graph metadata for contrastive hard-negative mining (same-position-group negatives still work), but they do not enter the learned representations.

This ablation is critical for the `pos_ablated_split_ctx` family of models, which combine position ablation with split-context edges and strong uniformity loss (lambda=1.0, t=4.0) to produce the best-performing embeddings for substitute retrieval. Registry pipelines `acts_in_dropout`, `acts_in_dropout_pos`, and `acts_in_dropout_pos_gu` extend that family with action-stream dropout and optional position / group-uniformity terms; see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for metrics.

---

## Split context edges

When `--split_context_edges` is active, the GNN encoder allocates separate `GATv2Conv` parameters for `context_for_tm` and `context_for_opp` instead of a single `context_for`. The edge type list is determined by `get_edge_types(split_context)` in `gnn_encoder.py`, driven by `GraphConfig.split_context_edges`.

The model constructor accepts an optional `graph_config` parameter to configure this at instantiation time.

---

## Encoding API: full graph vs counterfactual-safe

| Method | `acts_in` edges | Typical use |
|--------|-----------------|-------------|
| `encode_possession` | **Included** | Training, validation, any path that needs the same message passing as `forward` without heads. |
| `encode_possession_counterfactual` | **Dropped** | Phase 7 situation comparison, policy diagnostic, and related tools: extract `h_event` **without** the player→event channel that injects actor identity into the event node. `(event, performed_by, player)` and context/temporal edges remain. |

See `docs/PHASE7.md` and `docs/FUTURE_IMPROVEMENTS.md` §8 for rationale, training-vs-analysis scope, and optional further hardening.

---

## Code

| Component | Location |
|-----------|----------|
| Full model (FiLM + heads) | `src/phase4_model/model.py` |
| GNN encoder (HeteroConv, edge types) | `src/phase4_model/gnn_encoder.py` |
| Projections (event, player) | `src/phase4_model/projections.py` |
| Attention pooling | `src/phase4_model/pooling.py` |
| Config (dims, heads, dropout, etc.) | `src/config.py` → `ModelConfig`, `GraphConfig` |

---

## Key config (see SYSTEM_DESIGN.md)

| Parameter | Value |
|-----------|-------|
| `latent_dim` | 64 |
| `num_layers` | 2 |
| `num_heads` | 4 |
| `hidden_dim` | 128 |
| `position_embed_dim` | 16 |
| `dropout` | 0.1 |
| `n_action_types` | 14 |
| `n_angle_bins` | 9 |
| `n_length_bins` | 5 |
| `ablate_position` | False (True for pos_ablated variants) |

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 4 in full (rationale: FiLM, pooling, heads).
- [PHASE3.md](PHASE3.md) — Graph format consumed by the model.
- [PHASE5.md](PHASE5.md) — Training (masking, losses).
