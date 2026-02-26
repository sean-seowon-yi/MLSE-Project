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

### 4. FiLM conditioning (player → action prediction)

- γ = Linear(z_p), β = Linear(z_p).
- h_conditioned = (1 + γ) ⊙ h_event + β.
- Action heads (type, angle bin, length bin) take **h_conditioned** so that predictions depend multiplicatively on z_p.

### 5. Prediction heads

| Head | Input | Output |
|------|--------|--------|
| Action type | FiLM-conditioned h | 14 logits |
| Angle bin | FiLM-conditioned h | 9 logits |
| Length bin | FiLM-conditioned h | 5 logits |
| Outcome (shot/goal) | h_event only | 2 logits |

Outcome uses only h_event (no z_p) because possession outcomes are driven by game state.

---

## Code

| Component | Location |
|-----------|----------|
| Full model (FiLM + heads) | `src/phase4_model/model.py` |
| GNN encoder | `src/phase4_model/gnn_encoder.py` |
| Projections | `src/phase4_model/projections.py` |
| Attention pooling | `src/phase4_model/pooling.py` |
| Config (dims, heads, etc.) | `src/config.py` |

---

## Key config (see SYSTEM_DESIGN.md)

- latent_dim 64, num_layers 2, num_heads 4, hidden_dim 128.
- n_action_types 14, n_angle_bins 9, n_length_bins 5.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 4 in full (rationale: FiLM, pooling, heads).
- [PHASE3.md](PHASE3.md) — Graph format consumed by the model.
- [PHASE5.md](PHASE5.md) — Training (masking, losses).
