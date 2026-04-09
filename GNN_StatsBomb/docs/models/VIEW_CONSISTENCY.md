# View-Consistency Pipeline (`view_consistency`)

> **Not implemented** — This document is a **design proposal only**. There is no `ViewConsistencyLoss` in `src/phase5_training/losses.py`, no `--view_consistency` or `--lambda_vc` flags in `main.py`, and no `PIPELINE_REGISTRY` entry. **Do not copy-paste the command in §10**; it will not run. For the real training objective, see [PHASE5.md](../PHASE5.md) and [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md).
>
> **Extends (if built):** `pos_ablated_split_ctx`  
> **Goal:** Find the best substitute player by learning context-invariant player trait embeddings

---

## 1. Motivation

### The Problem with the Current Loss Stack

The best-performing model (`pos_ablated_split_ctx`) uses a 4-term loss:

```
L = L_action + 0.5*L_outcome + 0.5*L_contrastive + 1.0*L_uniformity
```

The EMA variant adds a 5th term (`L_alignment`), bringing the total to 6+ hyperparameters. This creates fragility -- a 30% change to a single weight (`lambda_pooled_contrast` 1.0 -> 0.7) caused complete model collapse in the `pos_ablated_split_ctx_v2` experiment.

The root cause is that each auxiliary loss was added to patch a specific weakness of the previous terms, creating conflicting gradient directions:

| Loss | Pulls toward | Conflicts with |
|------|-------------|----------------|
| L_contrastive | Same-player h_player close | L_uniformity pushing z_p apart |
| L_uniformity | All z_p spread on hypersphere | L_contrastive pulling clusters tight |
| L_alignment (EMA) | z_p stable across batches | L_action evolving representations |

### The Insight

`L_action` via FiLM conditioning is already the strongest anti-collapse mechanism when position is ablated. Without positional shortcuts, FiLM **must** differentiate z_p across players to predict actions correctly. This provides a natural spreading force that makes explicit uniformity the only needed safety net.

A single **view-consistency** loss can replace both contrastive and EMA alignment by directly operating on z_p (the level that matters for substitute search) rather than on h_player (per-possession level, wrong abstraction for the downstream task).

---

## 2. What Is Being Extended

### Base Model: `pos_ablated_split_ctx`

Key architectural properties inherited unchanged:

| Component | Detail |
|-----------|--------|
| **Position ablation** | `PlayerProjection` ignores `position_idx`, uses only (team_flag, dx, dy). FiLM conditions on z_p alone, no position embedding channel. |
| **Split-context edges** | 360 freeze-frame edges split into `context_for_tm` (teammate) and `context_for_opp` (opponent), giving HeteroConv separate attention weights per relation. |
| **GNN encoder** | 2-layer heterogeneous GATv2 with skip connections. 5 edge types: `next`, `prev`, `acts_in`, `performed_by`, `context_for_tm`, `context_for_opp`. |
| **Attention pooling** | Learned `score_net` (Linear -> Tanh -> Linear) aggregates per-possession actor embeddings into one z_p per player via scatter-softmax. |
| **FiLM conditioning** | `h_conditioned = (1 + gamma(z_p)) * h_event + beta(z_p)`. Scale centered at identity so event information flows from epoch 1. |
| **Prediction heads** | Action type (14-class focal), angle bin (9-class focal), length bin (5-class focal), outcome (2-D BCE for shot/goal). |

### What Made `pos_ablated_split_ctx` a Strong Retrieval Baseline

From [EVALUATION_RESULTS.md](../EVALUATION_RESULTS.md) (verify numbers after each re-run):

- **Spearman rho (overall): ~0.93** on the policy diagnostic — strong behavioral ordering for this architecture. (The highest rho in the suite is **`pos_ablated` at ~0.97**, driven in part by embedding compression; retrieval metrics for that model are weaker.)
- **Substitute quality ratio:** aggregate random-k JS / top-k JS ≈ **6.6** (see §4.2).
- **Ground-truth tier-1 hit@10:** **0.2** on the five tier-1 pairs (same as several split-context variants; not uniquely best).
- **Competition-split mean rank:** **~67** (strong cross-competition stability for the family).
- Position ablation forces z_p to carry player-distinguishing information, making FiLM depend on the embedding when position channels are zeroed.

---

## 3. What Is Changing

### Loss Function: Before and After

**Before (pos_ablated_split_ctx):**

```
L = L_action
  + 0.5  * L_outcome
  + 0.5  * L_contrastive      (InfoNCE on h_player, same-position hard negatives)
  + 1.0  * L_uniformity        (Gaussian-potential on z_p, t=4.0)
```

4 terms, 4 lambda weights + 2 internal knobs (temperature=0.05, uniformity_t=4.0).

**After (view_consistency):**

```
L = L_action
  + 0.5  * L_outcome
  + 0.5  * L_view_consistency  (cross-view cosine agreement on z_p)
  + 1.0  * L_uniformity        (Gaussian-potential on z_p, t=4.0)
```

4 terms, 3 lambda weights + 1 internal knob (uniformity_t=4.0).

### What Is Removed

| Removed Component | Reason |
|-------------------|--------|
| **ContrastiveLoss** (InfoNCE on h_player) | Subsumed by view-consistency at the z_p level. Contrastive operated at the wrong abstraction (per-possession h_player) for a task that cares about pooled z_p. |
| **EMA memory bank** | View-consistency provides intra-batch cross-view stability, a stronger and more immediate signal than cross-batch EMA smoothing. No external state needed. |
| **AlignmentLoss** | Replaced by view-consistency. |
| **Temperature annealing** | Only existed for contrastive loss. |

### What Is Added

- One new loss class: `ViewConsistencyLoss`.
- **Player-aware batch sampling** with K=32 players x M=8 possessions = 256 items per batch. VC requires multiple possessions per player to form meaningful mean-pooled "views" (4-vs-4 splits). Random shuffling at batch_size=256 gives mostly pairs (n=2) due to the birthday problem, which degenerates VC into a pairwise cosine loss between individual embeddings -- the wrong abstraction level. Player-aware sampling guarantees 8 possessions per player, enabling real view averaging. Contrastive-specific overrides (temperature annealing, pooled-weight annealing) are disabled since there is no contrastive loss.
- **Why this avoids the `player_samp` paradox:** The original `player_samp` models used InfoNCE contrastive loss, whose negative-pair term actively pushed different players into tight identity clusters (16 players per batch = 15 negatives per anchor every step). VC has **no negative-pair component** -- it only asks "do my two views agree?". Different-player separation is handled entirely by L_action (FiLM needs distinct z_p) and L_uniformity (geometric spread). With K=32 (vs K=16 in original `player_samp`), uniformity sees 2x more unique z_p per batch.

---

## 4. View-Consistency Loss: Design

### Core Idea

For each player with >= 2 possessions in a batch, compute the centroid (mean) of their per-possession actor embeddings, then pull every individual embedding toward the centroid via `1 - cosine_similarity(emb_i, centroid)`.

This per-embedding centroid-pull provides:

1. **Same-player attraction** (replaces contrastive): every embedding is pulled toward the player's average representation.
2. **Cross-context stability** (replaces EMA alignment): if individual possession embeddings cluster tightly around a centroid, any subset will produce a similar pooled z_p.
3. **Implicit anti-collapse**: if all z_p collapse to the same point, L_action spikes because identical z_p cannot differentiate players' actions via FiLM. The FiLM pressure + uniformity prevent collapse.

### Why Centroid-Pull (Not View-Splitting)

The original design randomly split possessions into two views, mean-pooled each, and compared them. This produced **1 gradient term per player** (one cosine comparison), with each embedding's gradient further diluted by 1/M from the mean-pooling. In practice, this was too weak: with 32 players per batch, VC's 32 gradient terms were overwhelmed by L_action's ~2,560 event-level terms. VC loss went *up* during training across three different batch configurations.

The centroid-pull produces **M gradient terms per player** (one per embedding), each with a direct, undiluted gradient. With K=32 players and M=8 possessions: 256 direct gradient terms — enough to meaningfully regularise alongside L_action.

### Algorithm

```
Input: dedup_embs  (N, d)   -- one h_player per (player, possession) pair
       dedup_pids  (N,)     -- corresponding player IDs

For each unique player_id with count >= 2:
    centroid = mean(dedup_embs where pid == player_id)     -> (d,)

    For each embedding emb_i of this player:
        loss_i = 1 - cosine_similarity(emb_i, centroid)

Return: mean(loss_i) over all eligible embeddings
```

### Gradient Flow

```
L_vc
  |
  v
1 - cos(emb_i, centroid)     [256 terms: 32 players x 8 embs each]
  |
  v
dedup_embs  (per-possession actor embeddings)
  |
  v
h_player  (GNN output for actor nodes)
  |
  v
GNN encoder  (GATv2 layers)
  |
  v
EventProjection, PlayerProjection  (input features)
```

Each embedding receives a direct gradient (no mean-pool dilution). The centroid is not detached, so the gradient also flows through the centroid computation — every embedding receives a small additional signal from other embeddings' comparisons with the centroid.

---

## 5. Complete Data Flow Pipeline

### Training Forward Pass

```
Batch of possession graphs (32 players x 8 possessions = 256, via PlayerAwareBatchSampler)
        |
        v
  [EventProjection]  126-D event features -> (E, 64)
  [PlayerProjection]  (team_flag, dx, dy) -> (P, 64)     [position ablated]
        |
        v
  [PossessionGNNEncoder]  2-layer HeteroGATv2
     edge types: next, prev, acts_in, performed_by,
                 context_for_tm, context_for_opp
        |
        v
  h_event (E, 64)          h_player (P, 64)
        |                       |
        |            [Gather actor nodes by player_id]
        |                       |
        |               dedup_embs (M, 64)    <-- NEW: exposed for L_vc
        |               dedup_pids (M,)       <-- NEW: exposed for L_vc
        |                       |
        |            [AttentionPooling: scatter-softmax per player]
        |                       |
        |                 z_p (K, 64)   unique pooled player embeddings
        |                       |
        |            [Broadcast z_p back to events]
        |                       |
  [FiLM: (1+gamma(z_p))*h_event + beta(z_p)]
        |
        v
  h_conditioned (E, 64)
        |
        +---> [ActionTypeHead]  -> (E, 14)   -> L_action (focal)
        +---> [AngleBinHead]    -> (E, 9)    -> L_action (focal)
        +---> [LengthBinHead]   -> (E, 5)    -> L_action (focal)
        |
  h_event (E, 64)
        +---> [OutcomeHead]     -> (E, 2)    -> L_outcome (BCE)

  z_p (K, 64) ---> L_uniformity  (Gaussian-potential, pushes z_p apart)

  dedup_embs, dedup_pids ---> L_view_consistency  (split, pool, cosine)
```

### Inference (Unchanged)

```
All possession graphs
        |
  [Same forward pass as training, no loss computation]
        |
  z_p per unique player -> save as player_embeddings.npy
        |
  [Cosine similarity search] -> top-K substitutes
```

---

## 6. Loss Term Reference

### L_action (ActionPredictionLoss) -- Primary

Focal loss over three discretised action targets, handling class imbalance (Pass/Carry dominate at ~70%):

- Action type: 14-class, inverse-sqrt class weights
- Angle bin: 9-class (8 directional sectors + no-angle)
- Length bin: 5-class, inverse-sqrt class weights

**Role**: Forces z_p to be behaviorally meaningful. Different players need different z_p to explain their different actions in the same situations. This is the primary learning signal.

### L_outcome (OutcomePredictionLoss) -- Secondary

Binary cross-entropy for possession-level shot/goal prediction. Operates on h_event (no z_p dependence), so it primarily shapes the GNN's situation understanding.

**Role**: Improves h_event quality for game-state representation.

### L_view_consistency (ViewConsistencyLoss) -- Auxiliary, NEW

`mean(1 - cosine_similarity(view_a, view_b))` over players with >= 2 possessions.

**Role**: Enforces that z_p is invariant to which subset of a player's possessions are observed. Directly targets the property needed for reliable substitute search: a player's embedding should not depend on which specific games or situations are sampled.

### L_uniformity (PooledUniformityLoss) -- Safety Net

Gaussian-potential uniformity from Wang & Isola (2020):
`L = log E[exp(-t * ||z_i - z_j||^2)]` over all pairs.

**Role**: Prevents embedding collapse. With t=4.0, strongly penalizes the closest pairs, ensuring spread on the hypersphere. Acts as a safety net in case FiLM pressure alone is insufficient.

---

## 7. Hyperparameters

### Fixed (inherited from pos_ablated_split_ctx)

| Parameter | Value | Source |
|-----------|-------|--------|
| `latent_dim` | 64 | ModelConfig |
| `hidden_dim` | 128 | ModelConfig |
| `num_heads` | 4 | ModelConfig |
| `num_layers` | 2 | ModelConfig |
| `dropout` | 0.1 | ModelConfig |
| `batch_size` | 256 | K=32 x M=8 via PlayerAwareBatchSampler |
| `players_per_batch` | 32 | 2x the original player_samp (16), maintains uniformity diversity |
| `possessions_per_player` | 8 | Enables 4-vs-4 view splits for meaningful mean-pooling |
| `learning_rate` | 1e-3 | TrainingConfig |
| `weight_decay` | 1e-5 | TrainingConfig |
| `patience` | 15 | TrainingConfig (early stopping on val_supervised) |
| `ablate_position` | True | ModelConfig |
| `split_context_edges` | True | GraphConfig |

### Loss Weights (new pipeline)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lambda_outcome` | 0.5 | Default, proven effective across all pipelines |
| `lambda_view_consistency` | 1.0 | Conservative starting point; acts as a regulariser alongside L_action |
| `lambda_pooled_contrast` (uniformity) | 1.0 | Same as pos_ablated_split_ctx; proven to maintain spread |
| `uniformity_t` | 4.0 | Same as pos_ablated_split_ctx; focuses gradient on closest pairs |

### Removed Hyperparameters

| Parameter | Old Value | Why Removed |
|-----------|-----------|-------------|
| `lambda_contrast` | 0.5 | Contrastive loss removed |
| `contrastive_temperature` | 0.05 | Contrastive loss removed |
| `lambda_alignment` | 0.3 | EMA alignment removed |
| `ema_momentum` | 0.999 | EMA memory bank removed |
| `temperature_start/end` | 0.10/0.02 | Annealing removed |
| `pooled_weight_start/end` | 0.10/0.50 | Annealing removed |

---

## 8. Reasoning: Why Each Decision

### Why drop contrastive instead of keeping it?

1. **Wrong abstraction level**: ContrastiveLoss operates on h_player (per-possession), but substitute search uses z_p (pooled). Pulling individual possession embeddings together doesn't guarantee the aggregate is stable.
2. **Evaluation evidence**: The `player_samp` pipeline (designed specifically to strengthen contrastive with player-aware batching) showed the "player-sampling paradox" -- tight within-context clusters but poor cross-player discrimination. Contrastive creates tight clusters, not meaningful differences.
3. **L_action already handles per-possession quality**: Through FiLM, L_action forces every h_player to carry player-specific signal for action prediction. This is a direct, task-relevant per-possession gradient. Contrastive's "cluster by player" is a less targeted proxy.
4. **Simplicity**: one fewer term = one fewer interaction surface = less fragility.

### Why drop EMA alignment?

View-consistency subsumes it. EMA alignment enforces `z_p(batch_t) ~ z_p(batch_{t-1})` via a momentum-updated memory bank. View-consistency enforces `z_p(subset_A) ~ z_p(subset_B)` within a single batch. The latter is:
- **Stronger**: operates on the live computation graph (gradients flow), not a detached target.
- **Simpler**: no external state (memory bank), no momentum hyperparameter.
- **More direct**: tests the actual property we want (subset-invariance of z_p).

### Why keep uniformity?

While FiLM provides a natural spreading force (identical z_p = high L_action), the uniformity loss provides a direct geometric guarantee. Previous experiments showed that without explicit uniformity, cosine similarities between all players compress into a narrow range (> 0.95), making top-K retrieval unreliable. The uniformity loss at t=4.0 with lambda=1.0 maintains healthy spread.

### Why extend from pos_ablated_split_ctx (not the EMA variant)?

- The EMA variant adds alignment loss that we're replacing with view-consistency. Building on it would mean removing EMA code and adding VC code -- more diff, more risk.
- `pos_ablated_split_ctx` is the clean base with the architectural features we want (position ablation + split context) and nothing we need to remove from the loss.
- Evaluation shows `pos_ablated_split_ctx` already achieves the best Spearman rho (0.974) and strong ground truth performance. We're adding view-consistency to address its one weakness: cross-context stability could be higher.

---

## 9. Expected Impact

Based on what view-consistency targets vs. current model weaknesses:

| Metric | Current (pos_ablated_split_ctx) | Expected Direction | Reasoning |
|--------|-------------------------------|-------------------|-----------|
| Self-consistency (competition-split) | 0.9702 | Higher | Directly trained for cross-view stability |
| Self-consistency (random-half) | 0.9790 | Higher | Same mechanism |
| Ground truth hit@10 | 22.2% (tier-1) | Stable or higher | Better z_p stability should improve retrieval |
| Spearman rho | 0.974 | Stable | L_action (primary signal) is unchanged |
| Substitute quality ratio | 1.59 | Stable or higher | Better z_p = better neighbor quality |
| Test macro-F1 | Baseline-level | Stable | Action heads unchanged |

### Risk

If FiLM + uniformity are insufficient to prevent collapse without contrastive, cosine similarities may compress. Mitigation: uniformity at t=4.0 / lambda=1.0 is kept as a safety net. If collapse still occurs, contrastive can be re-added as a lightweight 5th term.

---

## 10. Planned training command (not in the repo)

The following is **illustrative** for whoever implements this design. It is **not** wired up today.

```bash
# NOT AVAILABLE — flags below do not exist in main.py yet
python main.py --mode train \
    --tag view_consistency \
    --ablate_position \
    --split_context_edges \
    --view_consistency \
    --lambda_vc 0.5 \
    --uniformity_t 4.0 \
    --lambda_pooled 1.0
```

**Intended behaviour (spec):** `--view_consistency` would enable player-aware sampling (e.g. K=32, M=8) and disable contrastive loss and annealing.

---

## 11. Planned file changes (checklist — not applied)

| File | Planned change |
|------|----------------|
| `src/phase5_training/losses.py` | Add `ViewConsistencyLoss` class. Modify `CombinedLoss` to accept `lambda_view_consistency` and `view_consistency_loss` in forward. |
| `src/phase4_model/model.py` | Expose `dedup_actor_embs` and `dedup_actor_pids` in forward output dict. |
| `src/config.py` | Add `view_consistency: bool` and `lambda_view_consistency: float` to `TrainingConfig`. |
| `src/phase5_training/trainer.py` | Compute `ViewConsistencyLoss` from model outputs. Add history keys. When `view_consistency=True`, contrastive/alignment code paths remain inactive (lambdas=0). |
| `main.py` | Add pipeline registry entry. Add `--view_consistency` and `--lambda_vc` CLI args. Update `_configure_pipeline`. |

If implemented as specified, these changes would be additive: existing pipeline entries and checkpoints would remain valid without the new flag.
