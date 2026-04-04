# Phase 5: Training

Phase 5 trains the model with **masked imitation** (predict action from state), **outcome** (shot/goal), **contrastive** (same player → similar h_player, hard negatives), and **pooled uniformity** (spread pooled z_p on the hypersphere) objectives. Masking is applied at runtime so the model never sees the action or player identity in the input event features.

---

## Goal

- Minimise combined loss over possession graphs.
- Produce checkpoints that Phase 6 and Phase 7 use for embeddings and counterfactual predictions.

---

## Masked imitation (runtime masking)

The **dataset** (and inference path) zero out parts of the event feature vector so the model only sees "situational state" and must predict the action. Masked ranges include:

- Event type (0–13) — prediction target.
- End location, delta, distance/angle (16–22).
- **Position** (32–57) — player identity; must flow through player node → h_player → z_p.
- Body part (58–64).
- All outcome fields (65–94).
- Action-specific scalars (95, 98–103): duration, pass length, etc.

**Left unmasked**: location (14–15), play pattern (23–31), under_pressure/counterpress (96–97), pitch zone (104–112), period (122–125). Spatial_360 is already zeroed in the stored graph (Phase 3).

---

## Loss function

```
L = L_action + λ_outcome · L_outcome + λ_contrast · L_contrastive + λ_pooled · L_pooled_uniformity [+ λ_alignment · L_alignment]
```

Default weights: λ_outcome = 0.5, λ_contrast = 0.5, λ_pooled = 0.3. The alignment term is optional (enabled via `ema_alignment`).

- **L_action**: Focal loss (γ=2.0) with class weights (1/√count, mean 1) on action type and length bin; angle bin uses Focal without extra weights. The fixed count tables live in `losses.py` (`_ACTION_TYPE_COUNTS`, `_LENGTH_BIN_COUNTS`); they should match `event_metadata.parquet` after Phase 1 — if you refresh the corpus, recompute counts from `event_type` / length-bin targets and update those literals (see `DATA_QUALITY.md`).
- **L_outcome**: BCE for shot/goal head.
- **L_contrastive**: InfoNCE on **actor-only** `h_player` with **same-position-group hard negatives**:
  - Positives: same `player_id` across different events/possessions in the batch.
  - Negatives: different `player_id`s in the **same coarse position group** (GK / Defender / Midfielder / Forward / Unknown), using the `POSITION_IDX_TO_GROUP` mapping. If a row has no same-group negatives, it falls back to all different-player pairs.
  - The softmax denominator includes **both** positive and negative terms (standard supervised contrastive / InfoNCE form), so the loss stays bounded for a fixed batch size.
  - Temperature τ = 0.05 (default).
- **L_pooled_uniformity**: Gaussian-potential uniformity loss on **pooled `z_p`** (the embeddings used for similarity search). Pushes all player embeddings apart on the unit hypersphere to widen cosine similarity gaps and prevent embedding collapse. The sensitivity parameter `uniformity_t` (default 2.0) controls how aggressively close pairs are penalized.
- **L_alignment** (optional): EMA-based cross-batch alignment loss on pooled `z_p`. Encourages a player's current-batch embedding to be consistent with a smoothed historical representation maintained in an EMA memory bank. See the EMA alignment section below.

Validation loss uses the **same** formula (including contrastive, pooled uniformity, and alignment when enabled) so the full multi-objective loss is tracked consistently across train and val.

---

## Training details

- **Optimiser**: Adam (lr=1e-3, weight_decay=1e-5).
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5) — monitors `val_supervised` (action + λ_outcome × outcome only), which is immune to annealing-induced changes in contrastive/uniformity weights.
- **Gradient clipping**: max_norm=1.0.
- **Batch size**: 96 possession graphs (default random shuffle).
- **Split**: 70/15/15 by **match_id** (seed=42, no match leakage).
- **Early stopping**: patience=15, min_delta=1e-4, on `val_supervised`.

### Dual checkpoint strategy

| Checkpoint | Tracks | Purpose |
|---|---|---|
| `best_model.pt` | Lowest `val_supervised` (action + λ_outcome × outcome) | Primary checkpoint for inference — optimises supervised accuracy |
| `best_total_model.pt` | Lowest `val_total` (full multi-objective loss) | Useful for embedding-space diagnostics where auxiliary losses matter |
| `checkpoint_epoch_{n}.pt` | Every N epochs (default 10) | Periodic snapshots for analysis |
| `final_model.pt` | End of training | Always saved regardless of early stopping |

All checkpoints store: model weights, optimizer state, scheduler state, best-loss tracking, patience counter, and full training history.

### Training history

A `training_history.json` is saved at the end of training with per-epoch values for: `train_total`, `train_action`, `train_outcome`, `train_contrastive`, `train_pooled_uniform`, `train_alignment` (when EMA enabled), `val_total`, `val_supervised`, `val_action`, `val_outcome`, `val_contrastive`, `val_pooled_uniform`, `val_alignment` (when EMA enabled), and `learning_rate`.

### Resume from checkpoint

Training can resume from any checkpoint via `--resume <filename>`:

```bash
python main.py --mode train --resume checkpoint_epoch_130.pt
```

Restores model weights, optimizer state, scheduler state, best-loss tracking, patience counter, and history. Epoch numbering continues from where it left off.

---

## Player-aware batch sampling (optional)

When `--player_sampling` is set, the standard random-shuffle DataLoader is replaced with `PlayerAwareBatchSampler`, which constructs each batch as **K players × M possessions**:

- **K** (`players_per_batch`, default 16): distinct players per batch.
- **M** (`possessions_per_player`, default 6): possessions sampled per player.

This guarantees that every batch contains multiple possessions per player, ensuring the InfoNCE contrastive loss always has positive pairs. Without this, random shuffling produces batches where most players appear only once, starving the contrastive loss of signal. When the player pool runs low mid-epoch, it is **extended** with a fresh shuffle (leftover players are not discarded).

When `--player_sampling` is active, the CLI also applies a tuned hyperparameter preset:

| Parameter | Default | Player-sampling override |
|---|---|---|
| `contrastive_temperature` | 0.05 | 0.15 |
| `lambda_contrast` | 0.5 | 0.15 |
| `lambda_pooled_contrast` | 0.3 | 0.5 |
| Temperature annealing | off | 0.15 → 0.02 (cosine) |
| Pooled weight annealing | off | 0.10 → 0.50 (cosine) |

---

## Schedule annealing (optional)

When enabled (via `--player_sampling` or manual config), the trainer applies **cosine annealing** per epoch:

- **Temperature**: from `temperature_start` to `temperature_end` — sharpens the contrastive loss over training.
- **Pooled uniformity weight**: from `pooled_weight_start` to `pooled_weight_end` — gradually increases uniformity pressure.

The `val_supervised` metric used for early stopping and LR scheduling is deliberately immune to these changes.

---

## EMA alignment loss (optional)

When `--ema_alignment` is set, a cross-batch self-consistency signal is added to the loss. This addresses a gap in the standard training: contrastive and uniformity losses operate within a single batch, but there is no explicit signal encouraging a player's `z_p` to be stable across batches (where different subsets of that player's possessions are sampled).

### Mechanism

1. An **EMA memory bank** (`EMAPlayerMemoryBank`) maintains a smoothed historical embedding for each player, updated with exponential moving average (momentum = 0.999 by default).
2. Before the loss computation, the bank is queried for each player in the batch:
   - If the player exists in the bank, the stored embedding is returned as `z_p_hist`.
   - A mask indicates which players have valid history.
3. **AlignmentLoss** computes `1 - cosine_similarity(z_p_current, z_p_hist)` averaged over players with valid history. This penalizes drift in a player's embedding across batches.
4. After the optimizer step, the bank is updated with the current batch's `z_p` embeddings (detached from the graph).
5. The bank state is saved in checkpoints for resume compatibility.

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `ema_alignment` | False | Enable/disable the EMA alignment loss |
| `lambda_alignment` | 0.3 | Weight of the alignment loss term |
| `ema_momentum` | 0.999 | Exponential moving average decay rate |

Higher `lambda_alignment` (e.g. 0.3) improves behavioral fidelity (Spearman rho) at the cost of supervised F1. Lower values (e.g. 0.1) recover F1 while retaining most of the alignment benefit.

---

## Named pipeline presets (PIPELINE_REGISTRY)

`main.py` defines a `PIPELINE_REGISTRY` of named pipeline configurations that can be invoked via `--pipeline <name>` or iterated via `full_eval_all`. Each entry specifies all architectural and training flags:

| Pipeline | split_ctx | player_samp | ablate_pos | t | λ_pooled | ema | λ_align | Extra |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `baseline` | No | No | No | 2.0 | 0.3 | No | - | — |
| `player_samp` | No | Yes | No | 2.0 | 0.5 | No | - | — |
| `split_ctx` | Yes | No | No | 2.0 | 0.3 | No | - | — |
| `split_ctx_ps` | Yes | Yes | No | 2.0 | 0.5 | No | - | — |
| `pos_ablated` | No | No | Yes | 2.0 | 0.3 | No | - | — |
| `pos_ablated_split_ctx` | Yes | No | Yes | 4.0 | 1.0 | No | - | — |
| `pos_ablated_split_ctx_v2` | Yes | No | Yes | 4.0 | 0.7 | No | - | — |
| `pos_ablated_split_ctx_ema` | Yes | No | Yes | 4.0 | 1.0 | Yes | 0.3 | — |
| `acts_in_dropout` | Yes | No | Yes | 4.0 | 1.0 | No | - | `acts_in_dropout=0.3` |
| `acts_in_dropout_pos` | Yes | No | Yes | 4.0 | 1.0 | No | - | p=0.3, `lambda_pos=0.3` |
| `acts_in_dropout_pos_gu` | Yes | No | Yes | 4.0 | 1.0 | No | - | p=0.3, `lambda_pos=0.3`, `uniformity_group_weight=3.0` |

The `_configure_pipeline()` function in `main.py` translates each registry entry into the appropriate config overrides. Cross-model metrics for trained checkpoints live in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) (including archived `pos_ablated_split_ctx_ema_v2` under `evaluations/`).

---

## Experiment tagging

Use `--tag` to namespace checkpoints for different experiments:

```bash
python main.py --mode train --tag split_ctx --split_context_edges
python main.py --mode train --player_sampling --tag player_samp
python main.py --mode train --split_context_edges --ablate_position --uniformity_t 4.0 --lambda_pooled 1.0 --tag pos_ablated_split_ctx
python main.py --mode train --split_context_edges --ablate_position --uniformity_t 4.0 --lambda_pooled 1.0 --ema_alignment --lambda_alignment 0.3 --tag pos_ablated_split_ctx_ema
python main.py --mode train --pipeline acts_in_dropout
```

Tags route checkpoints to `checkpoints/{tag}/` while sharing Phase 1–2 data. The `--split_context_edges` flag must match the graphs used.

---

## Test-set evaluation (`--mode evaluate`)

After training, run evaluation on the **held-out test set** (same split as training):

```bash
python main.py --mode evaluate
python main.py --mode evaluate --tag split_ctx --split_context_edges
```

This loads `checkpoints/{tag}/best_model.pt` (or `checkpoints/baseline/best_model.pt` when no `--tag` is given), runs the model on the test graphs, and computes:

| Output | Metrics | Visualization |
|--------|--------|----------------|
| Action type | Accuracy, macro F1 | Confusion matrix (14×14) |
| Angle bin | Accuracy, macro F1 | Confusion matrix (9×9) |
| Length bin | Accuracy, macro F1 | Confusion matrix (5×5) |
| Outcome (shot/goal) | Accuracy, BCE, AUC-ROC | ROC curves |

Standalone `evaluate` writes to **`evaluations/{tag}/test_metrics/`** by default: `test_metrics.json` and PNG plots. The same step also runs under `full_eval` inside `evaluations/{tag}/test_metrics/`.

---

## Code

| Component | Location |
|-----------|----------|
| Trainer loop, validation, checkpointing | `src/phase5_training/trainer.py` |
| Losses (Focal, contrastive, uniformity, combined) | `src/phase5_training/losses.py` |
| Dataset (masking, targets, match-level split) | `src/phase5_training/dataset.py` |
| Action targets (bins) | `src/phase5_training/action_targets.py` |
| Test-set evaluation (metrics + plots) | `src/phase5_training/evaluator.py` |
| Player-aware batch sampler | `src/phase5_training/sampler.py` |
| CLI entry | `main.py` → `--mode train`, `--mode evaluate` |

---

## How to run

```bash
cd GNN_StatsBomb

# Basic training (baseline)
python main.py --mode train

# With player-aware sampling
python main.py --mode train --player_sampling --tag player_samp

# With split context edges
python main.py --mode train --tag split_ctx --split_context_edges

# Position-ablated + split-ctx + strong uniformity
python main.py --mode train --tag pos_ablated_split_ctx --split_context_edges --ablate_position --uniformity_t 4.0 --lambda_pooled 1.0

# Position-ablated + split-ctx + EMA alignment
python main.py --mode train --tag pos_ablated_split_ctx_ema --split_context_edges --ablate_position --uniformity_t 4.0 --lambda_pooled 1.0 --ema_alignment --lambda_alignment 0.3

# Resume from checkpoint
python main.py --mode train --resume checkpoint_epoch_130.pt

# Evaluate on test set
python main.py --mode evaluate
```

Requires Phase 1–3 outputs (especially `possession_graphs.pkl` or `possession_graphs_{tag}.pkl`).

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 5 (masking rationale, loss formula, audit fixes).
- [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) — Cross-pipeline test and retrieval metrics.
- [PHASE4.md](PHASE4.md) — Model architecture.
- [PHASE6.md](PHASE6.md) — Inference (same masking, embedding generation).
