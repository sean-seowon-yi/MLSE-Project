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
L = L_action + λ_outcome · L_outcome + λ_contrast · L_contrastive + λ_pooled · L_pooled_uniformity
```

Default weights: λ_outcome = 0.5, λ_contrast = 0.5, λ_pooled = 0.3.

- **L_action**: Focal loss (γ=2.0) with class weights (1/√count, mean 1) on action type and length bin; angle bin uses Focal without extra weights.
- **L_outcome**: BCE for shot/goal head.
- **L_contrastive**: InfoNCE on **actor-only** `h_player` with **same-position-group hard negatives**:
  - Positives: same `player_id` across different events/possessions in the batch.
  - Negatives: different `player_id`s in the **same coarse position group** (GK / Defender / Midfielder / Forward / Unknown), using the `POSITION_IDX_TO_GROUP` mapping. If a row has no same-group negatives, it falls back to all different-player pairs.
  - Temperature τ = 0.05 (default).
- **L_pooled_uniformity**: Gaussian-potential uniformity loss on **pooled `z_p`** (the embeddings used for similarity search). Pushes all player embeddings apart on the unit hypersphere to widen cosine similarity gaps and prevent embedding collapse.

Validation loss uses the **same** formula (including contrastive and pooled uniformity) so the full multi-objective loss is tracked consistently across train and val.

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

A `training_history.json` is saved at the end of training with per-epoch values for: `train_total`, `train_action`, `train_outcome`, `train_contrastive`, `train_pooled_uniform`, `val_total`, `val_supervised`, `val_action`, `val_outcome`, `val_contrastive`, `val_pooled_uniform`, and `learning_rate`.

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

This guarantees that every batch contains multiple possessions per player, ensuring the InfoNCE contrastive loss always has positive pairs. Without this, random shuffling produces batches where most players appear only once, starving the contrastive loss of signal.

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

## Experiment tagging

Use `--tag` to namespace checkpoints for different experiments:

```bash
python main.py --mode train --tag split_ctx --split_context_edges
python main.py --mode train --player_sampling --tag player_samp
python main.py --mode train --split_context_edges --player_sampling --tag split_ctx_ps
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

Results are saved to `checkpoints/{tag}/evaluation/`: `test_metrics.json` and PNG plots.

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

# Basic training
python main.py --mode train

# With player-aware sampling
python main.py --mode train --player_sampling --tag player_samp

# With split context edges
python main.py --mode train --tag split_ctx --split_context_edges

# Resume from checkpoint
python main.py --mode train --resume checkpoint_epoch_130.pt

# Evaluate on test set
python main.py --mode evaluate
```

Requires Phase 1–3 outputs (especially `possession_graphs.pkl` or `possession_graphs_{tag}.pkl`).

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 5 (masking rationale, loss formula, audit fixes).
- [PHASE4.md](PHASE4.md) — Model architecture.
- [PHASE6.md](PHASE6.md) — Inference (same masking, embedding generation).
