# Phase 5: Training

Phase 5 trains the model with **masked imitation** (predict action from state), **outcome** (shot/goal), **contrastive** (same player → similar h_player, hard negatives), and **pooled uniformity** (spread pooled z_p on the hypersphere) objectives. Masking is applied at runtime so the model never sees the action or player identity in the input event features.

---

## Goal

- Minimise combined loss over possession graphs.
- Produce a checkpoint that Phase 6 and Phase 7 use for embeddings and counterfactual predictions.

---

## Masked imitation (runtime masking)

The **dataset** (and inference path) zero out parts of the event feature vector so the model only sees “situational state” and must predict the action. Masked ranges include:

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
L = L_action + 0.5 · L_outcome + 0.5 · L_contrastive + 0.3 · L_pooled_uniformity
```

- **L_action**: Focal loss (γ=2.0) with class weights (1/√count, mean 1) on action type and length bin; angle bin uses Focal without extra weights.
- **L_outcome**: BCE for shot/goal head; weight 0.5.
- **L_contrastive**: InfoNCE on **actor-only** `h_player` with **same-position-group hard negatives**:
  - Positives: same `player_id` across different events/possessions in the batch.
  - Negatives: different `player_id`s in the **same coarse position group** (GK / Defender / Midfielder / Forward / Unknown), using the `POSITION_IDX_TO_GROUP` mapping. If a row has no same-group negatives, it falls back to all different-player pairs.
  - Temperature τ=0.05. Weight 0.5.
- **L_pooled_uniformity**: Gaussian-potential uniformity loss on **pooled `z_p`** (the embeddings used for similarity search). Pushes all player embeddings apart on the unit hypersphere to widen cosine similarity gaps and prevent embedding collapse. Weight 0.3.

Validation loss uses the **same** formula (including contrastive and pooled uniformity) so early stopping and learning-rate scheduling are consistent with training.

---

## Training details

- Optimiser: Adam (lr=1e-3, weight_decay=1e-5).
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5).
- Gradient clipping: max_norm=1.0.
- Batch size: 96 possession graphs.
- Split: 70/15/15 by **match_id** (no match leakage).
- Early stopping: patience=15, min_delta=1e-4.

---

## Code

| Component | Location |
|-----------|----------|
| Trainer loop, validation | `src/phase5_training/trainer.py` |
| Losses (Focal, contrastive, combined) | `src/phase5_training/losses.py` |
| Dataset (masking, targets) | `src/phase5_training/dataset.py` |
| Action targets (bins) | `src/phase5_training/action_targets.py` |
| Test-set evaluation | `src/phase5_training/evaluator.py` |
| CLI entry | `main.py` → `--mode train`, `--mode evaluate` |

---

## How to run

```bash
cd GNN_StatsBomb
python main.py --mode train
```

Requires Phase 1–3 outputs (especially `possession_graphs.pkl`). Checkpoints go to `checkpoints/`.

---

## Test-set evaluation (`--mode evaluate`)

After training, run evaluation on the **held-out test set** (same split as training):

```bash
python main.py --mode evaluate
```

This loads `checkpoints/best_model.pt`, runs the model on the test graphs, and computes:

| Output | Metrics | Visualization |
|--------|--------|----------------|
| Action type | Accuracy, macro F1 | Confusion matrix (14×14) |
| Angle bin | Accuracy, macro F1 | Confusion matrix (9×9) |
| Length bin | Accuracy, macro F1 | Confusion matrix (5×5) |
| Outcome (shot/goal) | Accuracy, BCE, AUC-ROC | ROC curves |

Results are saved to `checkpoints/evaluation/`: `test_metrics.json` and the PNG plots. This gives a direct comparison of predicted vs actual actions on unseen matches. The checkpoint must match the current model (126-D features, time-delta edges); re-run `--mode train` if you have an older checkpoint.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 5 (masking rationale, loss formula, audit fixes).
- [PHASE4.md](PHASE4.md) — Model architecture.
- [PHASE6.md](PHASE6.md) — Inference (same masking, embedding generation).
