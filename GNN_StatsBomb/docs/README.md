# GNN_StatsBomb Documentation

This folder contains phase-by-phase documentation for the player similarity pipeline. The system learns **player trait embeddings** from StatsBomb 360 data so that similar players (by decision-making in the same situations) get similar embeddings.

---

## Pipeline overview

| Phase | Doc | Description |
|-------|-----|-------------|
| **1** | [PHASE1.md](PHASE1.md) | Data preparation & 126-D feature encoding |
| **2** | [PHASE2.md](PHASE2.md) | Possession construction (grouping, labels, timestamps) |
| **3** | [PHASE3.md](PHASE3.md) | Heterogeneous graph construction (event + player nodes, edges) |
| **4** | [PHASE4.md](PHASE4.md) | Model architecture (GNN, FiLM, pooling, heads, position ablation) |
| **5** | [PHASE5.md](PHASE5.md) | Training (masked imitation, contrastive, uniformity, EMA alignment, pipeline registry) |
| **6** | [PHASE6.md](PHASE6.md) | Inference & similarity search |
| **7** | [PHASE7.md](PHASE7.md) | Situation-level analysis (reports, counterfactual comparisons, possession animation) |

---

## Other docs

| Document | Description |
|----------|-------------|
| [DATA_QUALITY.md](DATA_QUALITY.md) | Edge cases, clamping, missing data, integrity notes for Phase 1 |
| [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) | Comprehensive cross-model evaluation (registered GNN pipelines, archived runs, and heuristics): ground truth, self-consistency, policy diagnostics, empirical behavioral, test metrics, FIFA comparison |
| [pseudo_ground_truth.md](pseudo_ground_truth.md) | Curated pseudo ground-truth pairs: who is paired with whom and why (no results) |
| [PLAYER_SIMILARITY_FINAL_PLAN.md](PLAYER_SIMILARITY_FINAL_PLAN.md) | High-level plan and design notes |
| [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | SOTA assessment and roadmap of possible improvements (model, training, data, evaluation, ops) |
| [models/VIEW_CONSISTENCY.md](models/VIEW_CONSISTENCY.md) | View-consistency training variant — **design proposal only; not implemented** in `src/` (not in `PIPELINE_REGISTRY`) |

---

## Single source of truth

The **system design** (indices, masking, losses, audit/fixes) is in the parent folder:

- **[../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md)** — problem statement, dataset, full pipeline, rationale, config, file structure

Phase docs summarize and point to `SYSTEM_DESIGN.md` where appropriate; they do not duplicate it.
