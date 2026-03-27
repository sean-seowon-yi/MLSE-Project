# Project Documentation

This folder contains high-level documentation shared across the project.  
Pipeline-specific docs live inside each GNN subfolder.

---

## Contents of this folder

| Document | Description |
|----------|-------------|
| [data.md](data.md) | **SkillCorner** and **StatsBomb** data overview: formats, directory layout, coordinate systems, key fields, and how each source is used in the GNN pipelines. |

**Data not in the repo.** SkillCorner and StatsBomb datasets must be obtained and placed under `SkillCorner/` and `StatsBomb/data/` respectively. See the root [README.md](../README.md) and [data.md](data.md) for details.

---

## Pipeline-specific docs

- **SkillCorner pipeline** (`GNN_SkillCorner/`)
  - [GNN_SkillCorner/docs/PIPELINE_DETAILS.md](../GNN_SkillCorner/docs/PIPELINE_DETAILS.md) — Phases from raw tracking to player embeddings; graph construction, GATv2 autoencoder, training, and validation.

- **StatsBomb 360 pipeline** (`GNN_StatsBomb/`)
  - [GNN_StatsBomb/SYSTEM_DESIGN.md](../GNN_StatsBomb/SYSTEM_DESIGN.md) — End-to-end system design: 126‑D encoding, possession & graph construction, model, losses, Phase 7 analysis.
  - [GNN_StatsBomb/docs/README.md](../GNN_StatsBomb/docs/README.md) — Index of all phase docs (PHASE1–PHASE7) and related references.
  - [GNN_StatsBomb/docs/DATA_QUALITY.md](../GNN_StatsBomb/docs/DATA_QUALITY.md) — Data quality, missing data, edge cases.
  - [GNN_StatsBomb/docs/FUTURE_IMPROVEMENTS.md](../GNN_StatsBomb/docs/FUTURE_IMPROVEMENTS.md) — SOTA assessment, critical vulnerabilities & blind spots, improvement roadmap.
  - [GNN_StatsBomb/docs/EVALUATION_RESULTS.md](../GNN_StatsBomb/docs/EVALUATION_RESULTS.md) — Baseline and ablation study results, policy diagnostics, and FIFA comparison.

Together with the root [README.md](../README.md), these give a complete picture of what the system does (player similarity by behaviour in context) and how each phase is implemented.

