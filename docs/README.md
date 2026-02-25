# Project Documentation

This folder contains high-level documentation shared across the project.  
More detailed, pipeline-specific docs live inside each GNN subfolder.

---

## Top-level docs

- `data.md`  
  Overview of:
  - **SkillCorner** tracking data (formats, coordinate system, sampling).  
  - **StatsBomb** & **StatsBomb 360** event data (JSON layout, key fields).  
  - How these sources are used in the two GNN pipelines.

---

## Pipeline-specific docs

For detailed, up-to-date design documents, see:

- **SkillCorner pipeline** (`GNN_SkillCorner/`)
  - `GNN_SkillCorner/docs/PIPELINE_DETAILS.md`  
    - Phases from raw tracking to player embeddings.  
    - Graph construction, GATv2 autoencoder, training regime, and validation.

- **StatsBomb 360 pipeline** (`GNN_StatsBomb/`)
  - `GNN_StatsBomb/SYSTEM_DESIGN.md`  
    - End-to-end system design for situation-aware player similarity.  
    - Dataset choices, 122‑D encoding, possession & graph construction, model, losses, and Phase 7 analysis.  
  - `GNN_StatsBomb/docs/DATA_QUALITY.md`  
    - Data quality checks, missing data handling, edge cases, and soccer-specific assumptions.  
  - `GNN_StatsBomb/docs/PHASE1.md`  
    - Detailed description of Phase 1 encoding (feature groups, indices, normalisation, use of 360).

These documents, together with the root `README.md`, give a complete picture of:

- What the system is trying to achieve (player similarity by behaviour in context), and  
- How each phase in each pipeline is implemented to support that goal.

