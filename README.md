# MLSE-Project — Similarity Matching across Soccer Players

**STA2453 Project: Player Similarity.**  
Football analytics combining **SkillCorner** tracking data and **StatsBomb** event data for player similarity and tactical analysis.

## Branches

Individual Contributor will create their own branch to work on the project separately. 

# Main Branch Basic Information

This is the global start that may or may not be included in individuals' branches.

## Repository structure

This repo contains the GNN pipeline, config, and data docs. Scripts, notebooks, and generated reports are **not** tracked (local use only).

```
Project/
├── README.md                 # This file
├── .gitignore
│
├── docs/                     # Documentation
│   └── data.md               # SkillCorner & StatsBomb data specs
│
├── GNN_SkillCorner/          # GNN player similarity pipeline (SkillCorner tracking)
│   ├── README.md
│   ├── main.py               # Entry point (run from here)
│   ├── requirements.txt
│   ├── docs/
│   │   └── PIPELINE_DETAILS.md
│   ├── src/                  # Core package
│   │   ├── config.py
│   │   ├── data_preparation.py
│   │   ├── graph_assembly.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── utils.py
│   ├── checkpoints/          # Model checkpoints (generated)
│   ├── embeddings/           # Player profiles, similarity, plots (generated)
│   └── processed_data/       # Prepared graphs (generated)
│
├── GNN_StatsBomb/            # Event + 360 pipeline (Phase 1: event features)
│   ├── README.md
│   ├── main.py               # Entry point (Phase 1 prepare mode)
│   ├── requirements.txt
│   ├── docs/
│   │   ├── DATA_QUALITY.md   # Data quality and edge cases
│   │   └── PHASE1.md         # Phase 1: data preparation & encoding
│   ├── src/
│   │   ├── config.py
│   │   ├── data_preparation.py
│   │   └── feature_encoder.py
│   └── processed_data/       # event_features.npy, event_metadata.parquet, etc.
│
├── assets/                   # Media (e.g. sample videos; not tracked)
├── SkillCorner/              # SkillCorner data (not tracked; add locally)
└── StatsBomb/                # StatsBomb data (not tracked; add locally)
```

## Quick start

### GNN player similarity (SkillCorner tracking)

```bash
cd GNN_SkillCorner
pip install -r requirements.txt
python main.py --mode all --epochs 100
```

See `GNN_SkillCorner/README.md` and `GNN_SkillCorner/docs/PIPELINE_DETAILS.md` for details.

### Event + 360 features (StatsBomb)

Phase 1 prepares per-event features (including StatsBomb 360 spatial context) for later player-similarity modelling.

```bash
cd GNN_StatsBomb
pip install -r requirements.txt
python main.py --mode prepare
```

See `GNN_StatsBomb/README.md`, `GNN_StatsBomb/docs/PHASE1.md`, and `GNN_StatsBomb/docs/DATA_QUALITY.md` for details.

### Data

- **SkillCorner**: tracking data (10 Hz), A-League matches. See `SkillCorner/README.md`. Add locally; not in repo.
- **StatsBomb**: event + 360 data (passes, shots, freeze frames). See `GNN_StatsBomb/README.md`. Raw data lives under `StatsBomb/data/` (add locally; not in repo).
- **Data overview**: `docs/data.md`.

## Requirements

- Python 3.10+
- For GNN pipeline: PyTorch, PyTorch Geometric, see `GNN_SkillCorner/requirements.txt`
