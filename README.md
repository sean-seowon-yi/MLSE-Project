# MLSE-Project — Similarity Matching across Soccer Players

**STA2453 Project: Player Similarity.**  
Football analytics combining **SkillCorner** tracking data and **StatsBomb** event data for player similarity and tactical analysis.

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
├── GNN_SkillCorner/           # GNN player similarity pipeline
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

### Data

- **SkillCorner**: tracking data (10 Hz), A-League matches. See `SkillCorner/README.md`. Add locally; not in repo.
- **StatsBomb**: event data (passes, shots, etc.). See `StatsBomb/README.md`. Add locally; not in repo.
- **Data overview**: `docs/data.md`.

## Requirements

- Python 3.10+
- For GNN pipeline: PyTorch, PyTorch Geometric, see `GNN_SkillCorner/requirements.txt`
