# MLSE-Project — Similarity Matching across Soccer Players

**STA2453 Project: Player Similarity.**  
Football analytics combining **SkillCorner** tracking data and **StatsBomb** event data for player similarity and tactical analysis.

## Repository structure

```
Project/
├── README.md                 # This file
├── .gitignore
│
├── docs/                     # Documentation
│   └── data.md               # SkillCorner & StatsBomb data specs
│
├── notebooks/                # Jupyter notebooks (EDA, reports)
│   ├── eda_data.ipynb
│   └── eda_report_4page.ipynb
│
├── reports/                  # Generated reports (HTML/PDF)
│   ├── eda_data_no_code.html
│   ├── eda_data_report.html
│   ├── eda_report_4page.html
│   ├── eda_data_no_code.pdf
│   ├── eda_report.pdf
│   └── eda_report_4page.pdf
│
├── scripts/                  # One-off / utility scripts
│   └── create_report.py
│
├── GNN_approach/             # GNN player similarity pipeline
│   ├── README.md
│   ├── PIPELINE_DETAILS.md
│   ├── main.py               # Entry point
│   ├── config.py
│   ├── data_preparation.py
│   ├── graph_assembly.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── utils.py
│   ├── checkpoints/          # Model checkpoints
│   ├── embeddings/            # Player profiles, similarity, plots
│   ├── processed_data/        # Prepared graphs
│   └── requirements.txt
│
├── assets/                  # Media (e.g. sample videos)
├── SkillCorner/             # SkillCorner open data (tracking)
└── StatsBomb/                # StatsBomb open data (events)
```

## Quick start

### GNN player similarity (SkillCorner tracking)

```bash
cd GNN_approach
pip install -r requirements.txt
python main.py --mode all --epochs 100
```

See `GNN_approach/README.md` and `GNN_approach/PIPELINE_DETAILS.md` for details.

### EDA notebooks

Run notebooks with the **working directory set to the project root** so paths to `SkillCorner/` and `StatsBomb/` resolve:

- Open `notebooks/eda_data.ipynb` or `notebooks/eda_report_4page.ipynb`
- In Jupyter: set kernel cwd to the project root (e.g. `Project/`)

### Scripts

Run from project root, e.g. `python scripts/create_report.py` (writes output into the current directory).

### Data

- **SkillCorner**: tracking data (10 Hz), A-League matches. See `SkillCorner/README.md`.
- **StatsBomb**: event data (passes, shots, etc.). See `StatsBomb/README.md`.
- **Data overview**: `docs/data.md`.

## Requirements

- Python 3.10+
- For GNN pipeline: PyTorch, PyTorch Geometric, see `GNN_approach/requirements.txt`
