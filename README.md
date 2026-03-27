# MLSE-Project — Player Similarity in Soccer

**STA2453 Project: Player Similarity.**  
This repository contains end-to-end pipelines to learn **player embeddings** and find **similar players** in soccer from:

- **SkillCorner** tracking data (10 Hz trajectories), and  
- **StatsBomb 360** event data (on-ball actions + freeze-frame spatial context).

We define:

> Two players are similar if, when placed in the **same game situation** (location, pressure, teammates/opponents around them), they tend to choose **similar actions**.

Both pipelines ultimately produce:

- A vector embedding `z_p` for each player, and  
- Tools to rank other players by similarity and interpret *why* they are similar (behavioural and situation-level analysis).

**Data not included.** You must obtain both datasets yourself and place them in the repo. **SkillCorner** tracking data: obtain and place under `SkillCorner/`. **StatsBomb** open data (events + 360): clone or download and place under `StatsBomb/`.

---

## Documentation (links)

| Document | Description |
|----------|-------------|
| [docs/data.md](docs/data.md) | SkillCorner & StatsBomb data overview (shared). |
| [progress_reports/](progress_reports/) | Project progress reports (milestone / periodic reports). |
| **GNN_SkillCorner** | |
| [GNN_SkillCorner/README.md](GNN_SkillCorner/README.md) | SkillCorner pipeline overview and quick start. |
| [GNN_SkillCorner/docs/PIPELINE_DETAILS.md](GNN_SkillCorner/docs/PIPELINE_DETAILS.md) | Phase-by-phase pipeline details (tracking → graphs → train → embeddings). |
| **GNN_StatsBomb** | |
| [GNN_StatsBomb/README.md](GNN_StatsBomb/README.md) | StatsBomb 360 pipeline overview, CLI summary, and quick start. |
| [GNN_StatsBomb/SYSTEM_DESIGN.md](GNN_StatsBomb/SYSTEM_DESIGN.md) | Full system design, rationale, losses, masking, config, audit trail. |
| [GNN_StatsBomb/docs/README.md](GNN_StatsBomb/docs/README.md) | Documentation index and links to all phase docs. |
| [GNN_StatsBomb/docs/PHASE1.md](GNN_StatsBomb/docs/PHASE1.md) | Data preparation & 126-D feature encoding. |
| [GNN_StatsBomb/docs/PHASE2.md](GNN_StatsBomb/docs/PHASE2.md) | Possession construction. |
| [GNN_StatsBomb/docs/PHASE3.md](GNN_StatsBomb/docs/PHASE3.md) | Heterogeneous graph construction. |
| [GNN_StatsBomb/docs/PHASE4.md](GNN_StatsBomb/docs/PHASE4.md) | Model architecture (GNN, FiLM, pooling, heads). |
| [GNN_StatsBomb/docs/PHASE5.md](GNN_StatsBomb/docs/PHASE5.md) | Training (losses, masking, evaluation). |
| [GNN_StatsBomb/docs/PHASE6.md](GNN_StatsBomb/docs/PHASE6.md) | Inference & similarity search. |
| [GNN_StatsBomb/docs/PHASE7.md](GNN_StatsBomb/docs/PHASE7.md) | Situation-level analysis. |
| [GNN_StatsBomb/docs/DATA_QUALITY.md](GNN_StatsBomb/docs/DATA_QUALITY.md) | Data quality, edge cases, clamping, missing 360. |
| [GNN_StatsBomb/docs/FUTURE_IMPROVEMENTS.md](GNN_StatsBomb/docs/FUTURE_IMPROVEMENTS.md) | SOTA assessment, critical vulnerabilities & blind spots, and improvement roadmap. |
| [GNN_StatsBomb/docs/PLAYER_SIMILARITY_FINAL_PLAN.md](GNN_StatsBomb/docs/PLAYER_SIMILARITY_FINAL_PLAN.md) | Original high-level plan and design notes. |
| [GNN_StatsBomb/docs/EVALUATION_RESULTS.md](GNN_StatsBomb/docs/EVALUATION_RESULTS.md) | Baseline and ablation evaluation results, policy diagnostics, and FIFA comparison. |

---

## Repository structure

Only configuration, core code, and design docs are tracked. Raw data, intermediate artifacts, and heavy reports are kept local and ignored via `.gitignore`.

```text
Project/
├── README.md                 # This file
├── .gitignore
│
├── docs/                     # Shared documentation
│   └── data.md               # SkillCorner & StatsBomb data overview
│
├── GNN_SkillCorner/          # GNN player similarity (SkillCorner tracking)
│   ├── README.md
│   ├── main.py               # CLI entry point
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
├── GNN_StatsBomb/            # GNN player similarity (StatsBomb 360 events)
│   ├── README.md
│   ├── SYSTEM_DESIGN.md      # Full system design & rationale
│   ├── main.py               # Multi-phase CLI (Phases 1–7 + evaluate)
│   ├── match_fifa_players.py    # FIFA-StatsBomb player matcher
│   ├── test_fifa_comparison.py  # FIFA stat comparison & visualizations
│   ├── requirements.txt
│   ├── docs/
│   │   ├── README.md         # Docs index & phase links
│   │   ├── PHASE1.md … PHASE7.md   # Per-phase documentation
│   │   ├── DATA_QUALITY.md   # Data quality & edge cases
│   │   └── PLAYER_SIMILARITY_FINAL_PLAN.md
│   ├── src/
│   │   ├── config.py
│   │   ├── data_preparation.py
│   │   ├── feature_encoder.py
│   │   ├── phase2_possession/    # Possession grouping
│   │   ├── phase3_graph/         # Hetero event+player graphs (with 360 context)
│   │   ├── phase4_model/         # Encoder + FiLM + pooling + heads
│   │   ├── phase5_training/      # Losses, dataset, trainer
│   │   ├── phase6_inference/     # z_p generation + similarity search
│   │   └── phase7_analysis/      # Situation-level counterfactual analysis
│   ├── checkpoints/{tag}/     # Model checkpoints per pipeline variant (generated)
│   ├── embeddings/{tag}/      # Player embeddings, reports, PCA plots per variant (generated)
│   ├── evaluations/         # Unified evaluation outputs (generated)
│   └── processed_data/       # Encoded events, possessions, graphs (generated)
│
├── progress_reports/        # Project progress reports (e.g. milestone PDFs)
├── assets/                   # Media (e.g. sample videos; not tracked)
├── FIFA_data/               # FIFA/EA Sports FC matched player data (local only; not tracked)
├── SkillCorner/              # SkillCorner data (local only; not tracked)
└── StatsBomb/                # StatsBomb data (local only; not tracked)
```

The **`progress_reports/`** folder holds project progress reports (e.g. milestone or periodic reports, often as PDFs). Add new reports here as the project advances.

---

## Quick start

### 1. GNN player similarity — SkillCorner tracking

This is the original pipeline built around tracking data.

```bash
cd GNN_SkillCorner
pip install -r requirements.txt

# Full pipeline (data prep → graphs → train → embeddings)
python main.py --mode all --epochs 100
```

See:

- `GNN_SkillCorner/README.md`  
- `GNN_SkillCorner/docs/PIPELINE_DETAILS.md`

for detailed phase descriptions and model internals.

---

### 2. GNN player similarity — StatsBomb 360 (event + spatial context)

This is the newer pipeline designed around **StatsBomb 360** and the strict definition:

> similar = "would act similarly if put in the same situation".

```bash
cd GNN_StatsBomb
pip install -r requirements.txt
```

**Run the full pipeline (Phases 1 → 6A):**

```bash
python main.py --mode full_pipeline
```

This will:

1. **prepare** — load events + 360, encode 126-D event features, save metadata & freeze frames  
2. **build_possessions** — group events into StatsBomb possessions  
3. **build_graphs** — build heterogeneous event+player graphs per possession  
4. **train** — train the GNN with masked imitation + auxiliary objectives  
5. **inference** — generate global player embeddings `z_p` (one per player)

Optional after training:

- **evaluate** — run the best checkpoint on the held-out test set; reports accuracy, macro F1, and outcome metrics, and saves confusion matrices and ROC curves to `checkpoints/{tag}/evaluation/`.

Once embeddings exist, you can:

**Search for similar players:**

```bash
python main.py --mode search --player_id <STATS_BOMB_PLAYER_ID>
```

Similarity search is **gender-aware**: male query players only retrieve male candidates, and female queries only female candidates.

**Run full evaluation for all pipeline variants:**

```bash
python main.py --mode full_eval_all --eval_output_dir ./evaluations
```

This evaluates all four pipeline variants (baseline, split-context, player-sampling, combined) and saves organized results under `evaluations/{pipeline_name}/`.

**Run situation-level analysis (Phase 7):**

```bash
python main.py --mode analyze
```

Phase 7:

- Picks query players and their nearest neighbours in embedding space.  
- Samples real game situations (events) from the query player.  
- For each situation, compares **predicted action distributions** (type, direction, length) of the query vs candidates, holding the state fixed.  
- Saves text reports and visualisations under `GNN_StatsBomb/embeddings/{tag}/analysis/`:  
  - **Bar charts** — action-type and direction (angle-bin) probabilities per situation.  
  - **PCA plots** — global embedding space in three variants (by position group, by subgroup, by full position), plus per-query neighbourhood views (query and top-k highlighted).

**FIFA stat comparison (validates similarity against FIFA player attributes):**

```bash
python main.py --mode fifa_comparison
```

For full design details (data, model, loss functions, masking, graph structure, assumptions), see:

- **`GNN_StatsBomb/SYSTEM_DESIGN.md`** — kept aligned with the current implementation.  
- **`GNN_StatsBomb/docs/README.md`** — index of per-phase docs (PHASE1–PHASE7) and related references.

---

## Data

### SkillCorner

- Tracking data at 10 Hz (player and ball positions).
- Add locally under `SkillCorner/` according to its own README (not tracked in git).
- Used exclusively by `GNN_SkillCorner`.

### StatsBomb & StatsBomb 360

- Open-data repository: (see `StatsBomb/README.md` for exact instructions).
- Expected layout under `StatsBomb/data/`:
  - `events/{match_id}.json`
  - `three-sixty/{match_id}.json` (for matches with 360 freeze frames)
  - `matches/`, `competitions.json`, etc.
- `GNN_StatsBomb` defaults to `../StatsBomb/data` (configurable via `src/config.py`).

`GNN_StatsBomb` uses **StatsBomb 360** by default:

- Regular event data provides event type, locations, outcomes, xG, pressure, etc.
- 360 adds a per-event **freeze frame**: positions of visible teammates, opponents, and keeper.  
  This is critical for defining the *situation* (pressure, options, density) behind each action.

### FIFA / EA Sports FC

- External player attribute data from EA Sports FC (FIFA) game series.
- Downloaded separately and placed under `FIFA_data/` (not tracked in git).
- `match_fifa_players.py` matches StatsBomb 360 players to FIFA data by name, country, and year.
- Used by `test_fifa_comparison.py` to validate GNN similarity search against FIFA player attributes.

### Shared data docs

- See `docs/data.md` for a high-level summary of all sources and how they relate.

---

## Requirements

- Python **3.10+**

Recommended environment workflow:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

Install dependencies for each pipeline:

- **SkillCorner GNN:**

  ```bash
  cd GNN_SkillCorner
  pip install -r requirements.txt
  ```

- **StatsBomb 360 GNN:**

  ```bash
  cd GNN_StatsBomb
  pip install -r requirements.txt
  ```

Each `requirements.txt` pins the necessary versions of:

- PyTorch, PyTorch Geometric, and friends  
- Numerical stack: `numpy`, `pandas`, `scikit-learn`  
- Visualisation and utilities: `matplotlib`, `seaborn`, `tqdm`, etc.

---

## High-level goals & design principles

Across both pipelines, the core goals are:

- **Situation-aware similarity**:  
  Player embeddings must encode *how* players act given context, not just where they are on the pitch.

- **Role and sidedness awareness**:  
  Positions like Left Wing and Right Wing are kept distinct in the StatsBomb pipeline to respect preferred foot and tactical side. Mirroring is only used as an optional analysis trick, not during training.

- **Strong masking to avoid label leakage** (StatsBomb 360):  
  For imitation learning, any feature that directly reveals the action or its outcome (event type, end location, outcomes, xG, etc.) is masked from the GNN input at training and inference time. Only pre-action state and spatial context remain, forcing the model to genuinely learn *decision-making*.

- **Graph-based reasoning** (StatsBomb 360):  
  Possessions are encoded as heterogeneous graphs with:
  - Event nodes (126-D features, with Spatial_360 zeroed for the GNN).  
  - Player nodes (role embedding + team flag + relative geometry).  
  - Temporal, actor, and context edges (including off-ball players from 360).
  This lets the GNN reason jointly about sequence, actors, and surrounding players.

- **Player-level aggregation & similarity**:  
  Per-possession player embeddings are pooled with **attention** to form global `z_p` vectors. Similarity search is then simple cosine similarity in this embedding space, with filters on position group and sample size for robustness.

- **Gender-aware evaluation**:
  All similarity search, ground truth, self-consistency, and policy diagnostic evaluations filter candidates by gender, ensuring male players are only compared to males and female players to females.

The **StatsBomb 360** pipeline is the most faithful implementation of the “same situation, same action” notion and is documented in detail in `GNN_StatsBomb/SYSTEM_DESIGN.md`. Use that file alongside this README when working on or extending the system.

