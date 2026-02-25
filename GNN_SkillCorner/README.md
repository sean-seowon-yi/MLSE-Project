# GNN_SkillCorner — Player Similarity from Tracking Data

### Purpose

`GNN_SkillCorner` learns **player embeddings from SkillCorner tracking data**.  
The goal is to represent each player with a dense vector (“Player DNA”) that captures:

- Typical **field zones** they occupy,
- **Movement patterns** (depth, width, dropping, runs),
- **Tactical relationships** (how they position relative to teammates, ball, and goals),

so that players with similar roles and behaviours are close in embedding space.

This pipeline is self-supervised: it learns purely from tracking data by reconstructing masked positions, not from external labels.

---

## Pipeline overview

The pipeline is organised into phases, all orchestrated by `main.py`:

1. **Data preparation**  
   - Load SkillCorner tracking data (10 Hz).  
   - Normalise coordinates with **team-aware rotation** so both teams attack towards +X (right), preserving left/right sidedness.  
   - Compute per-frame features per player: position, velocity, sprint flag, team indicator, distances to ball/own goal/opponent goal.

2. **Graph assembly**  
   - Convert each 22-player frame (or sampled time window) into a **graph snapshot**:  
     - Nodes: players.  
     - Node features: 9‑dim vector (position, velocity, team, tactical distances).  
     - Edges: fully connected or team-based, allowing GAT attention to learn who influences whom.

3. **GNN encoder (GATv2)**  
   - A 2‑layer GATv2 encoder with **skip connections** encodes each player node into a 32‑D embedding.  
   - Skip (linear) projections prevent over-smoothing and preserve per-player identity.

4. **Self‑supervised training (autoencoder)**  
   - Randomly **mask** one player’s position in a frame.  
   - GATv2 encoder produces embeddings; an MLP decoder predicts the masked position.  
   - Loss = **MSE(position_pred, position_true)** + variance regulariser.  
   - This forces the encoder to understand spatial + tactical relationships to reconstruct the missing player.

5. **Embedding aggregation & similarity**  
   - For each player across time, average their 32‑D embeddings over snapshots.  
   - Use cosine similarity on these player profiles to find nearest neighbours.  
   - Optional validation checks: do nearest neighbours share positions/roles?

---

## File structure

```text
GNN_SkillCorner/
├── main.py                 # CLI entry point for all phases
├── requirements.txt
├── README.md               # This file
├── docs/
│   └── PIPELINE_DETAILS.md # Detailed description of each phase
├── src/
│   ├── config.py           # Hyperparameters and paths
│   ├── data_preparation.py # Phase 1: load, normalise, sample frames
│   ├── graph_assembly.py   # Phase 2: build per-frame graphs
│   ├── dataset.py          # Dataset & masking logic
│   ├── model.py            # GATv2 encoder + decoder
│   ├── train.py            # Training loop
│   ├── inference.py        # Embedding generation & similarity search
│   └── utils.py            # Visualisation and helpers
├── checkpoints/            # Trained models (generated)
├── embeddings/             # Player embeddings & reports (generated)
└── processed_data/         # Prepared graphs (generated)
```

---

## Installation

From the project root:

```bash
cd GNN_SkillCorner
pip install -r requirements.txt
```

Install PyTorch Geometric as per the [official instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your platform and PyTorch version, then:

```bash
pip install torch-geometric
```

---

## Usage

### Full pipeline

Run everything (data prep → graphs → training → embeddings → basic validation):

```bash
cd GNN_SkillCorner
python main.py --mode all
```

### Individual modes

```bash
# Phase 1 & 2: Data preparation and graph assembly
python main.py --mode prepare

# Phases 3 & 4: Model training
python main.py --mode train --graphs-path ./processed_data/graphs.pkl

# Phase 5: Inference and similarity search
python main.py --mode inference --model-path ./checkpoints/best_model.pt

# Validation: check learned roles / clusters
python main.py --mode validate --profiles-path ./embeddings/player_profiles.json
```

---

## Configuration (high level)

Key options in `src/config.py`:

| Group          | Examples (defaults may differ)                                 |
|----------------|-----------------------------------------------------------------|
| Data           | `sampling_rate`, `sprint_threshold`, tracking input paths      |
| Model          | `latent_dim=32`, `num_heads=4`, GAT depth & dropout           |
| Training       | `batch_size=64`, `learning_rate=1e-3`, `num_epochs=100`       |

See `docs/PIPELINE_DETAILS.md` for the full list and rationale.

---

## Outputs

- **Embeddings**:  
  - Per‑snapshot embeddings (32‑D) for each player node.  
  - Aggregated per‑player profiles (mean over time) in `embeddings/`.

- **Similarity search**:  
  - Cosine nearest neighbours between players.  
  - Used to find players with similar movement and role profiles.

- **Visualisations** (via `utils.py` and docs examples):  
  - t-SNE/UMAP plots of player embeddings.  
  - Similarity matrices and heatmaps.  
  - Reconstruction error diagnostics.

---

## Notes & limitations

- This pipeline **only** uses SkillCorner tracking, not StatsBomb events.  
- It captures **movement/role similarity**, not full action decision-making.  
- For situation-aware action similarity (what a player does given a state), see the StatsBomb 360 pipeline in `GNN_StatsBomb/`.

