# GNN Player Similarity System

A Graph Neural Network approach to learning player similarity from tracking data, inspired by principles from the TacticAI paper. The system learns to represent players as dense embeddings ("Player DNA") that capture tactical roles, movement patterns, and spatial behavior—all without using role labels during training.

## Overview

This system implements a self-supervised learning approach using a Graph Autoencoder with GATv2 (Graph Attention Networks v2). The key insight is that by training a model to predict masked player positions from the context of other players, it must learn meaningful representations of tactical relationships.

### Key Features

- **Team-Aware Normalization**: Preserves left/right sidedness by rotating coordinates so all teams attack towards positive X
- **GATv2 Encoder**: Uses attention mechanisms to learn which player relationships matter
- **Self-Supervised Training**: No labeled data required—learns from position reconstruction
- **Role Validation**: Validates that learned embeddings cluster by tactical role

## Architecture

```
Input: 22-player graph snapshot
       ↓
┌─────────────────────────────────────┐
│  Node Features (per player):        │
│  - Position (x, y)                  │
│  - Velocity (vx, vy)                │
│  - Sprint flag                      │
│  - Team indicator                   │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  GATv2 Encoder                      │
│  - Layer 1: Perception (4 heads)    │
│  - Layer 2: Compression             │
│  → Output: 32-dim "Player DNA"      │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│  MLP Decoder (training only)        │
│  - Reconstructs masked position     │
└─────────────────────────────────────┘
```

## File Structure

```
GNN_approach/
├── config.py            # Configuration and hyperparameters
├── data_preparation.py  # Phase 1: Load, filter, normalize data
├── graph_assembly.py    # Phase 2: Build graph structures
├── model.py             # Phase 3: GATv2 Autoencoder architecture
├── dataset.py           # PyTorch Dataset with masking
├── train.py             # Phase 4: Training loop
├── inference.py         # Phase 5: Embedding generation & similarity
├── utils.py             # Visualization and helper functions
├── main.py              # Entry point orchestrating the pipeline
├── requirements.txt     # Dependencies
├── README.md            # This file
└── PIPELINE_DETAILS.md  # Detailed pipeline documentation
```

For in-depth explanations of each phase, see **[PIPELINE_DETAILS.md](PIPELINE_DETAILS.md)**.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) based on your PyTorch and CUDA versions:
```bash
pip install torch-geometric
```

## Usage

### Full Pipeline

Run the complete pipeline (data preparation → training → inference → validation):

```bash
python main.py --mode all
```

### Individual Phases

```bash
# Phase 1 & 2: Data Preparation and Graph Assembly
python main.py --mode prepare

# Phase 3 & 4: Model Training
python main.py --mode train --graphs-path ./processed_data/graphs.pkl

# Phase 5: Inference and Similarity Search
python main.py --mode inference --model-path ./checkpoints/best_model.pt

# Validation: Role Consistency Check
python main.py --mode validate --profiles-path ./embeddings/player_profiles.json
```

### Python API

```python
from GNN_approach import (
    get_config,
    SkillCornerDataLoader,
    GraphAssembler,
    PlayerSimilarityAutoencoder,
    Trainer,
    EmbeddingGenerator,
    SimilaritySearcher
)

# Load configuration
config = get_config()

# Process data
loader = SkillCornerDataLoader(config.data)
frames = loader.process_all_matches()

# Build graphs
assembler = GraphAssembler(config.graph)
graphs = assembler.frames_to_graphs(frames)

# Create and train model
model = PlayerSimilarityAutoencoder(config.model)
trainer = Trainer(model, config.training, train_loader, val_loader)
trainer.train()

# Generate embeddings and find similar players
generator = EmbeddingGenerator(model, device)
embeddings = generator.generate_embeddings(graphs)
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 10 | Sample every N frames (10 = 1 second) |
| `sprint_threshold` | 7.0 | Speed threshold for sprinting (m/s) |
| `latent_dim` | 32 | Embedding dimension ("Player DNA" size) |
| `num_heads` | 4 | Number of attention heads |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `num_epochs` | 100 | Maximum training epochs |

## Training Details

### Self-Supervised Task

The model is trained using a **position reconstruction task**:

1. Take a graph of 22 players
2. Randomly mask one player's position (set to 0,0)
3. Model predicts the masked player's true position
4. Loss = MSE(predicted, actual)

This forces the model to understand tactical relationships to successfully predict positions.

### Team-Aware Normalization

Critical for preserving left/right sidedness:

- All teams are rotated to attack towards **positive X** (right)
- This ensures "Left" is always **negative Y**, "Right" is always **positive Y**
- A Left Back will always be at similar coordinates regardless of which team

## Output

### Embeddings

- Each player gets a 32-dimensional embedding per 1-second snapshot
- Player profiles are computed by averaging all embeddings

### Similarity Search

- Cosine similarity between player profiles
- Returns ranked list of similar players
- Validation checks if similar players share tactical roles

### Visualizations

- Training loss curves
- Similarity matrix heatmap (sorted by role)
- t-SNE embedding visualization
- Per-role reconstruction error analysis

## Expected Results

When properly trained, you should see:

1. **Clustering by Role**: t-SNE visualization shows players clustering by position
2. **High Role Match Rate**: Top-5 similar players share the same role >60% of the time
3. **Position Group Accuracy**: Similar players are in the same position group (Defender, Midfielder, etc.) >80% of the time

## References

- TacticAI: An AI Assistant for Football Tactics (DeepMind, 2024)
- Graph Attention Networks v2 (Brody et al., 2021)
- SkillCorner Open Data

## Notes

- The system uses only SkillCorner tracking data (no StatsBomb event data)
- With 10 matches, expect ~50,000 graph snapshots for training
- Training on CPU is feasible but GPU is recommended for faster iteration
