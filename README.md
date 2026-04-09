# MLSE-Project — Similarity Matching across Soccer Players

**STA2453 Project: Player Similarity.**  
Football analytics combining **SkillCorner** tracking data and **StatsBomb** event data for player similarity and tactical analysis.

## Branches

Individual Contributor will create their own branch to work on the project separately. 

# Main Branch Basic Information

This is the global start that may or may not be included in individuals' branches.

This branch belongs to Aidan

# Environment

Please refer to `environment.yml` for python environment

# Quick start

Open `pass_location_player_sim.ipynb` and run top-to-bottom.


# Structure
`docs` documentation of data provided by Sean
`notebook` contains all work in progress notebook
`src` contains all codes required for models and algorithm


## layout

```
src/
├── config.py                    All hyperparameters and pitch constants
├── data/
│   └── loader.py                StatsBomb data loading, score lookup, vocab
├── graph/
│   ├── visible_area.py          Polygon parsing, global features, boundary resampling
│   └── builder.py               PyG Data construction
├── model/
│   └── gnn.py                   model architecture and implementation
├── training/
│   ├── loss.py                  loss functions and evaluation metrics
│   └── trainer.py               code to train models
├── analysis/
│   ├── errors.py                evaluate errors from the model
│   ├── player_similarity.py     compute EMD + MMD + Feature based similarity
│   └── plots.py                 All plot functions (each returns a Figure)
└── utils/
    ├── checkpoint.py            save and load models
    └── inference.py             predict pass location with models and player
```
