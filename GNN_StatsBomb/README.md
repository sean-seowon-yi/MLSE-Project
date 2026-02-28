## Important notice

Current this is a copy of Sean code

Code remains for future references

# GNN StatsBomb Player Similarity

Player similarity from **StatsBomb 360** data: event stream plus **spatial context** (freeze frames) so that each event is encoded with where teammates and opponents were at that moment.

**Purpose**: Compare players by how they act in similar situations (action type, location, pressure, support). Phase 1 produces **event-level** feature vectors and metadata (including `player_id`); a later phase will aggregate events per player (e.g. mean embedding or histogram) to compute player similarity.

## Data choice: StatsBomb 360

This pipeline uses **StatsBomb 360** (not only the regular event stream):

- **Regular StatsBomb** gives per event: actor, action type, locations, outcomes, and context (pressure, xG, etc.), but **not** where other players were.
- **StatsBomb 360** adds, for selected events, a **freeze frame**: positions of other players (teammates, opponents, keeper) at that moment.

We use 360 so that player similarity can reflect **spatial context** (e.g. pressure, passing options, defensive density). Only matches that have a `three-sixty/{match_id}.json` file are loaded, and only events that have a 360 frame are kept. In the open dataset this is about **326 matches** and ~**1M frames**.


## Setup

```bash
cd GNN_StatsBomb
pip install -r requirements.txt
```

Ensure **StatsBomb** data lives at `../StatsBomb/data` (or set `statsbomb_base_path` in config), with:

- `events/{match_id}.json`
- `three-sixty/{match_id}.json` for matches you want to use
- `matches/`, `competitions.json`, etc.

## Usage

**Phase 1 — Prepare data and encode features (360 + event features):**

```bash
python main.py --mode prepare
```

With competition/season filter:

```bash
python main.py --mode prepare --competition 11 --season 1
```

Outputs (in `processed_data/`):

- `event_features.npy` — feature matrix including 360 spatial-context features
- `event_metadata.parquet` — event/match/player metadata
- `feature_names.json`, `data_stats.json`

## Configuration

In `src/config.py`:

- **`data.use_360`** (default `True`): Use 360 data; only matches with `three-sixty/{match_id}.json` and only events with a 360 frame are included.
- **`data.statsbomb_base_path`**: Path to StatsBomb `data` directory.
- **`data.three_sixty_dir`**: Subdirectory name for 360 files (default `"three-sixty"`).
- **`feature.mirror_sides`** (default `False`): If `True`, right-side positions are y-flipped and relabelled to left-side so flank roles are treated as equivalent; when `False`, Left Wing / Right Wing etc. stay distinct (preferred foot, tactical side).

Set **`use_360=False`** to run on the full event stream (all matches, no spatial context); then no 360 files are required and spatial-context features are zeroed.

## Feature vector

Each event is encoded as a fixed-length vector that includes:

- Event type, location, end location, deltas, play pattern, position, body part, outcomes, scalars (duration, pressure, xG, etc.), pitch zone.
- **StatsBomb 360 spatial context** (when `use_360=True` and a freeze frame exists): counts of visible teammates/opponents/keepers, mean teammate/opponent positions (normalised), and minimum distance to nearest teammate and opponent.

Coordinates are normalised to [0, 1]. **Left and right roles are kept distinct** (e.g. Left Wing vs Right Wing) so preferred foot and tactical side matter for similarity. Optional `feature.mirror_sides=True` in config can be set to mirror right-side positions to the left for legacy use. See `docs/DATA_QUALITY.md` for edge-case handling.

## Integrity & domain (player similarity)

The pipeline is designed for **player similarity in soccer**:

- **360 spatial context**: Freeze frames give pressure/support/density (teammate/opponent counts, mean positions, min distances) so similar situations are comparable across players.
- **Left/right roles distinct**: Left Wing and Right Wing (and other flank roles) are **not** collapsed—sidedness and preferred foot are preserved so similarity reflects real tactical differences.
- **Attack direction**: StatsBomb coordinates are already “attacking left→right”; we normalise to [0,1] without an extra half-time flip (handled in docs).
- **Event mix & positions**: On-ball event types, outcomes, play pattern, pitch zone (thirds × lanes), and named positions align with football notions for comparing players by behaviour and role.

## Project layout

```
GNN_StatsBomb/
├── main.py              # Entry point
├── requirements.txt
├── README.md            # This file
├── docs/
│   ├── DATA_QUALITY.md  # Data quality and edge cases
│   └── PHASE1.md        # Phase 1: data preparation & feature encoding
├── src/
│   ├── config.py
│   ├── data_preparation.py   # Load events + 360, filter to events with frame
│   └── feature_encoder.py    # Encode event + 360 spatial features
├── processed_data/     # Generated
├── embeddings/
└── checkpoints/
```
