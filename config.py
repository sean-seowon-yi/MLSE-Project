"""
config.py — Central configuration for PassLocationGNN.

All hyperparameters, feature dimensions, and pitch constants live here.
Import this module everywhere instead of scattering magic numbers.

360 data availability (StatsBomb open data, confirmed April 2026)
-----------------------------------------------------------------
Only competitions where `match_available_360` is non-null in
competitions.json actually have downloadable freeze frames.

  comp_id  season_id  competition             season      matches
  -------  ---------  ----------------------  ----------  -------
       9       281    1. Bundesliga           2023/2024    ~306
      11        90    La Liga                 2020/2021    ~380
      43       106    FIFA World Cup          2022          ~64
      55       282    UEFA Euro               2024          ~51

  Total across all four: ~800 matches, ~50-80k pass graphs.

To use a single competition, set COMPETITIONS to e.g. [(9, 281)].
To cap matches for quick experiments, set MAX_MATCHES_PER_COMP to an int.
"""

from typing import List, Optional, Tuple
import torch

# ── Pitch geometry ─────────────────────────────────────────────────────────────
PITCH_LEN: float = 120.0
PITCH_WID: float = 80.0
PITCH_DIAG: float = (PITCH_LEN**2 + PITCH_WID**2) ** 0.5
PITCH_AREA: float = PITCH_LEN * PITCH_WID

# ── Graph structure ────────────────────────────────────────────────────────────
N_CAM_PTS: int = 8  # boundary nodes resampled from visible_area polygon

NODE_FEATURE_NAMES: List[str] = [
    "x_norm",             # 0  player x / 120
    "y_norm",             # 1  player y / 80
    "is_actor",           # 2  1 = passer
    "is_teammate",        # 3  1 = same team as passer
    "is_keeper",          # 4  1 = goalkeeper
    "is_camera_boundary", # 5  1 = arc-length resampled visible-area node
]
EDGE_FEATURE_NAMES: List[str] = [
    "dist_norm",   # Euclidean distance / pitch diagonal
    "dx_norm",     # signed x displacement
    "dy_norm",     # signed y displacement
    "same_team",   # 1 if both nodes on the same side
    "angle_norm",  # bearing i→j / π
]
GLOBAL_FEATURE_NAMES: List[str] = [
    "visible_area_pct",  # fraction of pitch visible
    "vis_centroid_x",    # normalised centroid x
    "vis_centroid_y",    # normalised centroid y
    "vis_aspect_ratio",  # bounding-box width / height (normalised)
    "period_norm",       # period / 5
    "time_norm",         # elapsed seconds / 5400
    "goal_diff_norm",    # (passer_goals - opp_goals) clamped to [-3,3] / 3
    "is_drawing",
    "is_winning",
    "is_losing",
]
LABEL_NAMES: List[str] = ["x_norm", "y_norm"]

NODE_DIM:   int = len(NODE_FEATURE_NAMES)    # 6
EDGE_DIM:   int = len(EDGE_FEATURE_NAMES)    # 5
GLOBAL_DIM: int = len(GLOBAL_FEATURE_NAMES)  # 10
OUT_DIM:    int = len(LABEL_NAMES)           # 2

# ── Model architecture ─────────────────────────────────────────────────────────
EMBED_DIM: int = 16   # player identity embedding size
HIDDEN:    int = 64
HEADS:     int = 4
DROPOUT:   float = 0.1

# ── Loss ───────────────────────────────────────────────────────────────────────
DIR_LOSS_WEIGHT: float = 0.3  # λ for cosine direction term

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS:           int   = 100
BATCH_SIZE:       int   = 32
VAL_BATCH:        int   = 64
TEST_BATCH:       int   = 64
LR:               float = 1e-3
WEIGHT_DECAY:     float = 1e-4
GRAD_CLIP:        float = 1.0
TRAIN_SPLIT:      float = 0.8
VAL_SPLIT:        float = 0.1
RANDOM_SEED:      int   = 42

# ── Data loading — multi-competition ──────────────────────────────────────────
# All four confirmed-public 360 competitions. Comment out any you want to skip.
COMPETITIONS: List[Tuple[int, int]] = [
    (9,   281),   # 1. Bundesliga   2023/2024   ~306 matches
    # (11,   90),   # La Liga         2020/2021   ~380 matches
    # (7,   235),   # Ligue 1         2022/2023 
    # (7,   108),   # Ligue 1         2021/2022
    (43,  106),   # FIFA World Cup  2022         ~64 matches
    (55,  282),   # UEFA Euro       2024         ~51 matches
    (55,   43),   # UEFA Euro       2020
]

# Set to an int to cap how many matches are loaded per competition.
# Useful for quick local experiments before a full run.
# None = load every available match.
MAX_MATCHES_PER_COMP: Optional[int] = None

# ── Legacy single-competition aliases (backwards compat) ──────────────────────
COMP_ID:     int            = COMPETITIONS[0][0]
SEASON_ID:   int            = COMPETITIONS[0][1]
MAX_MATCHES: Optional[int]  = MAX_MATCHES_PER_COMP

# ── Cross-prediction (Section 17) ─────────────────────────────────────────────
N_SAMPLE:    int = 200
SAMPLE_SEED: int = 42

# ── Misc ───────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BG_COLOR: str = "#0d1117"  # dark background for plots
CHECKPOINT_PATH: str = "pass_location_gnn.pt"
