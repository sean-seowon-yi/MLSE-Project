"""
Configuration for GNN StatsBomb Player Similarity System.

All paths, categorical vocabularies, and hyperparameters live here.
Vocabularies are derived from a 20-match survey of the StatsBomb open data.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


# ── Categorical vocabularies (from data survey) ──────────────────────────

EVENT_TYPES: List[str] = [
    "Ball Recovery",
    "Block",
    "Carry",
    "Clearance",
    "Dribble",
    "Duel",
    "Foul Committed",
    "Foul Won",
    "Goal Keeper",
    "Interception",
    "Miscontrol",
    "Pass",
    "Pressure",
    "Shot",
]

PLAY_PATTERNS: List[str] = [
    "From Corner",
    "From Counter",
    "From Free Kick",
    "From Goal Kick",
    "From Keeper",
    "From Kick Off",
    "From Throw In",
    "Other",
    "Regular Play",
]

BODY_PARTS: List[str] = [
    "Drop Kick",
    "Head",
    "Keeper Arm",
    "Left Foot",
    "No Touch",
    "Other",
    "Right Foot",
]

PASS_OUTCOMES: List[str] = [
    "Complete",
    "Incomplete",
    "Injury Clearance",
    "Out",
    "Pass Offside",
    "Unknown",
]

SHOT_OUTCOMES: List[str] = [
    "Blocked",
    "Goal",
    "Off T",
    "Post",
    "Saved",
    "Saved Off Target",
    "Saved to Post",
    "Wayward",
]

DRIBBLE_OUTCOMES: List[str] = [
    "Complete",
    "Incomplete",
]

PASS_TYPES: List[str] = [
    "Corner",
    "Free Kick",
    "Goal Kick",
    "Interception",
    "Kick Off",
    "Recovery",
    "Regular",
    "Throw-in",
]

SHOT_TYPES: List[str] = [
    "Free Kick",
    "Open Play",
    "Penalty",
]

PASS_HEIGHTS: List[str] = [
    "Ground Pass",
    "High Pass",
    "Low Pass",
]

POSITIONS: List[str] = [
    "Center Attacking Midfield",
    "Center Back",
    "Center Defensive Midfield",
    "Center Forward",
    "Center Midfield",
    "Goalkeeper",
    "Left Attacking Midfield",
    "Left Back",
    "Left Center Back",
    "Left Center Forward",
    "Left Center Midfield",
    "Left Defensive Midfield",
    "Left Midfield",
    "Left Wing",
    "Left Wing Back",
    "Right Attacking Midfield",
    "Right Back",
    "Right Center Back",
    "Right Center Forward",
    "Right Center Midfield",
    "Right Defensive Midfield",
    "Right Midfield",
    "Right Wing",
    "Right Wing Back",
    "Secondary Striker",  # StatsBomb variant; map to forward
    "Unknown",            # Missing / non-player events
]

POSITION_GROUPS = {
    "Goalkeeper": ["Goalkeeper"],
    "Defender": [
        "Center Back",
        "Left Back",
        "Left Center Back",
        "Left Wing Back",
        "Right Back",
        "Right Center Back",
        "Right Wing Back",
    ],
    "Midfielder": [
        "Center Attacking Midfield",
        "Center Defensive Midfield",
        "Center Midfield",
        "Left Attacking Midfield",
        "Left Center Midfield",
        "Left Defensive Midfield",
        "Left Midfield",
        "Right Attacking Midfield",
        "Right Center Midfield",
        "Right Defensive Midfield",
        "Right Midfield",
    ],
    "Forward": [
        "Center Forward",
        "Left Center Forward",
        "Left Wing",
        "Right Center Forward",
        "Right Wing",
        "Secondary Striker",
    ],
}

# StatsBomb pitch dimensions (standardised)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


# ── Dataclass configs ────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Paths and data-loading parameters."""

    statsbomb_base_path: str = "../StatsBomb/data"
    output_dir: str = "./processed_data"
    three_sixty_dir: str = "three-sixty"  # subdir under statsbomb_base_path

    # Use StatsBomb 360 data (spatial context: freeze frames). When True, only matches
    # that have a three-sixty/{match_id}.json file are loaded, and only events that
    # have a 360 frame are kept.
    use_360: bool = True

    # Which event types to keep (on-ball actions with spatial data)
    event_types: List[str] = field(default_factory=lambda: list(EVENT_TYPES))

    # Competition / season filter (None = load all)
    competition_ids: Optional[List[int]] = None
    season_ids: Optional[List[int]] = None

    random_seed: int = 42


@dataclass
class FeatureConfig:
    """Feature-encoding parameters."""

    # Pitch
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH

    # Pitch zone grid (for one-hot zone features)
    x_zones: int = 3   # defensive / middle / attacking third
    y_zones: int = 3   # left / center / right lane

    # 360 spatial context: max distance (pitch units) for normalising min_dist features
    max_dist_for_norm: float = 50.0

    # Left/right mirroring: if True, right-side positions are y-flipped and relabelled to
    # left-side so flank roles are treated as equivalent. Default False: Left Wing and
    # Right Wing (and other flank roles) are kept distinct (preferred foot, tactical side).
    mirror_sides: bool = False


@dataclass
class PossessionConfig:
    """Phase 2: possession construction parameters."""

    min_events: int = 2
    max_events: int = 200

@dataclass
class GraphConfig:
    """Phase 3: graph construction parameters."""

    latent_dim: int = 64
    position_embed_dim: int = 16

    use_reverse_temporal: bool = True
    # Relation triplets enabled by default:
    #   ("event","next","event"), ("event","prev","event"),
    #   ("player","acts_in","event"), ("event","performed_by","player"),
    #   ("player","context_for","event")


@dataclass
class ModelConfig:
    """Phase 4: GNN encoder hyperparameters."""

    latent_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    position_embed_dim: int = 16
    n_positions: int = len(POSITIONS)

    # Action prediction head
    n_action_types: int = len(EVENT_TYPES)   # 14
    n_angle_bins: int = 9   # 8 directional sectors + 1 "no-angle" bin
    n_length_bins: int = 5


@dataclass
class TrainingConfig:
    """Phase 5: training parameters."""

    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    lambda_outcome: float = 0.5
    lambda_contrast: float = 1.0

    patience: int = 15
    min_delta: float = 1e-4

    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 10

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class InferenceConfig:
    """Phase 6: inference and similarity search parameters."""

    embedding_output_dir: str = "./embeddings"
    min_samples_per_player: int = 50
    top_k: int = 10
    similarity_metric: str = "cosine"


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        os.makedirs(self.data.output_dir, exist_ok=True)


def get_config() -> Config:
    """Factory function to get default configuration."""
    return Config()
