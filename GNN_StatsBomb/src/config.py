"""
Configuration for GNN StatsBomb Player Similarity System.

All paths, categorical vocabularies, and hyperparameters live here.
Vocabularies are derived from a 20-match survey of the StatsBomb open data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
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

PERIODS: List[str] = [
    "Period 1",
    "Period 2",
    "Extra Time 1",
    "Extra Time 2",
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

# Finer-grained subgroups (between 4 groups and 26 positions) for PCA and viz.
# Each position name maps to one subgroup; used for embeddings_pca_subgroup.png.
POSITION_SUBGROUPS: Dict[str, str] = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Center Back",
    "Left Center Back": "Center Back",
    "Right Center Back": "Center Back",
    "Left Back": "Full Back",
    "Right Back": "Full Back",
    "Left Wing Back": "Full Back",
    "Right Wing Back": "Full Back",
    "Center Defensive Midfield": "Defensive Mid",
    "Left Defensive Midfield": "Defensive Mid",
    "Right Defensive Midfield": "Defensive Mid",
    "Center Midfield": "Central Mid",
    "Center Attacking Midfield": "Central Mid",
    "Left Center Midfield": "Central Mid",
    "Right Center Midfield": "Central Mid",
    "Left Midfield": "Wide Mid",
    "Right Midfield": "Wide Mid",
    "Left Attacking Midfield": "Wide Mid",
    "Right Attacking Midfield": "Wide Mid",
    "Center Forward": "Forward",
    "Left Center Forward": "Forward",
    "Right Center Forward": "Forward",
    "Left Wing": "Forward",
    "Right Wing": "Forward",
    "Secondary Striker": "Forward",
    "Unknown": "Unknown",
}

# Map each position index to a coarse group index (for hard-negative contrastive loss).
# Groups: 0=Goalkeeper, 1=Defender, 2=Midfielder, 3=Forward, 4=Unknown
_POS_NAME_TO_GROUP_IDX: Dict[str, int] = {}
for _grp_name, _grp_idx in [("Goalkeeper", 0), ("Defender", 1), ("Midfielder", 2), ("Forward", 3)]:
    for _pos in POSITION_GROUPS[_grp_name]:
        _POS_NAME_TO_GROUP_IDX[_pos] = _grp_idx
POSITION_IDX_TO_GROUP: List[int] = [
    _POS_NAME_TO_GROUP_IDX.get(p, 4) for p in POSITIONS
]
NUM_POSITION_GROUPS: int = 5  # GK, Def, Mid, Fwd, Unknown

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

    # When True, 360 context edges are split into two relation types:
    #   ("player", "context_for_tm",  "event")  — teammate context
    #   ("player", "context_for_opp", "event")  — opponent context
    # This gives HeteroConv separate projection/attention weights for
    # offensive options vs defensive pressure.
    # Default False preserves backward compatibility with existing
    # checkpoints trained on a single "context_for" edge.  Enable via
    # --split_context_edges CLI flag for new experiments.
    split_context_edges: bool = False


@dataclass
class ModelConfig:
    """Phase 4: GNN encoder hyperparameters."""

    event_feature_dim: int = 126
    latent_dim: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    position_embed_dim: int = 16
    n_positions: int = len(POSITIONS)

    # When True, position information is ablated from the model:
    #   - PlayerProjection uses only continuous features (team_flag, dx, dy),
    #     ignoring position_idx entirely.
    #   - FiLM conditioning uses z_p alone (no position embedding channel).
    # Position still exists in the graph data (player.x[:, 0]) for
    # contrastive loss hard-negative mining.
    ablate_position: bool = False

    # Action prediction head
    n_action_types: int = len(EVENT_TYPES)   # 14
    n_angle_bins: int = 9   # 8 directional sectors + 1 "no-angle" bin
    n_length_bins: int = 5


@dataclass
class TrainingConfig:
    """Phase 5: training parameters."""

    batch_size: int = 96
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    lambda_outcome: float = 0.5
    lambda_contrast: float = 0.5
    lambda_pooled_contrast: float = 0.3
    uniformity_t: float = 2.0

    ema_alignment: bool = False
    lambda_alignment: float = 0.3
    ema_momentum: float = 0.999

    acts_in_dropout: float = 0.0

    lambda_pos: float = 0.0

    uniformity_group_weight: float = 1.0

    patience: int = 15
    min_delta: float = 1e-4

    checkpoint_dir: str = "./checkpoints/baseline"
    save_every_n_epochs: int = 10

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    contrastive_temperature: float = 0.05

    player_sampling: bool = False
    players_per_batch: int = 16
    possessions_per_player: int = 6

    anneal_temperature: bool = False
    temperature_start: float = 0.10
    temperature_end: float = 0.02

    anneal_pooled_weight: bool = False
    pooled_weight_start: float = 0.10
    pooled_weight_end: float = 0.50


@dataclass
class InferenceConfig:
    """Phase 6: inference and similarity search parameters."""

    embedding_output_dir: str = "./embeddings/baseline"
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

    tag: str = ""

    def __post_init__(self):
        os.makedirs(self.data.output_dir, exist_ok=True)

    def apply_tag(self, tag: str) -> None:
        """Namespace training / inference outputs under *tag*, leaving Phase 1-2 shared.

        Replaces the leaf directory (default ``baseline``) with *tag*:
          - Checkpoints  → ``checkpoints/{tag}/``
          - Embeddings   → ``embeddings/{tag}/``

        Possession graphs are **not** namespaced by tag: unified and split-context
        variants are shared across pipelines (see ``graphs_filename``).
        """
        if not tag:
            return
        self.tag = tag
        self.training.checkpoint_dir = str(
            Path(self.training.checkpoint_dir).parent / tag
        )
        self.inference.embedding_output_dir = str(
            Path(self.inference.embedding_output_dir).parent / tag
        )

    @property
    def graphs_filename(self) -> str:
        """Which graph pickle to load — keyed by topology, not training tag.

        ``baseline`` and ``player_samp`` share unified graphs; ``split_ctx`` and
        ``split_ctx_ps`` share split-context graphs (build once per variant).
        """
        if self.graph.split_context_edges:
            return "possession_graphs_split_ctx.pkl"
        return "possession_graphs.pkl"


def validate_graph_config_match(graphs, split_context_edges: bool) -> None:
    """Check that loaded graphs match the active split_context_edges setting.

    Raises ValueError on mismatch to prevent silent loss of context edges.
    """
    if not graphs:
        return
    sample_rels = {et[1] for et in graphs[0].edge_types}
    graph_is_split = "context_for_tm" in sample_rels or "context_for_opp" in sample_rels
    graph_is_unified = "context_for" in sample_rels

    if split_context_edges and not graph_is_split and graph_is_unified:
        raise ValueError(
            "Graph/config mismatch: --split_context_edges is set but the "
            "loaded graphs use a single 'context_for' edge.  Re-build "
            "graphs with --split_context_edges, or drop the flag."
        )
    if not split_context_edges and graph_is_split and not graph_is_unified:
        raise ValueError(
            "Graph/config mismatch: loaded graphs use split context edges "
            "(context_for_tm / context_for_opp) but --split_context_edges "
            "is not set.  Pass --split_context_edges to match these graphs."
        )


def get_config() -> Config:
    """Factory function to get default configuration."""
    return Config()
