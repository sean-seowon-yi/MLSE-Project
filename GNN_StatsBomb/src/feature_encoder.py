"""
Phase 1-B: Feature Encoder — fixed-length numeric vector per event.

Coordinate normalisation
────────────────────────
StatsBomb coordinates are **already** expressed so the acting team attacks
left → right (opponent goal at x = 120).  Raw values are normalised to [0, 1]:

    x_norm = x / 120
    y_norm = y / 80

Left and right roles (e.g. Left Wing vs Right Wing) are **not** mirrored:
they are kept distinct so that preferred foot, tactical side, and inverted
roles are reflected in the features.  Optional ``mirror=True`` can be used
to flip right-side positions to the left and relabel for legacy comparisons.

Feature groups
──────────────
 #  Group             Dims   Description
 ─  ─────             ────   ───────────
 1  event_type        14     One-hot
 2  location           2     (x, y) normalised [0, 1]
 3  end_location       3     (end_x, end_y, has_end) — 0 if absent
 4  delta              2     (dx, dy) = end − start (0 if no end)
 5  distance & angle   2     Euclidean distance, signed angle of delta
 6  play_pattern       9     One-hot
 7  position          26     One-hot (sided: Left/Right Wing etc. distinct unless mirror=True)
 8  body_part          7     One-hot
 9  pass_outcome       6     One-hot (zeros for non-pass)
10  shot_outcome       8     One-hot (zeros for non-shot)
11  dribble_outcome    2     One-hot (zeros for non-dribble)
12  pass_type          8     One-hot
13  pass_height        3     One-hot
14  shot_type          3     One-hot
15  scalars            9     duration, under_pressure, counterpress,
                             pass_length, pass_angle, pass_switch,
                             pass_cross, shot_xg, shot_first_time
16  pitch_zone         9     3×3 grid one-hot (thirds × lanes)
17  spatial_360        9     StatsBomb 360: teammate/opponent counts & positions, min dists
                      ───
              Total   122
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    FeatureConfig,
    get_config,
    EVENT_TYPES,
    PLAY_PATTERNS,
    BODY_PARTS,
    PASS_OUTCOMES,
    SHOT_OUTCOMES,
    DRIBBLE_OUTCOMES,
    PASS_TYPES,
    PASS_HEIGHTS,
    SHOT_TYPES,
    POSITIONS,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

# ── Position mirroring map ───────────────────────────────────────────────
# Maps each "Right …" position to its "Left …" mirror so that after
# y-flip both are labelled identically.

_MIRROR_POSITION: Dict[str, str] = {
    "Right Attacking Midfield": "Left Attacking Midfield",
    "Right Back": "Left Back",
    "Right Center Back": "Left Center Back",
    "Right Center Forward": "Left Center Forward",
    "Right Center Midfield": "Left Center Midfield",
    "Right Defensive Midfield": "Left Defensive Midfield",
    "Right Midfield": "Left Midfield",
    "Right Wing": "Left Wing",
    "Right Wing Back": "Left Wing Back",
}

# Set of all "right-side" positions that trigger the y-flip
_RIGHT_POSITIONS = set(_MIRROR_POSITION.keys())


class EventFeatureEncoder:
    """
    Converts a DataFrame of ``ParsedEvent`` rows into a numeric matrix.

    Parameters
    ----------
    config : FeatureConfig
        Pitch dimensions and zone grid settings.
    mirror : bool
        If *True*, right-side positions are y-flipped and relabelled to
        their left-side counterparts (legacy option).  Default *False*:
        Left Wing, Right Wing, and other flank roles stay distinct for
        preferred foot and tactical side.
    """

    def __init__(self, config: FeatureConfig, mirror: bool = False):
        self.config = config
        self.mirror = mirror

        # Pre-build vocabulary → index maps for one-hot encoding
        self._vocab = {
            "event_type":      {v: i for i, v in enumerate(EVENT_TYPES)},
            "play_pattern":    {v: i for i, v in enumerate(PLAY_PATTERNS)},
            "body_part":       {v: i for i, v in enumerate(BODY_PARTS)},
            "pass_outcome":    {v: i for i, v in enumerate(PASS_OUTCOMES)},
            "shot_outcome":    {v: i for i, v in enumerate(SHOT_OUTCOMES)},
            "dribble_outcome": {v: i for i, v in enumerate(DRIBBLE_OUTCOMES)},
            "pass_type":       {v: i for i, v in enumerate(PASS_TYPES)},
            "pass_height":     {v: i for i, v in enumerate(PASS_HEIGHTS)},
            "shot_type":       {v: i for i, v in enumerate(SHOT_TYPES)},
            "position":        {v: i for i, v in enumerate(POSITIONS)},
        }

        # Feature-group sizes (for slicing / inspection)
        self.group_sizes = {
            "event_type":      len(EVENT_TYPES),       # 14
            "location":        2,
            "end_location":    3,                       # x, y, has_end
            "delta":           2,
            "dist_angle":      2,
            "play_pattern":    len(PLAY_PATTERNS),      #  9
            "position":        len(POSITIONS),          # 26
            "body_part":       len(BODY_PARTS),         #  7
            "pass_outcome":    len(PASS_OUTCOMES),      #  6
            "shot_outcome":    len(SHOT_OUTCOMES),      #  8
            "dribble_outcome": len(DRIBBLE_OUTCOMES),   #  2
            "pass_type":       len(PASS_TYPES),         #  8
            "pass_height":     len(PASS_HEIGHTS),       #  3
            "shot_type":       len(SHOT_TYPES),         #  3
            "scalars":         9,
            "pitch_zone":      self.config.x_zones * self.config.y_zones,  # 9
            "spatial_360":     9,   # StatsBomb 360: counts + mean positions + min dists
        }
        self.feature_dim = sum(self.group_sizes.values())

    # ── public API ───────────────────────────────────────────────────

    def encode_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Encode every row of *df* (from ``StatsBombDataLoader``) into a
        fixed-length numeric feature vector.

        Returns
        -------
        features : ndarray, shape (n_events, feature_dim)
        metadata : DataFrame
            Copy of identifying columns (ids, player, team, match, …)
            aligned row-for-row with *features*.
        """
        n = len(df)
        X = np.zeros((n, self.feature_dim), dtype=np.float32)

        for i in range(n):
            row = df.iloc[i]
            X[i] = self._encode_row(row)

        meta_cols = [
            "event_id", "match_id", "competition_id", "season_id",
            "player_id", "player_name", "team_id", "team_name",
            "position_name", "event_type", "period", "minute", "second",
            "possession_number", "possession_team_id", "possession_team_name",
            "shot_outcome", "shot_xg",
        ]
        meta = df[[c for c in meta_cols if c in df.columns]].copy().reset_index(drop=True)

        return X, meta

    def get_feature_names(self) -> List[str]:
        """Human-readable name for every column in the feature matrix."""
        names: List[str] = []
        for group, size in self.group_sizes.items():
            if size == 1:
                names.append(group)
            else:
                for j in range(size):
                    names.append(f"{group}_{j}")
        return names

    # ── internals ────────────────────────────────────────────────────

    def _encode_row(self, row: pd.Series) -> np.ndarray:
        """Build the full feature vector for one event row."""

        # -- Position (for mirror check; resolve Unknown early) -----------
        _pos_raw = row.get("position_name")
        if _pos_raw is None or _is_nan(_pos_raw) or not isinstance(_pos_raw, str):
            _pos_raw = "Unknown"
        needs_mirror = self.mirror and (_pos_raw in _RIGHT_POSITIONS)

        # -- Spatial values (apply mirror before encoding) ----------------
        loc_x = row["location_x"]
        loc_y = row["location_y"]
        end_x = row.get("end_location_x")
        end_y = row.get("end_location_y")
        carry_end_x = row.get("carry_end_x")
        carry_end_y = row.get("carry_end_y")

        if needs_mirror:
            loc_y = PITCH_WIDTH - loc_y
            if end_y is not None and not _is_nan(end_y):
                end_y = PITCH_WIDTH - end_y
            if carry_end_y is not None and not _is_nan(carry_end_y):
                carry_end_y = PITCH_WIDTH - carry_end_y

        # Normalise to [0, 1] and clip (handles OOB from raw data)
        nx = max(0.0, min(1.0, loc_x / PITCH_LENGTH))
        ny = max(0.0, min(1.0, loc_y / PITCH_WIDTH))
        has_end = 0.0
        nex, ney = 0.0, 0.0

        if end_x is not None and not _is_nan(end_x):
            nex = max(0.0, min(1.0, end_x / PITCH_LENGTH))
            ney = max(0.0, min(1.0, (end_y if (end_y is not None and not _is_nan(end_y)) else 0.0) / PITCH_WIDTH))
            has_end = 1.0
        elif carry_end_x is not None and not _is_nan(carry_end_x):
            nex = max(0.0, min(1.0, carry_end_x / PITCH_LENGTH))
            ney = max(0.0, min(1.0, (carry_end_y if (carry_end_y is not None and not _is_nan(carry_end_y)) else 0.0) / PITCH_WIDTH))
            has_end = 1.0

        dx = nex - nx if has_end else 0.0
        dy = ney - ny if has_end else 0.0
        dist = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx) if has_end else 0.0

        # -- Position (mirror label if needed) ----------------------------
        pos_name = _pos_raw
        if needs_mirror:
            pos_name = _MIRROR_POSITION.get(pos_name, pos_name)

        # -- Body-part: combine pass and shot body parts ------------------
        body_part = None
        etype = row.get("event_type") or "Pass"
        if etype == "Pass":
            body_part = row.get("pass_body_part")
        elif etype == "Shot":
            body_part = row.get("shot_body_part")
        if body_part is None or _is_nan(body_part):
            body_part = None

        # Mirror body part: swap Left/Right Foot when position is mirrored
        if needs_mirror and body_part in ("Left Foot", "Right Foot"):
            body_part = "Right Foot" if body_part == "Left Foot" else "Left Foot"

        # -- Scalars -------------------------------------------------------
        duration = max(float(row.get("duration", 0.0) or 0.0), 0.0)
        under_pressure = 1.0 if row.get("under_pressure") else 0.0
        counterpress = 1.0 if row.get("counterpress") else 0.0
        pass_length = float(row.get("pass_length") or 0.0)
        if not _is_nan(pass_length):
            pass_length = max(0.0, min(1.0, pass_length / _MAX_PASS_LENGTH))
        else:
            pass_length = 0.0
        pass_angle_raw = float(row.get("pass_angle") or 0.0)
        if _is_nan(pass_angle_raw):
            pass_angle_raw = 0.0
        if needs_mirror:
            pass_angle_raw = -pass_angle_raw  # mirror the angle
        pass_angle_norm = pass_angle_raw / math.pi  # normalise to [-1, 1]
        pass_switch = 1.0 if row.get("pass_switch") else 0.0
        pass_cross = 1.0 if row.get("pass_cross") else 0.0
        shot_xg = float(row.get("shot_xg") or 0.0)
        if _is_nan(shot_xg):
            shot_xg = 0.0
        shot_xg = max(0.0, min(1.0, shot_xg))
        shot_first_time = 1.0 if row.get("shot_first_time") else 0.0

        # -- Pitch zone (3×3 grid) ----------------------------------------
        zone_x = min(int(nx * self.config.x_zones), self.config.x_zones - 1)
        zone_y = min(int(ny * self.config.y_zones), self.config.y_zones - 1)
        zone_idx = zone_x * self.config.y_zones + zone_y

        # -- Assemble vector -----------------------------------------------
        parts: List[np.ndarray] = []

        parts.append(self._one_hot("event_type", row.get("event_type") or "Pass"))
        parts.append(np.array([nx, ny], dtype=np.float32))
        parts.append(np.array([nex, ney, has_end], dtype=np.float32))
        parts.append(np.array([dx, dy], dtype=np.float32))
        parts.append(np.array([dist, angle], dtype=np.float32))
        parts.append(self._one_hot("play_pattern", row.get("play_pattern") or "Regular Play"))
        parts.append(self._one_hot("position", pos_name))
        parts.append(self._one_hot_optional("body_part", body_part))
        parts.append(self._one_hot_optional("pass_outcome", row.get("pass_outcome")))
        parts.append(self._one_hot_optional("shot_outcome", row.get("shot_outcome")))
        parts.append(self._one_hot_optional("dribble_outcome", row.get("dribble_outcome")))
        parts.append(self._one_hot_optional("pass_type", row.get("pass_type")))
        parts.append(self._one_hot_optional("pass_height", row.get("pass_height")))
        parts.append(self._one_hot_optional("shot_type", row.get("shot_type")))

        duration_norm = max(0.0, min(1.0, duration / _MAX_DURATION))
        scalars = np.array([
            duration_norm,
            under_pressure,
            counterpress,
            pass_length,
            pass_angle_norm,
            pass_switch,
            pass_cross,
            shot_xg,
            shot_first_time,
        ], dtype=np.float32)
        parts.append(scalars)

        zone_vec = np.zeros(self.config.x_zones * self.config.y_zones, dtype=np.float32)
        zone_vec[zone_idx] = 1.0
        parts.append(zone_vec)

        # StatsBomb 360 spatial context (freeze frame)
        spatial = self._spatial_360_vector(
            row.get("freeze_frame"),
            loc_x, loc_y,
            needs_mirror,
        )
        parts.append(spatial)

        return np.concatenate(parts)

    # ── helpers ──────────────────────────────────────────────────────

    def _one_hot(self, vocab_name: str, value: str) -> np.ndarray:
        vocab = self._vocab[vocab_name]
        vec = np.zeros(len(vocab), dtype=np.float32)
        idx = vocab.get(value)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def _one_hot_optional(self, vocab_name: str, value) -> np.ndarray:
        vocab = self._vocab[vocab_name]
        vec = np.zeros(len(vocab), dtype=np.float32)
        if value is not None and not _is_nan(value):
            idx = vocab.get(value)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def _spatial_360_vector(
        self,
        freeze_frame: Optional[List],
        actor_x: float,
        actor_y: float,
        needs_mirror: bool,
    ) -> np.ndarray:
        """
        Build a 9-dim vector from StatsBomb 360 freeze frame: counts (teammates, opponents, keepers),
        mean teammate/opponent positions (normalised), min distance to teammate/opponent (normalised).
        Returns zeros when freeze_frame is None or empty.
        """
        out = np.zeros(9, dtype=np.float32)
        if not freeze_frame or not isinstance(freeze_frame, list):
            return out

        max_dist = getattr(self.config, "max_dist_for_norm", 50.0)
        teammates_xy: List[Tuple[float, float]] = []
        opponents_xy: List[Tuple[float, float]] = []
        n_keepers = 0

        for entry in freeze_frame:
            if not isinstance(entry, dict):
                continue
            loc = entry.get("location")
            if not loc or len(loc) < 2:
                continue
            x = max(0.0, min(float(loc[0]), PITCH_LENGTH))
            y = max(0.0, min(float(loc[1]), PITCH_WIDTH))
            if needs_mirror:
                y = PITCH_WIDTH - y
            if entry.get("actor"):
                continue
            if entry.get("keeper"):
                n_keepers += 1
                continue
            if entry.get("teammate"):
                teammates_xy.append((x, y))
            else:
                opponents_xy.append((x, y))

        n_teammates = len(teammates_xy)
        n_opponents = len(opponents_xy)
        out[0] = min(1.0, n_teammates / 11.0)
        out[1] = min(1.0, n_opponents / 11.0)
        out[2] = float(min(1, n_keepers))

        if teammates_xy:
            out[3] = max(0.0, min(1.0, sum(p[0] for p in teammates_xy) / len(teammates_xy) / PITCH_LENGTH))
            out[4] = max(0.0, min(1.0, sum(p[1] for p in teammates_xy) / len(teammates_xy) / PITCH_WIDTH))
            min_d = min(
                math.sqrt((actor_x - p[0]) ** 2 + (actor_y - p[1]) ** 2) for p in teammates_xy
            )
            out[7] = max(0.0, min(1.0, min_d / max_dist))
        if opponents_xy:
            out[5] = max(0.0, min(1.0, sum(p[0] for p in opponents_xy) / len(opponents_xy) / PITCH_LENGTH))
            out[6] = max(0.0, min(1.0, sum(p[1] for p in opponents_xy) / len(opponents_xy) / PITCH_WIDTH))
            min_d = min(
                math.sqrt((actor_x - p[0]) ** 2 + (actor_y - p[1]) ** 2) for p in opponents_xy
            )
            out[8] = max(0.0, min(1.0, min_d / max_dist))

        return out


# ── Module-level constants ───────────────────────────────────────────────

# Approximate upper bounds for scalar normalisation (from data survey)
_MAX_PASS_LENGTH = 100.0   # metres; longest StatsBomb passes are ~80 m
_MAX_DURATION = 15.0       # seconds; most events are < 10 s


def _is_nan(v) -> bool:
    """Return True for None, NaN, or pandas NA."""
    if v is None:
        return True
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from .data_preparation import StatsBombDataLoader

    config = get_config()
    loader = StatsBombDataLoader(config.data)

    comps = loader.load_competitions()
    matches = loader.load_matches(comps[:1])
    print(f"Matches: {len(matches)}")

    df = loader.load_all_events(matches[:2])
    print(f"Events loaded: {len(df)}")

    encoder = EventFeatureEncoder(config.feature, mirror=True)
    print(f"Feature dimension: {encoder.feature_dim}")

    X, meta = encoder.encode_dataframe(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Sample vector (first 20 dims): {X[0, :20]}")
