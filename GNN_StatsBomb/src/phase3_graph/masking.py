"""
Feature-masking utilities for the GNN branch.

Two independent masks are applied to the 122-D Phase 1 event vectors:

1. **mask_spatial_360** — zeros out the 9-dim Spatial_360 block so the
   GNN must learn spatial context via explicit player nodes and
   ``E_context`` edges rather than the pre-aggregated summary stats.

2. **mask_future_info** — zeros out every dimension that reveals the
   *action, its outcome, or the actor's identity* from event features.
   Used for the masked-action / imitation training objective (Phase 5).
   After masking, the surviving non-zero features are purely pre-action
   situational state: location, play_pattern, under_pressure,
   counterpress, and pitch_zone.

   Position (32-57) is masked because it is a **player-level attribute**,
   not a situational one.  Two events at the same pitch location with the
   same surrounding players are the "same situation" regardless of
   whether the actor is a Left Wing or a Center Back.  Keeping position
   in the event features lets the model predict actions from role alone
   (e.g. "Left Wings pass forward"), bypassing ``z_p`` and reducing its
   informativeness.  Position information still reaches the model through
   the player node's ``position_idx`` embedding → ``h_player`` → ``z_p``,
   which is the correct channel for player identity.

Index Layout (122-D)
────────────────────
  0-13   event_type        (14)  ← masked (IS the prediction target)
 14-15   location           (2)
 16-18   end_location       (3)  ← masked
 19-20   delta              (2)  ← masked
 21-22   dist_angle         (2)  ← masked
 23-31   play_pattern       (9)
 32-57   position          (26)  ← masked (player identity, not situation)
 58-64   body_part          (7)  ← masked (reveals action modality)
 65-70   pass_outcome       (6)  ← masked
 71-78   shot_outcome       (8)  ← masked
 79-80   dribble_outcome    (2)  ← masked
 81-88   pass_type          (8)  ← masked
 89-91   pass_height        (3)  ← masked
 92-94   shot_type          (3)  ← masked
 95-103  scalars            (9)  ← partially masked (see below)
104-112  pitch_zone         (9)
113-121  spatial_360        (9)  ← masked for GNN (separate mask)

Scalars (indices 95-103):
  95  duration           — MASKED (determined by the action taken)
  96  under_pressure     — allowed (pre-action)
  97  counterpress       — allowed (pre-action)
  98  pass_length        — MASKED
  99  pass_angle         — MASKED
 100  pass_switch        — MASKED
 101  pass_cross         — MASKED
 102  shot_xg            — MASKED
 103  shot_first_time    — MASKED
"""

from typing import List

import numpy as np


# ── Spatial_360 ──────────────────────────────────────────────────────────

_SPATIAL_360_START = 113
_SPATIAL_360_END = 122  # exclusive


def get_spatial_360_indices() -> List[int]:
    """Return indices [113 .. 121] for the 9-dim Spatial_360 block."""
    return list(range(_SPATIAL_360_START, _SPATIAL_360_END))


def mask_spatial_360(x: np.ndarray) -> np.ndarray:
    """Zero out Spatial_360 dims.  Works on (122,) or (N, 122)."""
    out = x.copy()
    out[..., _SPATIAL_360_START:_SPATIAL_360_END] = 0.0
    return out


# ── Future / action info ────────────────────────────────────────────────

_MASKED_RANGES: List[range] = [
    range(0, 14),     # event_type one-hot — IS the prediction target
    range(16, 23),    # end_location (3) + delta (2) + dist_angle (2)
    range(32, 58),    # position — player identity, not situation; flows via player nodes
    range(58, 65),    # body_part — reveals action modality (foot/head/etc.)
    range(65, 95),    # pass_outcome (6) + shot_outcome (8) + dribble_outcome (2)
                      # + pass_type (8) + pass_height (3) + shot_type (3)
    range(95, 96),    # duration — determined by the action, not pre-action
    range(98, 104),   # action-revealing scalars: pass_length, pass_angle,
                      # pass_switch, pass_cross, shot_xg, shot_first_time
]


def get_future_info_indices() -> List[int]:
    """Return all feature indices that are masked (action, outcome, and player identity)."""
    indices: List[int] = []
    for r in _MASKED_RANGES:
        indices.extend(r)
    return sorted(indices)


def mask_future_info(x: np.ndarray) -> np.ndarray:
    """Zero out action/outcome/identity dims.  Works on (122,) or (N, 122)."""
    out = x.copy()
    for r in _MASKED_RANGES:
        out[..., r.start:r.stop] = 0.0
    return out
