"""
Discretise continuous action attributes into categorical targets for the
masked-action / imitation objective.

Targets per event:
  1. ``action_type``  — index into EVENT_TYPES (14 classes).
  2. ``angle_bin``    — action direction partitioned into 8 sectors
                        (0 = forward, going clockwise).  Events without
                        an end-location get a special "no-angle" bin (8).
  3. ``length_bin``   — action displacement magnitude in 5 bins:
                        [very-short, short, medium, long, very-long].
                        Events without displacement get bin 0.
"""

import logging
import math
from typing import Dict, List

import numpy as np
import torch

from ..config import EVENT_TYPES, ModelConfig

logger = logging.getLogger(__name__)

_EVENT_TYPE_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(EVENT_TYPES)}

# Angle bins: 8 sectors each spanning 45 degrees, plus 1 "no-angle" bin
_N_ANGLE_BINS_RAW = 8
_NO_ANGLE_BIN = _N_ANGLE_BINS_RAW  # index 8

# Length bins: thresholds in normalised [0, 1] distance
# 0: ≤0.05  (very short / no movement)
# 1: ≤0.15  (short)
# 2: ≤0.30  (medium)
# 3: ≤0.50  (long)
# 4: >0.50  (very long / switch)
_LENGTH_THRESHOLDS = [0.05, 0.15, 0.30, 0.50]


class ActionTargetEncoder:
    """
    Compute discrete action targets from Phase 1 feature vectors.

    The 126-D vector layout is used directly (see ``masking.py`` for
    the index map).  Target indices (0-21) are in the first 22 dims
    and are unaffected by the period feature appended at the end.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def encode(
        self,
        event_features: np.ndarray,
        event_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        event_features : (T, 126)
            **Un-masked** Phase 1 vectors (action info still present).
        event_types : list[str], length T

        Returns
        -------
        dict with keys:
          "action_type" : LongTensor (T,)
          "angle_bin"   : LongTensor (T,)
          "length_bin"  : LongTensor (T,)
        """
        T = len(event_types)

        action_type = torch.zeros(T, dtype=torch.long)
        angle_bin = torch.full((T,), _NO_ANGLE_BIN, dtype=torch.long)
        length_bin = torch.zeros(T, dtype=torch.long)

        for i in range(T):
            etype = event_types[i]
            idx = _EVENT_TYPE_TO_IDX.get(etype)
            if idx is None:
                logger.warning("Unknown event type %r at index %d, defaulting to 0", etype, i)
                idx = 0
            action_type[i] = idx

            # Delta (indices 19, 20); distance recomputed from raw deltas
            # so binning is independent of normalisation at index 21.
            dx = float(event_features[i, 19])
            dy = float(event_features[i, 20])
            dist = math.sqrt(dx * dx + dy * dy)
            has_end = float(event_features[i, 18])  # end_location_2 flag

            if has_end > 0.5 and (abs(dx) > 1e-6 or abs(dy) > 1e-6):
                angle = math.atan2(dy, dx)  # radians [-π, π]
                # Map to [0, 2π) then to bin
                if angle < 0:
                    angle += 2 * math.pi
                bin_idx = int(angle / (2 * math.pi) * _N_ANGLE_BINS_RAW)
                angle_bin[i] = min(bin_idx, _N_ANGLE_BINS_RAW - 1)

                # Length bin
                for j, thresh in enumerate(_LENGTH_THRESHOLDS):
                    if dist <= thresh:
                        length_bin[i] = j
                        break
                else:
                    length_bin[i] = len(_LENGTH_THRESHOLDS)
            else:
                angle_bin[i] = _NO_ANGLE_BIN
                length_bin[i] = 0

        return {
            "action_type": action_type,
            "angle_bin": angle_bin,
            "length_bin": length_bin,
        }
