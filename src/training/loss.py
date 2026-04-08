"""
training/loss.py — Loss functions and evaluation metrics.

Loss
----
L = SmoothL1(pred_xy, true_xy)
  + DIR_LOSS_WEIGHT * (1 - cosine_similarity(pred_dir, true_dir))

where pred_dir = normalise(pred_xy_norm - passer_xy_norm)

Smooth L1 clips gradients for large residuals (long passes that miss by a lot),
preventing them from dominating weight updates.
The direction term penalises angle errors independently of distance.

Metric
------
Mean Euclidean distance in yards between predicted and true end location.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

from config import PITCH_LEN, PITCH_WID, DIR_LOSS_WEIGHT


# ── Loss ───────────────────────────────────────────────────────────────────────

def xy_direction_loss(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    passer_pos: torch.Tensor,
    dir_weight: float = DIR_LOSS_WEIGHT,
) -> Tuple[torch.Tensor, float, float]:
    """
    Combined Smooth L1 + cosine direction loss.

    Parameters
    ----------
    pred_norm   : (B, 2)  model output in [0, 1]^2
    y_norm      : (B, 2)  ground-truth in [0, 1]^2
    passer_pos  : (B, 2)  passer position in **yards**
    dir_weight  : λ for the direction term

    Returns
    -------
    loss    : scalar tensor
    mse_val : float  (smooth_l1 component)
    dir_val : float  (cosine direction component)
    """
    smooth_l1 = F.smooth_l1_loss(pred_norm, y_norm)

    scale = torch.tensor(
        [PITCH_LEN, PITCH_WID], dtype=torch.float, device=pred_norm.device
    )
    passer_norm = passer_pos / scale                      # (B, 2) in [0, 1]
    pred_dir = F.normalize(pred_norm - passer_norm, dim=1, eps=1e-8)
    true_dir = F.normalize(y_norm - passer_norm, dim=1, eps=1e-8)
    dir_loss = (1.0 - (pred_dir * true_dir).sum(dim=1)).mean()

    total = smooth_l1 + dir_weight * dir_loss
    return total, smooth_l1.item(), dir_loss.item()


# ── Metric ─────────────────────────────────────────────────────────────────────

def euclidean_error_yards(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
) -> float:
    """
    Mean Euclidean distance in yards between predicted and true end location.

    Parameters
    ----------
    pred_norm, y_norm : (B, 2)  normalised coordinates in [0, 1]^2
    """
    scale = torch.tensor(
        [PITCH_LEN, PITCH_WID], dtype=torch.float, device=pred_norm.device
    )
    return torch.linalg.norm((pred_norm - y_norm) * scale, dim=1).mean().item()
