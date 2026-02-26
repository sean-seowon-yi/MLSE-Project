"""
Loss functions for the player similarity training objectives.

Combined loss:
  L = L_action + λ_outcome · L_outcome + λ_contrast · L_contrast

1. **ActionPredictionLoss** (primary) — focal loss over discretised
   action targets (type, angle bin, length bin) to handle severe class
   imbalance (Pass/Carry dominate at ~70% of events).
2. **OutcomePredictionLoss** (secondary) — BCE for shot/goal flags.
3. **ContrastiveLoss** (auxiliary) — InfoNCE-style, player-ID-based.
4. **CombinedLoss** — weighted sum wrapper.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sqrt_weights(counts: List[int]) -> torch.Tensor:
    """Compute class weights proportional to 1/sqrt(count), normalised to mean 1."""
    inv = [1.0 / math.sqrt(c) for c in counts]
    s = sum(inv)
    n = len(counts)
    return torch.tensor([v / s * n for v in inv], dtype=torch.float32)


# Pre-computed from the training data (737 K events).
_ACTION_TYPE_COUNTS = [
    29416, 11820, 241151, 12697, 8141, 19445,
    6874, 6520, 4628, 6657, 8108, 281860, 91692, 8130,
]
ACTION_TYPE_WEIGHTS = _inverse_sqrt_weights(_ACTION_TYPE_COUNTS)

_LENGTH_BIN_COUNTS = [368974, 177437, 128760, 49408, 12560]
LENGTH_BIN_WEIGHTS = _inverse_sqrt_weights(_LENGTH_BIN_COUNTS)


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the contribution of easy/dominant-class examples so the model
    pays more attention to rare, harder classes (Shot, Dribble, etc.).
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()


class ActionPredictionLoss(nn.Module):
    """Focal loss over discretised action targets."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.type_loss = FocalLoss(gamma=gamma, weight=ACTION_TYPE_WEIGHTS)
        self.angle_loss = FocalLoss(gamma=gamma)
        self.length_loss = FocalLoss(gamma=gamma, weight=LENGTH_BIN_WEIGHTS)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        preds:   {"action_type": (E, C1), "angle_bin": (E, C2), "length_bin": (E, C3)}
        targets: {"action_type": (E,),    "angle_bin": (E,),    "length_bin": (E,)}
        """
        loss = self.type_loss(preds["action_type"], targets["action_type"])
        loss = loss + self.angle_loss(preds["angle_bin"], targets["angle_bin"])
        loss = loss + self.length_loss(preds["length_bin"], targets["length_bin"])
        return loss


class OutcomePredictionLoss(nn.Module):
    """BCE loss for possession-level shot / goal prediction."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits:  (B, 2)  — [ends_in_shot, ends_in_goal]
        targets: (B, 2)  — same, float {0, 1}
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss (auxiliary).

    Positives:  same player_id across different possessions within the batch.
    Negatives:  different player_ids within the batch.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        player_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeddings: (N, d)  — player-level embeddings within a batch.
        player_ids: (N,)    — player IDs so we know who matches whom.

        For each anchor, positives are other embeddings of the same player.
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb = F.normalize(embeddings, dim=-1)
        sim = emb @ emb.T / self.temperature  # (N, N)

        # Positive mask: same player, different sample
        pid = player_ids.unsqueeze(0)  # (1, N)
        pos_mask = (pid == pid.T).float()
        pos_mask.fill_diagonal_(0.0)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Log-sum-exp over all negatives (everything except self)
        neg_mask = torch.ones_like(sim)
        neg_mask.fill_diagonal_(0.0)
        log_denom = torch.logsumexp(sim * neg_mask + (1 - neg_mask) * (-1e9), dim=-1)

        # Mean of positive log-probs
        log_num = sim  # numerator logits
        pos_logprob = log_num - log_denom.unsqueeze(-1)
        loss = -(pos_logprob * pos_mask).sum() / pos_mask.sum()

        return loss


class CombinedLoss(nn.Module):
    """
    L = L_action + λ_outcome · L_outcome + λ_contrast · L_contrast
    """

    def __init__(
        self,
        lambda_outcome: float = 0.5,
        lambda_contrast: float = 1.0,
    ):
        super().__init__()
        self.action_loss = ActionPredictionLoss()
        self.outcome_loss = OutcomePredictionLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.lambda_outcome = lambda_outcome
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        action_targets: Dict[str, torch.Tensor],
        outcome_logits: Optional[torch.Tensor] = None,
        outcome_targets: Optional[torch.Tensor] = None,
        contrastive_embeddings: Optional[torch.Tensor] = None,
        contrastive_player_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with "total", "action", "outcome", "contrastive" losses.
        """
        l_action = self.action_loss(preds, action_targets)
        total = l_action.clone()

        l_outcome = torch.tensor(0.0, device=l_action.device)
        if outcome_logits is not None and outcome_targets is not None:
            l_outcome = self.outcome_loss(outcome_logits, outcome_targets)
            total = total + self.lambda_outcome * l_outcome

        l_contrast = torch.tensor(0.0, device=l_action.device)
        if contrastive_embeddings is not None and contrastive_player_ids is not None:
            l_contrast = self.contrastive_loss(
                contrastive_embeddings, contrastive_player_ids
            )
            total = total + self.lambda_contrast * l_contrast

        return {
            "total": total,
            "action": l_action,
            "outcome": l_outcome,
            "contrastive": l_contrast,
        }
