"""
Loss functions for the player similarity training objectives.

Combined loss (weights usually from ``TrainingConfig`` in ``config.py``):
  L = L_action + lambda_outcome * L_outcome + lambda_contrast * L_contrast
               + lambda_pooled * L_uniformity + lambda_align * L_alignment
               + lambda_pos * L_pos_group

1. **ActionPredictionLoss** (primary) -- focal loss over discretised
   action targets (type, angle bin, length bin) to handle severe class
   imbalance (Pass/Carry dominate at ~70% of events).
2. **OutcomePredictionLoss** (secondary) -- BCE for shot/goal flags.
3. **ContrastiveLoss** (auxiliary) -- InfoNCE-style, player-ID-based.
4. **CombinedLoss** -- weighted sum wrapper.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_sqrt_weights(counts: List[int]) -> torch.Tensor:
    """Compute class weights proportional to 1/sqrt(count), normalised to mean 1."""
    inv = [1.0 / math.sqrt(max(c, 1)) for c in counts]
    s = sum(inv)
    n = len(counts)
    return torch.tensor([v / s * n for v in inv], dtype=torch.float32)


# Pre-computed from processed_data/event_metadata + event_features (737,020 events; default 360 corpus).
_ACTION_TYPE_COUNTS = [
    29416, 11820, 241151, 12697, 8141, 19445,
    6874, 6520, 4624, 6657, 8108, 281860, 91692, 8015,
]
ACTION_TYPE_WEIGHTS = _inverse_sqrt_weights(_ACTION_TYPE_COUNTS)

_LENGTH_BIN_COUNTS = [369024, 177268, 128760, 49408, 12560]
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
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(0, targets)
            focal = alpha_t * focal
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
    InfoNCE-style contrastive loss with hard negatives (auxiliary).

    Positives:  same player_id across different possessions within the batch.
    Negatives:  different player_ids **in the same position group** within the
                batch.  This forces the model to separate players who share a
                positional role rather than relying on easy cross-role negatives.

    When no position-group information is supplied the loss falls back to
    using all different-player pairs as negatives (original behaviour).
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        player_ids: torch.Tensor,
        position_groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        embeddings      : (N, d)  — player-level embeddings within a batch.
        player_ids      : (N,)    — player IDs so we know who matches whom.
        position_groups : (N,)    — coarse position-group index per player
                                    (0=GK, 1=Def, 2=Mid, 3=Fwd, 4=Unknown).
                                    If None, all negatives are used.
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)

        emb = F.normalize(embeddings, dim=-1)
        sim = emb @ emb.T / self.temperature  # (N, N)

        pid = player_ids.unsqueeze(0)  # (1, N)
        pos_mask = (pid == pid.T).float()
        pos_mask.fill_diagonal_(0.0)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Hard-negative mask: only consider negatives from the same position
        # group so the loss pushes apart players with similar roles.
        if position_groups is not None:
            pg = position_groups.unsqueeze(0)  # (1, N)
            same_group = (pg == pg.T).float()  # (N, N)
        else:
            same_group = torch.ones_like(sim)

        # Negative mask: same group, different player, not self
        neg_mask = same_group * (1.0 - (pid == pid.T).float())
        neg_mask.fill_diagonal_(0.0)

        # Standard InfoNCE denominator: positives + hard negatives (all
        # non-self entries in the relevant set).  Including the positive in
        # the denominator bounds the loss to [0, log(N)] and stabilises
        # the gradient balance with other loss terms.
        denom_mask = torch.clamp(pos_mask + neg_mask, max=1.0)

        # If an anchor has zero denominator entries (no positives or
        # negatives in its group), fall back to all non-self pairs.
        has_denom = denom_mask.sum(dim=-1) > 0
        if not has_denom.all():
            fallback = torch.ones_like(sim)
            fallback.fill_diagonal_(0.0)
            denom_mask = torch.where(
                has_denom.unsqueeze(-1), denom_mask, fallback,
            )

        log_denom = torch.logsumexp(
            sim + (1.0 - denom_mask) * (-1e9), dim=-1,
        )

        pos_logprob = sim - log_denom.unsqueeze(-1)
        loss = -(pos_logprob * pos_mask).sum() / pos_mask.sum()

        return loss


class PooledUniformityLoss(nn.Module):
    """Push pooled z_p vectors apart on the unit hypersphere.

    Uses the Gaussian-potential uniformity loss from Wang & Isola (2020):
        L_uniform = log E[ exp(-t · ||z_i - z_j||^2) ]
    where the expectation is over all pairs i != j and *t* controls
    sensitivity (higher *t* -> stronger penalty on close pairs).

    When ``group_weight`` > 1 and position-group labels are supplied,
    same-group pairs receive extra weight so that the finite repulsive
    budget is focused on within-group separation (e.g. midfielder vs
    midfielder) where fine distinctions matter most for retrieval.
    """

    def __init__(self, t: float = 2.0, group_weight: float = 1.0):
        super().__init__()
        self.t = t
        self.group_weight = group_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        position_groups: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device)
        z = F.normalize(embeddings, dim=-1)
        sq_dists = torch.cdist(z, z, p=2).pow(2)
        n = z.shape[0]
        mask = 1.0 - torch.eye(n, device=z.device)
        exp_vals = torch.exp(-self.t * sq_dists) * mask

        if position_groups is not None and self.group_weight != 1.0:
            same_group = position_groups.unsqueeze(0) == position_groups.unsqueeze(1)
            weights = torch.where(same_group, self.group_weight, 1.0)
            weights = weights * mask
            loss = torch.log((exp_vals * weights).sum() / weights.sum())
        else:
            loss = torch.log(exp_vals.sum() / mask.sum())
        return loss


class EMAPlayerMemoryBank:
    """Momentum-updated memory of per-player pooled embeddings z_p.

    Stores a running EMA of each player's z_p across training batches.
    Used as a stable target for the alignment loss so that a player's
    pooled embedding stays self-consistent regardless of which
    possessions appear in the current batch.
    """

    def __init__(self, embed_dim: int, momentum: float = 0.999):
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.bank: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def lookup(
        self,
        player_ids: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_hist, mask) for the given player IDs.

        ``mask[i]`` is True when ``player_ids[i]`` already has a stored
        entry; otherwise ``z_hist[i]`` is a zero vector placeholder.
        """
        n = player_ids.shape[0]
        z_hist = torch.zeros(n, self.embed_dim)
        mask = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            pid = int(player_ids[i].item())
            if pid in self.bank:
                z_hist[i] = self.bank[pid]
                mask[i] = True
        return z_hist.to(device), mask.to(device)

    @torch.no_grad()
    def update(self, player_ids: torch.Tensor, z_p: torch.Tensor) -> None:
        """EMA-update the bank with the current batch's pooled embeddings."""
        m = self.momentum
        for i in range(player_ids.shape[0]):
            pid = int(player_ids[i].item())
            if pid < 0:
                continue
            vec = z_p[i].detach().cpu()
            if pid in self.bank:
                self.bank[pid] = m * self.bank[pid] + (1.0 - m) * vec
            else:
                self.bank[pid] = vec

    def state_dict(self) -> Dict:
        return {
            "bank": {k: v.clone() for k, v in self.bank.items()},
            "momentum": self.momentum,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.bank = state["bank"]
        self.momentum = state.get("momentum", self.momentum)


class AlignmentLoss(nn.Module):
    """Cosine alignment between current z_p and historical EMA z_p."""

    def forward(
        self,
        z_p: torch.Tensor,
        z_p_hist: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return torch.tensor(0.0, device=z_p.device)
        z_curr = F.normalize(z_p[mask], dim=-1)
        z_hist = F.normalize(z_p_hist[mask], dim=-1)
        return (1.0 - (z_curr * z_hist).sum(dim=-1)).mean()


class CombinedLoss(nn.Module):
    """
    L = L_action + lambda_outcome * L_outcome + lambda_contrast * L_contrast
                 + lambda_pooled * L_pooled_uniformity
                 + lambda_alignment * L_alignment
                 + lambda_pos * L_pos_group

    Default ``lambda_*`` here match ``TrainingConfig``; the trainer passes
    config values explicitly.
    """

    def __init__(
        self,
        lambda_outcome: float = 0.5,
        lambda_contrast: float = 0.5,
        lambda_pooled_contrast: float = 0.3,
        contrastive_temperature: float = 0.05,
        uniformity_t: float = 2.0,
        lambda_alignment: float = 0.0,
        lambda_pos: float = 0.0,
        uniformity_group_weight: float = 1.0,
    ):
        super().__init__()
        self.action_loss = ActionPredictionLoss()
        self.outcome_loss = OutcomePredictionLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.pooled_uniformity_loss = PooledUniformityLoss(
            t=uniformity_t, group_weight=uniformity_group_weight,
        )
        self.lambda_outcome = lambda_outcome
        self.lambda_contrast = lambda_contrast
        self.lambda_pooled_contrast = lambda_pooled_contrast
        self.lambda_alignment = lambda_alignment
        self.lambda_pos = lambda_pos

    def set_schedule(self, temperature: float, lambda_pooled: float) -> None:
        """Update temperature and pooled-uniformity weight mid-training."""
        self.contrastive_loss.temperature = temperature
        self.lambda_pooled_contrast = lambda_pooled

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        action_targets: Dict[str, torch.Tensor],
        outcome_logits: Optional[torch.Tensor] = None,
        outcome_targets: Optional[torch.Tensor] = None,
        contrastive_embeddings: Optional[torch.Tensor] = None,
        contrastive_player_ids: Optional[torch.Tensor] = None,
        contrastive_position_groups: Optional[torch.Tensor] = None,
        pooled_embeddings: Optional[torch.Tensor] = None,
        pooled_position_groups: Optional[torch.Tensor] = None,
        alignment_loss: Optional[torch.Tensor] = None,
        pos_group_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with "total", "action", "outcome", "contrastive",
        "pooled_uniform", "alignment", and "pos_group" losses.
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
                contrastive_embeddings,
                contrastive_player_ids,
                contrastive_position_groups,
            )
            total = total + self.lambda_contrast * l_contrast

        l_pooled = torch.tensor(0.0, device=l_action.device)
        if pooled_embeddings is not None and pooled_embeddings.shape[0] >= 2:
            l_pooled = self.pooled_uniformity_loss(
                pooled_embeddings,
                position_groups=pooled_position_groups,
            )
            total = total + self.lambda_pooled_contrast * l_pooled

        l_align = torch.tensor(0.0, device=l_action.device)
        if alignment_loss is not None:
            l_align = alignment_loss
            total = total + self.lambda_alignment * l_align

        l_pos = torch.tensor(0.0, device=l_action.device)
        if pos_group_loss is not None:
            l_pos = pos_group_loss
            total = total + self.lambda_pos * l_pos

        return {
            "total": total,
            "action": l_action,
            "outcome": l_outcome,
            "contrastive": l_contrast,
            "pooled_uniform": l_pooled,
            "alignment": l_align,
            "pos_group": l_pos,
        }
