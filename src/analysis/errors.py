"""
analysis/errors.py — Validation error collection and breakdown analysis.

evaluate()           : collect predictions and errors across the val set
zone_breakdown()     : mean error by pass origin zone
distance_breakdown() : mean error by pass distance
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

from config import PITCH_LEN, PITCH_WID, OUT_DIM, DEVICE


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    player_vocab: Dict,
    device=DEVICE,
) -> Dict[str, np.ndarray]:
    """
    Run the model over the full validation set and collect results.

    Returns
    -------
    dict with keys:
        preds          (N, 2)  predicted end location in yards
        targets        (N, 2)  true end location in yards
        errors         (N,)    Euclidean error in yards
        passer_positions (N, 2) passer position in yards
        player_ids     list[Optional[int]]  player_id per graph
    """
    scale = torch.tensor([PITCH_LEN, PITCH_WID])
    idx_to_pid = {v: k for k, v in player_vocab.items()}

    all_preds, all_targets, all_errors = [], [], []
    all_passer_pos, all_player_ids = [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)                  # (B, 2) normalised
            y = batch.y.view(-1, OUT_DIM).cpu()
            pp = batch.passer_pos.view(-1, 2).cpu()

            pred_xy = pred.cpu() * scale
            true_xy = y.cpu() * scale
            errs = torch.linalg.norm(pred_xy - true_xy, dim=1)

            all_preds.append(pred_xy)
            all_targets.append(true_xy)
            all_errors.append(errs)
            all_passer_pos.append(pp)

            actor_idxs = batch.actor_idx.view(-1).cpu().tolist()
            all_player_ids.extend(
                [idx_to_pid.get(int(a), None) for a in actor_idxs]
            )

    return {
        "preds":             torch.cat(all_preds).cpu().numpy(),
        "targets":           torch.cat(all_targets).cpu().numpy(),
        "errors":            torch.cat(all_errors).cpu().numpy(),
        "passer_positions":  torch.cat(all_passer_pos).cpu().numpy(),
        "player_ids":        all_player_ids,
    }


def print_summary(errors: np.ndarray) -> None:
    print(f"Val set size:     {len(errors)}")
    print(f"Mean error:       {errors.mean():.2f} yards")
    print(f"Median error:     {np.median(errors):.2f} yards")
    print(f"90th percentile:  {np.percentile(errors, 90):.2f} yards")
    print(f"Within  5 yards:  {(errors < 5).mean():.1%}")
    print(f"Within 10 yards:  {(errors < 10).mean():.1%}")
    print(f"Within 20 yards:  {(errors < 20).mean():.1%}")


def zone_breakdown(errors: np.ndarray, passer_positions: np.ndarray) -> pd.DataFrame:
    """Mean error by pass origin zone (defensive / middle / attacking third)."""
    origin_x = passer_positions[:, 0]
    zones = pd.cut(
        origin_x, bins=[0, 40, 80, 120],
        labels=["Defensive third", "Middle third", "Attacking third"],
    )
    df = pd.DataFrame({"zone": zones, "error_yards": errors})
    return df.groupby("zone", observed=True)["error_yards"].agg(
        ["mean", "median", "count"]
    )


def distance_breakdown(
    errors: np.ndarray,
    passer_positions: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    """Mean error by pass distance (short / medium / long)."""
    dists = np.linalg.norm(passer_positions - targets, axis=1)
    bins = pd.cut(
        dists, bins=[0, 15, 40, 120],
        labels=["Short (<15 yds)", "Medium (15-40)", "Long (>40)"],
    )
    df = pd.DataFrame({"distance": bins, "error_yards": errors})
    return df.groupby("distance", observed=True)["error_yards"].agg(
        ["mean", "median", "count"]
    )
