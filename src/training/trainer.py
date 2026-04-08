"""
training/trainer.py — Training and validation loop for PassLocationGNN.

Usage
-----
    history = train(model, train_loader, val_loader, device)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import (
    EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, OUT_DIM, DEVICE,
)
from src.training.loss import xy_direction_loss, euclidean_error_yards


@dataclass
class TrainHistory:
    """Container for per-epoch training statistics."""
    train_loss:   List[float] = field(default_factory=list)
    train_smooth_l1: List[float] = field(default_factory=list)
    train_dir:    List[float] = field(default_factory=list)
    val_loss:     List[float] = field(default_factory=list)
    val_error_yards: List[float] = field(default_factory=list)

    @property
    def best_val_error(self) -> float:
        return min(self.val_error_yards) if self.val_error_yards else float("inf")

    @property
    def best_epoch(self) -> int:
        return self.val_error_yards.index(self.best_val_error) + 1


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device=DEVICE,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip: float = GRAD_CLIP,
    log_every: int = 5,
) -> TrainHistory:
    """
    Run the full training loop.

    Parameters
    ----------
    model        : PassLocationGNN (already moved to device)
    train_loader : PyG DataLoader for training graphs
    val_loader   : PyG DataLoader for validation graphs
    device       : torch device
    epochs       : number of training epochs
    log_every    : print summary every this many epochs (also prints epoch 1)

    Returns
    -------
    TrainHistory  — per-epoch statistics
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    history = TrainHistory()
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = total_sl1 = total_dir = 0.0

        pbar = tqdm(
            train_loader, unit="batch",
            desc=f"Epoch {epoch}/{epochs}", leave=False,
        )
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)                          # (B, 2)
            y = batch.y.view(-1, OUT_DIM)
            pp = batch.passer_pos.view(-1, 2)
            loss, sl1_v, dir_v = xy_direction_loss(pred, y, pp)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = batch.num_graphs
            total_loss += loss.item() * bs
            total_sl1 += sl1_v * bs
            total_dir += dir_v * bs
            pbar.set_postfix(loss=f"{loss.item():.5f}", dir=f"{dir_v:.4f}")

        avg_train = total_loss / n_train
        history.train_loss.append(avg_train)
        history.train_smooth_l1.append(total_sl1 / n_train)
        history.train_dir.append(total_dir / n_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = val_err_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                y = batch.y.view(-1, OUT_DIM)
                pp = batch.passer_pos.view(-1, 2)
                loss, _, _ = xy_direction_loss(pred, y, pp)
                val_loss_sum += loss.item() * batch.num_graphs
                val_err_sum += euclidean_error_yards(pred, y) * batch.num_graphs

        avg_val_loss = val_loss_sum / n_val
        avg_val_err = val_err_sum / n_val
        scheduler.step(avg_val_loss)

        history.val_loss.append(avg_val_loss)
        history.val_error_yards.append(avg_val_err)

        if epoch % log_every == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={avg_train:.5f}  "
                f"val={avg_val_loss:.5f}  "
                f"err={avg_val_err:.2f} yds  "
                f"lr={lr_now:.2e}"
            )

    print(
        f"\nBest val error: {history.best_val_error:.2f} yds "
        f"at epoch {history.best_epoch}"
    )
    return history
