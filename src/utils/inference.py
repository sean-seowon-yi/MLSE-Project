"""
utils/inference.py — Inference helpers for PassLocationGNN.

predict()             : run the model on a single graph
predict_with_player() : swap actor_idx and re-run (player style transfer)
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple

from config import PITCH_LEN, PITCH_WID, DEVICE


def predict(
    graph_data: Data,
    model: torch.nn.Module,
    device=DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single graph.

    Returns
    -------
    pred_xy   : (2,) predicted end location in yards
    pred_norm : (2,) raw model output [x_norm, y_norm]
    """
    model.eval()
    with torch.no_grad():
        d = graph_data.clone().to(device)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
        pred_norm = model(d)[0].cpu().numpy()

    pred_xy = pred_norm * np.array([PITCH_LEN, PITCH_WID])
    return pred_xy, pred_norm


def predict_with_player(
    graph_data: Data,
    player_id: int,
    model: torch.nn.Module,
    player_vocab: Optional[Dict] = None,
    device=DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with a specific player identity injected as actor_idx.

    Useful for counterfactual analysis: "where would player X have passed
    in this exact freeze-frame situation?"

    Parameters
    ----------
    graph_data  : a single (unbatched) PyG Data object
    player_id   : StatsBomb player_id to inject
    model       : trained PassLocationGNN
    player_vocab: dict player_id -> embed_idx  (uses 0=UNK if player unknown)
    device      : torch device

    Returns
    -------
    pred_xy   : (2,) predicted end location in yards
    pred_norm : (2,) raw model output [x_norm, y_norm]
    """
    vocab = player_vocab or {}
    idx = vocab.get(player_id, 0)
    if idx == 0:
        print(f"Warning: player_id {player_id} not in vocab — using <UNK>")

    model.eval()
    with torch.no_grad():
        d = graph_data.clone().to(device)
        d.actor_idx = torch.tensor([idx], dtype=torch.long, device=device)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
        pred_norm = model(d)[0].cpu().numpy()

    pred_xy = pred_norm * np.array([PITCH_LEN, PITCH_WID])
    return pred_xy, pred_norm
