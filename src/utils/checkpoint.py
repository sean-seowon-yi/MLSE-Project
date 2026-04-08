"""
utils/checkpoint.py — Model save and load helpers.

PyTorch 2.6 compatibility note
--------------------------------
torch.load() changed its default from weights_only=False to weights_only=True
in PyTorch 2.6. Checkpoints that contain non-tensor Python objects (dicts,
plain ints, floats) require weights_only=False to load correctly.

We address this on both sides:
  - save()  casts all player_vocab keys and values to plain Python ints so no
            numpy scalars are embedded in the file. This keeps new checkpoints
            clean and loadable with weights_only=True in future.
  - load()  passes weights_only=False explicitly. This is safe because the
            checkpoint is a file you saved yourself from this codebase.
            A warning is printed as a reminder.
"""

from __future__ import annotations

import torch
from typing import Dict

from config import CHECKPOINT_PATH
from src.model.gnn import PassLocationGNN


def save(
    model: PassLocationGNN,
    player_vocab: Dict,
    val_error_yards: float,
    path: str = CHECKPOINT_PATH,
) -> None:
    """
    Save model weights, architecture config, and vocabulary.

    player_vocab keys and values are cast to plain Python ints before saving
    so that no numpy scalar types are embedded in the checkpoint file.
    This avoids UnpicklingError on PyTorch >= 2.6.
    """
    # Cast vocab to pure Python ints — player_ids from pandas/numpy are int64
    clean_vocab = {int(k): int(v) for k, v in player_vocab.items()}

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # Architecture (all plain Python ints — safe to pickle)
            "node_dim":   int(model.input_proj.in_features),
            "edge_dim":   int(model.conv1.edge_dim),
            "global_dim": int(model.global_proj[0].in_features),
            "out_dim":    int(model.regressor[-1].out_features),
            "vocab_size": int(model.player_embed.num_embeddings),
            "embed_dim":  int(model.player_embed.embedding_dim),
            "hidden":     int(model.hidden),
            "heads":      int(model.conv1.heads),
            # Metadata
            "val_error_yards": float(val_error_yards),
            "player_vocab":    clean_vocab,
        },
        path,
    )
    print(f"Saved checkpoint -> {path}  (val_err={val_error_yards:.2f} yds)")


def load(path: str = CHECKPOINT_PATH, device=None) -> tuple:
    """
    Load a checkpoint and reconstruct the model.

    Uses weights_only=False because the checkpoint contains non-tensor Python
    objects (player_vocab dict, architecture ints). This is safe for checkpoints
    produced by this codebase. Do not use this function on untrusted files.

    Returns
    -------
    model        : PassLocationGNN  (eval mode, on device)
    player_vocab : Dict[player_id -> embed_idx]
    ckpt         : raw checkpoint dict (contains val_error_yards etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False is required for checkpoints that contain plain Python
    # objects alongside tensors (PyTorch >= 2.6 defaults to True).
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = PassLocationGNN(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        global_dim=ckpt["global_dim"],
        out_dim=ckpt["out_dim"],
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        hidden=ckpt["hidden"],
        heads=ckpt["heads"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_err = ckpt.get("val_error_yards", float("nan"))
    print(f"Loaded checkpoint from {path}  (val_err={val_err:.2f} yds)")
    return model, ckpt["player_vocab"], ckpt
