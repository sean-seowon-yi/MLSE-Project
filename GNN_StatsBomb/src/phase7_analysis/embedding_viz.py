"""
Phase 7 – Embedding-space visualisations.

Provides:
  - Global PCA scatter (coloured by position group).
  - Per-query neighbourhood view (query + neighbours highlighted).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import POSITION_GROUPS


_POS_TO_GROUP: Dict[str, str] = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _grp


def compute_pca_coords(
    Z: np.ndarray,
    player_info: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    """Return {player_id: (x, y)} from a 2-component PCA of *Z*."""
    if Z.shape[0] == 0:
        return {}
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Z)
    pid_array = player_info["player_id"].to_numpy()
    return {int(pid_array[i]): coords[i] for i in range(len(pid_array))}


def plot_pca_global(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Path,
) -> None:
    """2-D PCA scatter of all players, coloured by position group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if Z.shape[0] == 0:
        return

    pca = PCA(n_components=2, random_state=0)
    Z_2d = pca.fit_transform(Z)

    df = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "position_name": player_info["position_name"].values,
    })
    df["group"] = df["position_name"].map(lambda p: _POS_TO_GROUP.get(p, "Other"))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="group", alpha=0.7, s=20, palette="tab10")
    plt.title("Player embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Position group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_pca.png", dpi=200)
    plt.close()


def plot_pca_neighbourhood(
    query_pid: int,
    neighbour_pids: List[int],
    pid_to_coord: Dict[int, np.ndarray],
    player_info: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Local PCA view: all players grey, query red, neighbours blue + labelled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pid_to_coord:
        return

    names = {
        int(row["player_id"]): row.get("player_name", "")
        for _, row in player_info.iterrows()
    }

    coords = np.stack(list(pid_to_coord.values()), axis=0)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c="lightgrey", s=10, alpha=0.4, label="Others")

    if query_pid in pid_to_coord:
        qx, qy = pid_to_coord[query_pid]
        plt.scatter([qx], [qy], c="red", s=60, zorder=5, label="Query")
        plt.annotate(
            names.get(query_pid, str(query_pid)),
            (qx, qy), fontsize=7, color="red",
            textcoords="offset points", xytext=(5, 5),
        )

    for cpid in neighbour_pids:
        if cpid not in pid_to_coord:
            continue
        cx, cy = pid_to_coord[cpid]
        plt.scatter([cx], [cy], c="blue", s=35, alpha=0.8, zorder=4)
        plt.annotate(
            names.get(cpid, str(cpid)),
            (cx, cy), fontsize=6, color="blue",
            textcoords="offset points", xytext=(5, 5),
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA neighbourhood — {names.get(query_pid, query_pid)}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / f"pca_neighbourhood_{query_pid}.png", dpi=200)
    plt.close()
