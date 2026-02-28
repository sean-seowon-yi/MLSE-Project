"""
Phase 7 – Embedding-space visualisations.

Provides:
  - Global PCA scatter in three variants: by position group (4), by subgroup (~8),
    and by full position (all 26+).
  - Per-query neighbourhood view (query + neighbours highlighted).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import POSITION_GROUPS, POSITION_SUBGROUPS


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


def _build_pca_df(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    pid_to_coord: Optional[Dict[int, np.ndarray]] = None,
) -> pd.DataFrame:
    """Build 2D PCA coords and add group/subgroup/position columns. Returns empty df if no data."""
    if Z.shape[0] == 0:
        return pd.DataFrame()
    if pid_to_coord is not None and len(pid_to_coord) == Z.shape[0]:
        pids = player_info["player_id"].to_numpy()
        Z_2d = np.stack([pid_to_coord[int(p)] for p in pids], axis=0)
    else:
        pca = PCA(n_components=2, random_state=0)
        Z_2d = pca.fit_transform(Z)
    position_names = player_info["position_name"].values
    df = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "position_name": position_names,
    })
    df["group"] = df["position_name"].map(lambda p: _POS_TO_GROUP.get(p, "Other"))
    df["subgroup"] = df["position_name"].map(lambda p: POSITION_SUBGROUPS.get(p, "Unknown"))
    return df


def _save_pca_scatter(
    df: pd.DataFrame,
    hue_col: str,
    title: str,
    legend_title: str,
    output_path: Path,
    palette: Optional[List] = None,
) -> None:
    """Single PCA scatter: same layout and dpi for all variants."""
    if df.empty or hue_col not in df.columns:
        return
    figsize = (10, 6) if df[hue_col].nunique() > 12 else (8, 6)
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x="x", y="y", hue=hue_col, alpha=0.7, s=20, palette=palette)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pca_global(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Path,
    pid_to_coord: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """2-D PCA scatter of all players in three variants: group, subgroup, and full position.

    Produces:
      - embeddings_pca.png          — coloured by position group (4 categories).
      - embeddings_pca_subgroup.png — coloured by subgroup (~8 categories).
      - embeddings_pca_position.png — coloured by full position name (all 26+).

    If *pid_to_coord* is provided (from ``compute_pca_coords``), reuses
    those coordinates instead of refitting PCA so all three plots share the same 2D layout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _build_pca_df(Z, player_info, pid_to_coord)
    if df.empty:
        return

    # 1) Coarse: 4 groups (existing behaviour)
    _save_pca_scatter(
        df,
        hue_col="group",
        title="Player embeddings (PCA) — position group",
        legend_title="Position group",
        output_path=output_dir / "embeddings_pca.png",
        palette="tab10",
    )

    # 2) Medium: subgroups
    _save_pca_scatter(
        df,
        hue_col="subgroup",
        title="Player embeddings (PCA) — subgroup",
        legend_title="Subgroup",
        output_path=output_dir / "embeddings_pca_subgroup.png",
        palette="tab10",
    )

    # 3) Fine: all positions (need palette with enough colours)
    n_pos = df["position_name"].nunique()
    palette_pos = sns.color_palette("husl", n_colors=n_pos) if n_pos > 10 else "tab10"
    _save_pca_scatter(
        df,
        hue_col="position_name",
        title="Player embeddings (PCA) — position",
        legend_title="Position",
        output_path=output_dir / "embeddings_pca_position.png",
        palette=palette_pos,
    )


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
