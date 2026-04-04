"""
Phase 7 – Embedding-space visualisations.

Provides:
  - Global PCA scatter in three variants: by position group (4), by subgroup (~8),
    and by full position (all 26+).
  - Global t-SNE scatter in the same three variants.
  - Per-query neighbourhood view (query + neighbours highlighted) in both PCA and t-SNE.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import POSITION_GROUPS, POSITION_SUBGROUPS


_POS_TO_GROUP: Dict[str, str] = {}
for _grp, _positions in POSITION_GROUPS.items():
    for _pos in _positions:
        _POS_TO_GROUP[_pos] = _grp


# ── coordinate computation ───────────────────────────────────────────────

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


def compute_tsne_coords(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    perplexity: float = 30.0,
) -> Dict[int, np.ndarray]:
    """Return {player_id: (x, y)} from a 2-component t-SNE of *Z*.

    For high-dimensional embeddings (D > 50), PCA pre-reduction to 50
    dims is applied first to speed up t-SNE and improve quality.
    """
    if Z.shape[0] == 0:
        return {}
    n = Z.shape[0]
    effective_perp = min(perplexity, (n - 1) / 3.0)
    if effective_perp < 2:
        return {}

    Z_input = Z
    if Z.shape[1] > 50:
        pca = PCA(n_components=50, random_state=0)
        Z_input = pca.fit_transform(Z)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perp,
        random_state=0,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(Z_input)
    pid_array = player_info["player_id"].to_numpy()
    return {int(pid_array[i]): coords[i] for i in range(len(pid_array))}


# ── shared helpers ───────────────────────────────────────────────────────

def _build_scatter_df(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    pid_to_coord: Optional[Dict[int, np.ndarray]] = None,
    reducer: str = "pca",
) -> pd.DataFrame:
    """Build 2D coords and add group/subgroup/position columns.

    If *pid_to_coord* is provided and has the right size, reuse it.
    Otherwise fit PCA or t-SNE from scratch.
    """
    if Z.shape[0] == 0:
        return pd.DataFrame()
    if pid_to_coord is not None and len(pid_to_coord) == Z.shape[0]:
        pids = player_info["player_id"].to_numpy()
        Z_2d = np.stack([pid_to_coord[int(p)] for p in pids], axis=0)
    elif reducer == "tsne":
        n = Z.shape[0]
        if n < 4:
            Z_2d = PCA(n_components=min(2, n), random_state=0).fit_transform(Z)
            if Z_2d.shape[1] < 2:
                Z_2d = np.column_stack([Z_2d, np.zeros(n)])
        else:
            Z_2d = TSNE(n_components=2, perplexity=min(30, (n - 1) / 3),
                         random_state=0, init="pca", learning_rate="auto").fit_transform(Z)
    else:
        Z_2d = PCA(n_components=2, random_state=0).fit_transform(Z)
    position_names = player_info["position_name"].values
    df = pd.DataFrame({
        "x": Z_2d[:, 0],
        "y": Z_2d[:, 1],
        "position_name": position_names,
    })
    df["group"] = df["position_name"].map(lambda p: _POS_TO_GROUP.get(p, "Other"))
    df["subgroup"] = df["position_name"].map(lambda p: POSITION_SUBGROUPS.get(p, "Unknown"))
    return df


def _save_scatter(
    df: pd.DataFrame,
    hue_col: str,
    title: str,
    legend_title: str,
    output_path: Path,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    palette: Optional[List] = None,
) -> None:
    """Single scatter: same layout and dpi for all variants."""
    if df.empty or hue_col not in df.columns:
        return
    figsize = (10, 6) if df[hue_col].nunique() > 12 else (8, 6)
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x="x", y="y", hue=hue_col, alpha=0.7, s=20, palette=palette)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_global_variants(
    df: pd.DataFrame,
    output_dir: Path,
    method: str,
    x_label: str,
    y_label: str,
) -> None:
    """Save three scatter variants (group, subgroup, position) for one method."""
    if df.empty:
        return

    _save_scatter(
        df,
        hue_col="group",
        title=f"Player embeddings ({method}) — position group",
        legend_title="Position group",
        output_path=output_dir / f"embeddings_{method.lower()}.png",
        x_label=x_label, y_label=y_label,
        palette="tab10",
    )

    _save_scatter(
        df,
        hue_col="subgroup",
        title=f"Player embeddings ({method}) — subgroup",
        legend_title="Subgroup",
        output_path=output_dir / f"embeddings_{method.lower()}_subgroup.png",
        x_label=x_label, y_label=y_label,
        palette="tab10",
    )

    n_pos = df["position_name"].nunique()
    palette_pos = sns.color_palette("husl", n_colors=n_pos) if n_pos > 10 else "tab10"
    _save_scatter(
        df,
        hue_col="position_name",
        title=f"Player embeddings ({method}) — position",
        legend_title="Position",
        output_path=output_dir / f"embeddings_{method.lower()}_position.png",
        x_label=x_label, y_label=y_label,
        palette=palette_pos,
    )


# ── PCA global & neighbourhood ──────────────────────────────────────────

def plot_pca_global(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Path,
    pid_to_coord: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """2-D PCA scatter of all players in three variants.

    Produces ``embeddings_pca.png``, ``embeddings_pca_subgroup.png``,
    ``embeddings_pca_position.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _build_scatter_df(Z, player_info, pid_to_coord, reducer="pca")
    _plot_global_variants(df, output_dir, "PCA", "PC1", "PC2")


def plot_pca_neighbourhood(
    query_pid: int,
    neighbour_pids: List[int],
    pid_to_coord: Dict[int, np.ndarray],
    player_info: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Local PCA view: all players grey, query red, neighbours blue + labelled."""
    _plot_neighbourhood(
        query_pid, neighbour_pids, pid_to_coord, player_info,
        output_dir, method="pca", x_label="PC1", y_label="PC2",
    )


# ── t-SNE global & neighbourhood ────────────────────────────────────────

def plot_tsne_global(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    output_dir: Path,
    pid_to_coord: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """2-D t-SNE scatter of all players in three variants.

    Produces ``embeddings_tsne.png``, ``embeddings_tsne_subgroup.png``,
    ``embeddings_tsne_position.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _build_scatter_df(Z, player_info, pid_to_coord, reducer="tsne")
    _plot_global_variants(df, output_dir, "t-SNE", "t-SNE 1", "t-SNE 2")


def plot_tsne_neighbourhood(
    query_pid: int,
    neighbour_pids: List[int],
    pid_to_coord: Dict[int, np.ndarray],
    player_info: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Local t-SNE view: all players grey, query red, neighbours blue + labelled."""
    _plot_neighbourhood(
        query_pid, neighbour_pids, pid_to_coord, player_info,
        output_dir, method="tsne", x_label="t-SNE 1", y_label="t-SNE 2",
    )


# ── shared neighbourhood plotter ────────────────────────────────────────

def _plot_neighbourhood(
    query_pid: int,
    neighbour_pids: List[int],
    pid_to_coord: Dict[int, np.ndarray],
    player_info: pd.DataFrame,
    output_dir: Path,
    method: str = "pca",
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
) -> None:
    """Neighbourhood scatter for either PCA or t-SNE coordinates."""
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

    first_neighbour = True
    for cpid in neighbour_pids:
        if cpid not in pid_to_coord:
            continue
        cx, cy = pid_to_coord[cpid]
        lbl = "Neighbour" if first_neighbour else None
        plt.scatter([cx], [cy], c="blue", s=35, alpha=0.8, zorder=4, label=lbl)
        first_neighbour = False
        plt.annotate(
            names.get(cpid, str(cpid)),
            (cx, cy), fontsize=6, color="blue",
            textcoords="offset points", xytext=(5, 5),
        )

    method_upper = method.upper() if method == "pca" else "t-SNE"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{method_upper} neighbourhood — {names.get(query_pid, query_pid)}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / f"{method}_neighbourhood_{query_pid}.png", dpi=200)
    plt.close()
