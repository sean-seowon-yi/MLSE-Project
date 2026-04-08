"""
analysis/plots.py — All visualisation functions for PassLocationGNN.

Each function takes the data it needs and returns a matplotlib Figure,
making it easy to call from a notebook or a script.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from mplsoccer import Pitch

from config import PITCH_LEN, PITCH_WID, BG_COLOR as BG, DEVICE
from src.utils.inference import predict_with_player

_TICK_KW = dict(colors="white")
_SPINE_COLOR = "#3d4f61"
_LEGEND_KW = dict(labelcolor="white", facecolor="#1e2a3a")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history) -> plt.Figure:
    """Plot total loss, loss components, and val error over epochs."""
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.set_facecolor(BG)

    # Total loss
    ax = axes[0]; ax.set_facecolor(BG)
    ax.plot(epochs, history.train_loss, color="#60a5fa", lw=2, label="Train")
    ax.plot(epochs, history.val_loss,   color="#f97316", lw=2, label="Val")
    ax.set_title("Total Loss (MSE + Dir)", color="white", fontsize=11)
    ax.set_xlabel("Epoch", color="white")
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)

    # Loss components
    ax = axes[1]; ax.set_facecolor(BG)
    ax.plot(epochs, history.train_smooth_l1, color="#a78bfa", lw=2, label="Smooth L1")
    ax.plot(epochs, history.train_dir,       color="#34d399", lw=2, label="Direction")
    ax.set_title("Train Loss Components", color="white", fontsize=11)
    ax.set_xlabel("Epoch", color="white")
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)

    # Val error
    ax = axes[2]; ax.set_facecolor(BG)
    ax.plot(epochs, history.val_error_yards, color="#34d399", lw=2, label="Val error")
    ax.set_title("Val Error (yards)", color="white", fontsize=11)
    ax.set_xlabel("Epoch", color="white")
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)

    fig.suptitle("Training and Validation Curves", color="white", fontsize=14)
    plt.tight_layout()
    return fig


# ── Error distribution ────────────────────────────────────────────────────────

def plot_error_distribution(errors: np.ndarray, targets: np.ndarray, preds: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.set_facecolor(BG)

    # Histogram
    ax = axes[0]; ax.set_facecolor(BG)
    ax.hist(errors, bins=40, color="#60a5fa", edgecolor="white", lw=0.4)
    ax.axvline(errors.mean(),     color="#f97316", lw=2,
               label=f"Mean={errors.mean():.1f} yds")
    ax.axvline(np.median(errors), color="#34d399", lw=2, ls="--",
               label=f"Median={np.median(errors):.1f} yds")
    ax.set_title("Prediction Error Distribution", color="white", fontsize=11)
    ax.set_xlabel("Euclidean error (yards)", color="white")
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)

    # Predicted vs actual x
    ax = axes[1]; ax.set_facecolor(BG)
    ax.scatter(targets[:, 0], preds[:, 0], alpha=0.25, s=8, c="#a78bfa")
    lims = [0, PITCH_LEN]
    ax.plot(lims, lims, "white", lw=1, ls="--", label="Perfect")
    ax.set_title("Predicted vs Actual — x coordinate", color="white", fontsize=11)
    ax.set_xlabel("Actual x (yards)", color="white")
    ax.set_ylabel("Predicted x (yards)", color="white")
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)

    plt.tight_layout()
    return fig


# ── Spatial error heatmap ─────────────────────────────────────────────────────

def plot_spatial_error(errors: np.ndarray, targets: np.ndarray) -> plt.Figure:
    from scipy.stats import binned_statistic_2d

    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG,
                  line_color=_SPINE_COLOR, linewidth=1)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(16, 6))
    fig.set_facecolor(BG)

    sc = axes[0].scatter(
        targets[:, 0], targets[:, 1],
        c=errors, cmap="RdYlGn_r", s=15, alpha=0.7,
        vmin=0, vmax=np.percentile(errors, 95),
    )
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.8)
    cbar.set_label("Error (yards)", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    axes[0].set_title("Error by Actual End Location", color="white", fontsize=11, pad=8)

    stat, xedge, yedge, _ = binned_statistic_2d(
        targets[:, 0], targets[:, 1], errors,
        statistic="mean", bins=[16, 12],
        range=[[0, PITCH_LEN], [0, PITCH_WID]],
    )
    pcm = axes[1].pcolormesh(
        xedge, yedge, stat.T, cmap="RdYlGn_r", shading="auto",
        vmin=0, vmax=np.nanpercentile(stat, 95),
    )
    cbar2 = fig.colorbar(pcm, ax=axes[1], shrink=0.8)
    cbar2.set_label("Mean error (yards)", color="white")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
    axes[1].set_title("Mean Error per Zone", color="white", fontsize=11, pad=8)

    fig.suptitle("PassLocationGNN — Spatial Error Analysis", color="white", fontsize=13)
    plt.tight_layout()
    return fig


# ── Single pass visualisation ─────────────────────────────────────────────────

def visualise_pass_prediction(graph_data, model, ax, title: str = "", pred=None) -> None:
    """Draw one freeze-frame with predicted and actual pass end locations."""
    model.eval()
    with torch.no_grad():
        d = graph_data.to(DEVICE)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
    if pred is None:
        pred_n = model(d)[0].cpu().numpy()
    else:
        pred_n = pred
    pos = d.pos.cpu().numpy()
    x_feats = d.x.cpu().numpy()
    is_actor  = x_feats[:, 2] > 0.5
    is_tm     = x_feats[:, 3] > 0.5
    is_keeper = x_feats[:, 4] > 0.5
    is_cam    = x_feats[:, 5] > 0.5
    true_xy = d.end_loc_raw.cpu().numpy()
    pred_xy = pred_n * np.array([PITCH_LEN, PITCH_WID])

    cam_pos = pos[is_cam]
    if len(cam_pos):
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1],
                   c="#6b7280", s=30, marker="s", zorder=3, alpha=0.5, edgecolors="none")

    for i in range(len(pos)):
        if is_cam[i]:
            continue
        x, y = pos[i]
        if is_actor[i]:
            ax.scatter(x, y, c="#22c55e", s=300, marker="*", zorder=6, edgecolors="white", lw=1)
        elif is_keeper[i]:
            ax.scatter(x, y, c="#f59e0b", s=180, marker="D", zorder=5, edgecolors="white", lw=1)
        elif is_tm[i]:
            ax.scatter(x, y, c="#3b82f6", s=120, marker="o", zorder=5, edgecolors="white", lw=1, alpha=0.85)
        else:
            ax.scatter(x, y, c="#ef4444", s=100, marker="o", zorder=4, edgecolors="white", lw=0.8, alpha=0.7)

    actor_pos = pos[is_actor][0] if is_actor.any() else pos[0]
    ax.annotate("", xy=pred_xy, xytext=actor_pos,
                arrowprops=dict(arrowstyle="->", color="#f97316",
                                lw=2, connectionstyle="arc3,rad=0.1"))
    ax.scatter(*pred_xy, c="#f97316", s=220, marker="X", zorder=8, edgecolors="white", lw=1)
    ax.scatter(*true_xy, c="white",   s=200, marker="*", zorder=8, edgecolors="black", lw=0.5)

    err = np.linalg.norm(pred_xy - true_xy)
    ax.set_title(f"{title}  |  err={err:.1f} yds", color="white", fontsize=8, pad=5)


def plot_best_worst_predictions(val_set, model, n: int = 3) -> plt.Figure:
    """Show the n best and n worst predictions from the first 200 val graphs."""
    sample_errors = []
    model.eval()
    for i, g in enumerate(val_set[:200]):
        with torch.no_grad():
            d = g.to(DEVICE)
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=DEVICE)
            pred_n = model(d)[0].cpu().numpy()
        pred_xy = pred_n * np.array([PITCH_LEN, PITCH_WID])
        sample_errors.append((np.linalg.norm(pred_xy - g.end_loc_raw.cpu().numpy()), i))

    sample_errors.sort()
    best_idxs  = [i for _, i in sample_errors[:n]]
    worst_idxs = [i for _, i in sample_errors[-n:]]

    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG,
                  line_color=_SPINE_COLOR, linewidth=1)
    fig, axes = pitch.draw(nrows=2, ncols=n, figsize=(7 * n, 12))
    fig.set_facecolor(BG)

    labels = [f"Best #{k+1}" for k in range(n)] + [f"Worst #{k+1}" for k in range(n)]
    for ax, idx, label in zip(list(axes[0]) + list(axes[1]), best_idxs + worst_idxs, labels):
        visualise_pass_prediction(val_set[idx], model, ax, title=label)

    legend_handles = [
        plt.scatter([], [], c="#22c55e", marker="*", s=150, edgecolors="white", label="Passer"),
        plt.scatter([], [], c="#3b82f6", marker="o", s=80,  edgecolors="white", label="Teammate"),
        plt.scatter([], [], c="#ef4444", marker="o", s=80,  edgecolors="white", label="Opponent"),
        plt.scatter([], [], c="#f59e0b", marker="D", s=80,  edgecolors="white", label="GK"),
        plt.scatter([], [], c="#6b7280", marker="s", s=50,  edgecolors="none",  label="Camera boundary"),
        plt.scatter([], [], c="#f97316", marker="X", s=100, edgecolors="white", label="Predicted end"),
        plt.scatter([], [], c="white",   marker="*", s=100, edgecolors="black", label="Actual end"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7,
               facecolor="#1e2a3a", labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"PassLocationGNN — Best and Worst Predictions", color="white", fontsize=14)
    plt.tight_layout()
    return fig


# ── Zone/distance breakdown bars ─────────────────────────────────────────────

def plot_zone_breakdown(zone_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.set_facecolor(BG); ax.set_facecolor(BG)
    zone_summary["mean"].plot(kind="bar", ax=ax, color="#60a5fa", edgecolor="white")
    ax.set_title("Mean Error by Pass Origin Zone", color="white", fontsize=12)
    ax.set_xlabel("Zone", color="white"); ax.set_ylabel("Mean error (yards)", color="white")
    ax.tick_params(**_TICK_KW); ax.tick_params(axis="x", rotation=15)
    ax.spines[:].set_color(_SPINE_COLOR)
    plt.tight_layout()
    return fig


def plot_distance_breakdown(pass_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.set_facecolor(BG); ax.set_facecolor(BG)
    pass_summary["mean"].plot(kind="bar", ax=ax, color="#60a5fa", edgecolor="white")
    ax.set_title("Mean Error by Pass Distance", color="white", fontsize=12)
    ax.set_xlabel("Pass distance", color="white")
    ax.set_ylabel("Mean error (yards)", color="white")
    ax.tick_params(**_TICK_KW); ax.tick_params(axis="x", rotation=15)
    ax.spines[:].set_color(_SPINE_COLOR)
    plt.tight_layout()
    return fig


# ── Player similarity heatmaps ────────────────────────────────────────────────

def plot_prediction_with_player(
    graph_data,
    pid: List[int],
    model,
    player_vocab: Dict[int, int],
    pid_to_label: Dict[int, str],
) -> plt.Figure:
    """Visualise pass prediction with a specific player as the 'actor'."""
    
    sample_graph = graph_data
    print('True end location:', sample_graph.end_loc_raw.cpu().numpy())
    print()
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG, 
                line_color=_SPINE_COLOR, linewidth=1)
    n = len(pid)
    fig, axes = pitch.draw(nrows=1, ncols=n, figsize=(7 * n, 6))
    fig.set_facecolor(BG)
    labels = [f"{pid_to_label.get(i)}" for i in pid]
    for ax, idx, label in zip(list(axes), pid, labels):
        xy, xy_norm = predict_with_player(sample_graph, idx, model, player_vocab)
        visualise_pass_prediction(sample_graph, model, ax, title=label, pred=xy_norm)
        print(f'  player={pid_to_label.get(idx)}  ->  ({xy[0]:.1f}, {xy[1]:.1f}) yds')

def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    labels: List[str],
    title: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor(BG)
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5, color="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, color="white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine similarity", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(title, color="white", fontsize=11)

    if standalone:
        plt.tight_layout()
    return fig


def plot_four_method_comparison(
    matrices: List[np.ndarray],
    titles: List[str],
    labels: List[str],
) -> plt.Figure:
    """4-panel heatmap comparing all four similarity methods."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    fig.set_facecolor(BG)
    for ax, mat, title in zip(axes, matrices, titles):
        plot_similarity_heatmap(mat, labels, title, ax=ax)
    fig.suptitle("Player Similarity — Four Methods Compared", color="white", fontsize=13)
    plt.tight_layout()
    return fig


def plot_centroid_scatter(
    pred_centroids: pd.DataFrame,
    pid_to_label: Dict[int, str],
    n_label: int = 20,
) -> plt.Figure:
    """Scatter all players' predicted pass centroids on the pitch."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG,
                  line_color=_SPINE_COLOR, linewidth=1)
    fig, ax = pitch.draw(figsize=(14, 8))
    fig.set_facecolor(BG)

    cx = pred_centroids["mean_pred_x"].values
    cy = pred_centroids["mean_pred_y"].values
    nc = pred_centroids["n_passes"].values

    sc = ax.scatter(cx, cy, c=nc, cmap="plasma", s=60, alpha=0.85,
                    edgecolors="white", linewidths=0.4, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("N predictions per player", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for pid in list(pred_centroids.index)[:n_label]:
        x, y = pred_centroids.loc[pid, ["mean_pred_x", "mean_pred_y"]]
        surname = pid_to_label.get(pid, str(pid)).split(" (")[0].split()[-1]
        ax.annotate(surname, (x, y), color="white", fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points", zorder=6)

    ax.set_title("Predicted Pass Centroids (controlled situations per player)",
                 color="white", fontsize=12, pad=8)
    plt.tight_layout()
    return fig


def plot_kde_comparison(
    cross_buckets: Dict,
    pred_centroids: pd.DataFrame,
    pid_to_label: Dict[int, str],
    p1: int,
    p2: int,
) -> plt.Figure:
    """KDE density plot on pitch for two players' predicted distributions."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG,
                  line_color=_SPINE_COLOR, linewidth=1)
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(16, 6))
    fig.set_facecolor(BG)

    for ax, pid in zip(axes, [p1, p2]):
        pts = np.array(cross_buckets[pid])[:, :2]   # (N, 2)
        pitch.scatter(pts[:, 0], pts[:, 1], ax=ax,
                      s=20, color="#60a5fa", alpha=0.25, edgecolors="none")
        pitch.kdeplot(pts[:, 0], pts[:, 1], ax=ax,
                      cmap="Blues", fill=True, alpha=0.55, levels=6, bw_adjust=0.8)
        cx = pred_centroids.loc[pid, "mean_pred_x"]
        cy = pred_centroids.loc[pid, "mean_pred_y"]
        ax.scatter(cx, cy, c="#f97316", s=120, marker="X",
                   zorder=8, edgecolors="white", lw=1, label="Centroid")
        ax.legend(**_LEGEND_KW, fontsize=8)
        ax.set_title(pid_to_label.get(pid, str(pid)), color="white", fontsize=9, pad=6)

    fig.suptitle("Most Divergent Players — Cross-Predicted Distributions",
                 color="white", fontsize=13)
    plt.tight_layout()
    return fig


def print_cross_method_correlation(
    matrices: List[np.ndarray],
    method_names: List[str],
    shared_pids: List[int],
) -> None:
    """Print a Pearson correlation table comparing four similarity methods."""
    tri = np.triu_indices(len(shared_pids), k=1)
    vecs = {name: mat[tri] for name, mat in zip(method_names, matrices)}
    corr_mat = np.array([
        [float(np.corrcoef(vecs[a], vecs[b])[0, 1]) for b in method_names]
        for a in method_names
    ])
    header = f"  {'':20s}" + "".join(f"{l:>16s}" for l in method_names)
    print(header)
    for i, row_lbl in enumerate(method_names):
        row = f"  {row_lbl:20s}" + "".join(f"{corr_mat[i, j]:16.3f}" for j in range(len(method_names)))
        print(row)
    print()
    print("r ≈ 1.0  → feature-based profiles capture the same signal as direct comparison")
    print("r ≈ 0.0  → raw distributions contain structure the 10-d feature vector misses")


def plot_distinctiveness_bar(
    mean_ed: np.ndarray,
    labels: List[str],
) -> plt.Figure:
    """Bar chart of mean Energy Distance per player (style distinctiveness)."""
    P = len(mean_ed)
    order = np.argsort(mean_ed)
    colors = ["#34d399" if mean_ed[i] < np.median(mean_ed) else "#f97316" for i in order]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.set_facecolor(BG); ax.set_facecolor(BG)
    ax.bar(range(P), mean_ed[order], color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(P))
    ax.set_xticklabels([labels[i] for i in order],
                       rotation=55, ha="right", fontsize=7, color="white")
    ax.axhline(mean_ed.mean(), color="white", lw=1, ls="--", alpha=0.6,
               label=f"Mean = {mean_ed.mean():.1f} yds")
    ax.set_ylabel("Mean Energy Distance to all others (yds)", color="white")
    ax.set_title("Passing Style Distinctiveness — Energy Distance", color="white", fontsize=12)
    ax.tick_params(**_TICK_KW); ax.spines[:].set_color(_SPINE_COLOR)
    ax.legend(**_LEGEND_KW)
    plt.tight_layout()
    return fig
