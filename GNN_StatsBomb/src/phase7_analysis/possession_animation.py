"""
Animated visualization of a possession: ball trajectory + counterfactual
predicted displacement (angle × length) for two players' global traits.

Uses one ``encode_possession_counterfactual`` forward per possession, then
FiLM + heads per event (see :class:`SituationComparator.compare_possession`).

Default playback is slow (0.5 FPS) so each event is easy to read; override
with ``fps`` / ``--animation-fps``. Arrows are capped in length for display
only; the pitch uses minimal markings to reduce visual clutter.

**Geometry:** Event (x, y) comes from the same normalised location as Phase 1
(``event_features[:, 14:16]`` scaled by ``PITCH_LENGTH`` / ``PITCH_WIDTH``). The yellow
trail connects *consecutive event locations* in the possession graph, not a
continuous ball trajectory, so long segments (e.g. after a long pass) are
expected and are not a projection bug.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..config import Config, EVENT_TYPES, PITCH_LENGTH, PITCH_WIDTH, validate_graph_config_match
from ..phase3_graph import PossessionGraphBuilder
from ..phase4_model import PlayerSimilarityModel
from .situation_comparison import SituationComparator

# Length-bin midpoints in normalized sqrt(dx^2+dy^2) space (matches action_targets.py)
_LENGTH_MID_NORM = np.array([0.025, 0.10, 0.225, 0.40, 0.75], dtype=np.float32)

# Display-only: very long predicted displacements are hard to read; cap arrow length (meters).
_DISPLAY_ARROW_MAX_M = 28.0

# Default animation speed: frames per second (low = slower, easier to read).
_DEFAULT_ANIMATION_FPS = 0.5

# Horizontal offset (m) between two counterfactual arrows from the same ball location.
_ARROW_ORIGIN_SEP_M = 1.15

_ANGLE_LABELS = [
    "Fwd", "Fwd-R", "Right", "Back-R",
    "Back", "Back-L", "Left", "Fwd-L",
    "No angle",
]
_LENGTH_LABELS = [
    "V. short", "Short", "Medium", "Long", "V. long",
]


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen--Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def _gt_label(sit: dict) -> str:
    """One-line ground truth summary from the situation dict."""
    gt_act_idx = sit.get("gt_action_type_idx")
    gt_ang = sit.get("gt_angle_bin")
    gt_len = sit.get("gt_length_bin")
    if gt_act_idx is None:
        return "GT: (unavailable)"
    act = EVENT_TYPES[gt_act_idx] if gt_act_idx < len(EVENT_TYPES) else f"act{gt_act_idx}"
    alab = _ANGLE_LABELS[gt_ang] if gt_ang is not None and gt_ang < len(_ANGLE_LABELS) else "?"
    llab = _LENGTH_LABELS[gt_len] if gt_len is not None and gt_len < len(_LENGTH_LABELS) else "?"
    return f"GT: {act}  |  Dir: {alab}  |  Len: {llab}"


def _match_symbols(pred_idx: int, gt_idx: int) -> str:
    """Return a check or cross symbol."""
    return "\u2713" if pred_idx == gt_idx else "\u2717"


def _format_player_prediction_block(
    pid: int,
    pr: dict,
    name: str,
    tag: str,
    *,
    gt_action_idx: Optional[int] = None,
    gt_angle_bin: Optional[int] = None,
    gt_length_bin: Optional[int] = None,
    is_actor: bool = False,
) -> str:
    """Compact text block for sidebar: top actions, direction, length.

    When *is_actor* and ground-truth bins are given, adds match/mismatch symbols.
    """
    at = np.asarray(pr["action_type"], dtype=np.float64).ravel()
    ang = np.asarray(pr["angle_bin"], dtype=np.float64).ravel()
    ln = np.asarray(pr["length_bin"], dtype=np.float64).ravel()
    top_idx = np.argsort(at)[::-1][:3]
    lines_act = []
    for j in top_idx:
        ji = int(j)
        if ji >= len(EVENT_TYPES):
            continue
        lines_act.append(f"  {EVENT_TYPES[ji][:18]:18s} {at[ji]:5.1%}")
    ia = int(np.argmax(ang))
    il = int(np.argmax(ln))
    alab = _ANGLE_LABELS[ia] if ia < len(_ANGLE_LABELS) else f"bin {ia}"
    llab = _LENGTH_LABELS[il] if il < len(_LENGTH_LABELS) else f"bin {il}"
    nm = name[:20] if name else str(pid)
    header = f"{tag}\n{nm}  (id {pid})"
    body = (
        "\nTop actions:\n"
        + "\n".join(lines_act)
        + f"\nDir: {alab} ({ang[ia]:.0%})   Len: {llab} ({ln[il]:.0%})"
    )

    if is_actor and gt_action_idx is not None:
        pred_act = int(np.argmax(at))
        act_sym = _match_symbols(pred_act, gt_action_idx)
        gt_act_name = EVENT_TYPES[gt_action_idx][:14] if gt_action_idx < len(EVENT_TYPES) else "?"
        ang_sym = _match_symbols(ia, gt_angle_bin) if gt_angle_bin is not None else "?"
        len_sym = _match_symbols(il, gt_length_bin) if gt_length_bin is not None else "?"
        gt_alab = _ANGLE_LABELS[gt_angle_bin] if gt_angle_bin is not None and gt_angle_bin < len(_ANGLE_LABELS) else "?"
        gt_llab = _LENGTH_LABELS[gt_length_bin] if gt_length_bin is not None and gt_length_bin < len(_LENGTH_LABELS) else "?"
        body += (
            f"\n--- vs Ground Truth ---"
            f"\nAct: {act_sym} (GT={gt_act_name})"
            f"\nDir: {ang_sym} (GT={gt_alab})"
            f"\nLen: {len_sym} (GT={gt_llab})"
        )

    return header + "\n" + body


def _top_action_label(pr: dict, max_chars: int = 12) -> str:
    """Argmax action-type name, truncated."""
    at = np.asarray(pr["action_type"], dtype=np.float64).ravel()
    idx = int(np.argmax(at))
    if idx >= len(EVENT_TYPES):
        return f"act{idx}"
    return EVENT_TYPES[idx][:max_chars]


def _possession_change_caption(
    prev_tid: Optional[int],
    curr_tid: Optional[int],
) -> str:
    """Short label for the first event after a possession boundary (multi-segment anim)."""
    if prev_tid is None or curr_tid is None:
        return "NEW POSSESSION SEGMENT"
    if prev_tid != curr_tid:
        return (
            f"NEW POSSESSION — team {curr_tid} has the ball now "
            f"(previous segment: team {prev_tid})"
        )
    return (
        f"NEW POSSESSION — same controlling team (ID {curr_tid}); "
        "new StatsBomb possession chain"
    )


def _draw_progress_bar(
    ax,
    current: int,
    total: int,
    *,
    y_frac: float = 0.0,
    bar_height: float = 0.012,
    bg_color: str = "#1e293b",
    fill_color: str = "#22d3ee",
    possession_segment: Optional[Tuple[int, int]] = None,
    segment_boundary_highlight: bool = False,
) -> None:
    """Thin progress bar at the bottom of the axes (in axes coords)."""
    from matplotlib.patches import FancyBboxPatch
    if segment_boundary_highlight:
        fill_color = "#10b981"
    ax.add_patch(FancyBboxPatch(
        (0.02, y_frac), 0.96, bar_height,
        boxstyle="round,pad=0.002", transform=ax.transAxes,
        facecolor=bg_color, edgecolor="#475569", linewidth=0.5,
        clip_on=False, zorder=20,
    ))
    frac = (current + 1) / max(total, 1)
    ax.add_patch(FancyBboxPatch(
        (0.02, y_frac), 0.96 * frac, bar_height,
        boxstyle="round,pad=0.002", transform=ax.transAxes,
        facecolor=fill_color, edgecolor="none", alpha=0.85,
        clip_on=False, zorder=21,
    ))
    parts: List[str] = []
    if possession_segment is not None:
        ps, pt = possession_segment
        parts.append(f"Possession seg. {ps}/{pt}")
    parts.append(f"Event {current + 1} / {total}")
    label = "  |  ".join(parts)
    ax.text(
        0.5, y_frac + bar_height / 2,
        label,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=6.5, color="white", fontweight="bold", zorder=22,
    )


_PANEL_COLORS = ["#22d3ee", "#fb923c", "#a78bfa", "#34d399"]


def _draw_comparison_panel(
    ax_panel,
    pids: Sequence[int],
    preds: dict,
    names: dict,
    *,
    role_tags: Optional[Sequence[str]] = None,
    subtitle: str = "Same situation, different trait z_p",
    situation: Optional[dict] = None,
    actor_pid: Optional[int] = None,
) -> None:
    """Right-hand rail drawn with per-player color-coded text blocks.

    When *situation* and *actor_pid* are provided, ground-truth bins
    are shown alongside the actor's predictions with match/mismatch symbols.
    """
    ax_panel.clear()
    ax_panel.set_facecolor("#0f172a")
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")

    plist = [p for p in pids if p in preds]
    tags = role_tags if role_tags is not None else [f"Player {i + 1}" for i in range(len(plist))]

    gt_act = situation.get("gt_action_type_idx") if situation else None
    gt_ang = situation.get("gt_angle_bin") if situation else None
    gt_len = situation.get("gt_length_bin") if situation else None

    y_cursor = 0.97
    x_left = 0.06

    # title
    ax_panel.text(
        x_left, y_cursor, subtitle,
        transform=ax_panel.transAxes, ha="left", va="top",
        fontsize=7.5, color="#e2e8f0", fontweight="bold",
    )
    y_cursor -= 0.045

    # ground truth line
    if situation is not None:
        gt_txt = _gt_label(situation)
        ax_panel.text(
            x_left, y_cursor, gt_txt,
            transform=ax_panel.transAxes, ha="left", va="top",
            fontsize=6.5, color="#fbbf24", family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#422006",
                      edgecolor="#fbbf24", alpha=0.9, linewidth=0.6),
        )
        y_cursor -= 0.04

    # JS divergence
    if len(plist) >= 2:
        js = _js_divergence(
            preds[plist[0]]["action_type"],
            preds[plist[1]]["action_type"],
        )
        ax_panel.text(
            x_left, y_cursor,
            f"JS divergence = {js:.4f}  (0 = identical)",
            transform=ax_panel.transAxes, ha="left", va="top",
            fontsize=6.8, color="#94a3b8", family="monospace",
        )
        y_cursor -= 0.035

    # separator
    ax_panel.plot(
        [0.04, 0.96], [y_cursor, y_cursor],
        transform=ax_panel.transAxes,
        color="#475569", lw=0.6, clip_on=False,
    )
    y_cursor -= 0.015

    for i, pid in enumerate(plist):
        tag = tags[i] if i < len(tags) else f"Player {i + 1}"
        clr = _PANEL_COLORS[i % len(_PANEL_COLORS)]
        pr = preds[pid]
        nm = str(names.get(pid, ""))[:20] or str(pid)
        is_actor = (actor_pid is not None and pid == actor_pid)
        block = _format_player_prediction_block(
            pid, pr, nm, tag,
            gt_action_idx=gt_act if is_actor else None,
            gt_angle_bin=gt_ang if is_actor else None,
            gt_length_bin=gt_len if is_actor else None,
            is_actor=is_actor,
        )
        ax_panel.text(
            x_left, y_cursor, block,
            transform=ax_panel.transAxes, ha="left", va="top",
            fontsize=6.5, color=clr, family="monospace",
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b",
                      edgecolor=clr, alpha=0.85, linewidth=0.7),
        )
        y_cursor -= 0.46 if is_actor else 0.38
        if i < len(plist) - 1:
            ax_panel.plot(
                [0.04, 0.96], [y_cursor + 0.01, y_cursor + 0.01],
                transform=ax_panel.transAxes,
                color="#475569", lw=0.4, linestyle=":", clip_on=False,
            )
            y_cursor -= 0.01

    # ── "How to Read" guide at panel bottom ──
    guide_lines = (
        "HOW TO READ THIS VISUALIZATION\n"
        "\u25cb  White circle  = ball at this event\u2019s start location\n"
        "\u2500\u2500 Yellow trail  = line through prior event locations\n"
        "     (discrete StatsBomb events; NOT continuous ball motion)\n"
        "     Long jumps are normal (e.g. pass to far receiver).\n"
        "\u25cf  Blue dots     = teammate positions (freeze-frame)\n"
        "\u25cf  Red dots      = opponent positions (freeze-frame)\n"
        "- -> Dashed arrow  = model\u2019s predicted action\n"
        "     (direction & length from softmax over bins)\n"
        "     Label at tip  = most-likely action type\n"
        "\u2713/\u2717 in panel   = prediction matches / mismatches GT"
    )
    ax_panel.text(
        0.5, 0.01, guide_lines,
        transform=ax_panel.transAxes, ha="center", va="bottom",
        fontsize=5.4, color="#94a3b8", family="monospace",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a",
                  edgecolor="#334155", alpha=0.85, linewidth=0.5),
    )


_VIDEO_FPS = 30
_HOLD_SECONDS_PER_EVENT = 2.0


def _save_smooth_animation(
    fig,
    frame_fn,
    n_logical_frames: int,
    output_path: Path,
    logical_fps: float,
    dpi: int,
) -> Path:
    """Save animation as a smooth, seekable MP4.

    Instead of encoding at ``logical_fps`` (e.g. 0.5 = 1 frame every 2 s),
    each logical frame is repeated ``hold`` times at ``_VIDEO_FPS`` so that
    the resulting file is a standard 30-FPS video.  This makes seek, pause,
    and scrubbing buttery-smooth in any player.
    """
    hold_seconds = max(1.0 / max(logical_fps, 0.01), 0.5)
    hold = max(1, int(round(hold_seconds * _VIDEO_FPS)))
    total_real = n_logical_frames * hold
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _cur_logical = [-1]

    def _real_frame(ri: int):
        li = ri // hold
        if li != _cur_logical[0]:
            _cur_logical[0] = li
            frame_fn(li)

    anim = animation.FuncAnimation(
        fig, _real_frame, frames=total_real,
        interval=1000.0 / _VIDEO_FPS, repeat=False,
    )

    try:
        writer = animation.FFMpegWriter(
            fps=_VIDEO_FPS,
            codec="libx264",
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-crf", "20",
                "-preset", "medium",
                "-movflags", "+faststart",
            ],
        )
        mp4_path = output_path.with_suffix(".mp4")
        anim.save(str(mp4_path), writer=writer, dpi=dpi)
        plt.close(fig)
        return mp4_path
    except (OSError, ValueError, RuntimeError):
        pass

    try:
        writer_fallback = animation.FFMpegWriter(
            fps=_VIDEO_FPS,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        mp4_path = output_path.with_suffix(".mp4")
        anim.save(str(mp4_path), writer=writer_fallback, dpi=dpi)
        plt.close(fig)
        return mp4_path
    except (OSError, ValueError, RuntimeError):
        pass

    gif_path = output_path.with_suffix(".gif")
    simple_anim = animation.FuncAnimation(
        fig, frame_fn, frames=n_logical_frames,
        interval=1000.0 / max(logical_fps, 0.05), repeat=False,
    )
    simple_anim.save(str(gif_path), writer="pillow",
                     fps=max(logical_fps, 0.05), dpi=max(dpi - 20, 80))
    plt.close(fig)
    return gif_path


def _arrow_end(
    ang_prob: np.ndarray,
    len_prob: np.ndarray,
    x_m: float,
    y_m: float,
    pitch_length: float,
    pitch_width: float,
) -> Tuple[float, float]:
    """Expected displacement endpoint in meters (softmax over bins)."""
    ang = np.asarray(ang_prob, dtype=np.float64).ravel()
    if ang.shape[0] >= 9:
        p_dir = ang[:8].copy()
    else:
        p_dir = ang[: min(8, ang.shape[0])].copy()
    n_dir = int(p_dir.shape[0])
    # s = P(has a direction).  Scale expected length by s so that
    # high no-angle probability naturally shrinks the arrow toward zero.
    s = float(p_dir.sum())
    if s < 1e-9:
        return x_m, y_m
    p_dir = p_dir / s
    angles = (np.arange(n_dir) + 0.5) * (2 * np.pi / 8.0)
    ex = float(np.sum(p_dir * np.cos(angles)))
    ey = float(np.sum(p_dir * np.sin(angles)))
    n_len = min(len(len_prob), len(_LENGTH_MID_NORM))
    exp_len = float(np.sum(len_prob[:n_len] * _LENGTH_MID_NORM[:n_len])) * s
    dx = exp_len * ex
    dy = exp_len * ey
    return x_m + dx * pitch_length, y_m + dy * pitch_width


def _cap_arrow_display(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    max_len_m: float = _DISPLAY_ARROW_MAX_M,
) -> Tuple[float, float, bool]:
    """Return endpoint clipped to *max_len_m* from (x0,y0).  Third value = was clipped."""
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.hypot(dx, dy))
    if dist <= max_len_m or dist < 1e-9:
        return x1, y1, False
    s = max_len_m / dist
    return x0 + dx * s, y0 + dy * s, True


def _draw_ball_trail(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    zorder: int = 2,
) -> None:
    """Past ball positions as small markers + faint connector (easier than a thick polyline)."""
    if len(xs) < 2:
        return
    ax.plot(
        xs, ys,
        color="#fef9c3", lw=1.0, alpha=0.28, linestyle="-", zorder=zorder,
    )
    ax.scatter(
        xs[:-1], ys[:-1],
        s=28, c="#facc15", edgecolors="#422006", linewidths=0.4,
        alpha=0.75, zorder=zorder + 1,
    )


def _get_event_context_positions(
    g,
    local_ev_idx: int,
    ball_x: float,
    ball_y: float,
    pl: float,
    pw: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Extract teammate/opponent (x, y) from freeze-frame player nodes.

    Returns ``(teammates_xy, opponents_xy)`` in pitch coordinates.
    Works for both split-context and generic-context graph layouts.
    """
    player_feat = g["player"].x
    ctx_tm: List[int] = []
    ctx_opp: List[int] = []
    ctx_generic: List[int] = []

    for et in g.edge_types:
        if et[0] != "player":
            continue
        rel = et[1]
        if rel not in ("context_for", "context_for_tm", "context_for_opp"):
            continue
        ei = g[et].edge_index
        mask = ei[1] == local_ev_idx
        nodes = ei[0][mask].tolist()
        if rel == "context_for_tm":
            ctx_tm.extend(nodes)
        elif rel == "context_for_opp":
            ctx_opp.extend(nodes)
        else:
            ctx_generic.extend(nodes)

    def _to_xy(node_idx: int) -> Tuple[float, float]:
        f = player_feat[node_idx]
        return (ball_x + float(f[2]) * pl, ball_y + float(f[3]) * pw)

    teammates: List[Tuple[float, float]] = []
    opponents: List[Tuple[float, float]] = []

    if ctx_tm or ctx_opp:
        teammates = [_to_xy(n) for n in ctx_tm]
        opponents = [_to_xy(n) for n in ctx_opp]
    else:
        for n in ctx_generic:
            xy = _to_xy(n)
            if float(player_feat[n][1]) > 0.5:
                teammates.append(xy)
            else:
                opponents.append(xy)
    return teammates, opponents


def _draw_pitch(ax, pitch_length: float, pitch_width: float) -> None:
    """Full FIFA-standard pitch markings for spatial context."""
    line = "#e2e8f0"
    faint = 0.50
    ax.set_facecolor("#15251a")

    ax.add_patch(plt.Rectangle(
        (0, 0), pitch_length, pitch_width,
        fc="#2d6a3e", ec=line, lw=1.4, alpha=1.0,
    ))

    mid_x = pitch_length / 2
    mid_y = pitch_width / 2
    # halfway line
    ax.plot([mid_x, mid_x], [0, pitch_width], color=line, lw=1.0, alpha=faint)
    # center circle (9.15 m radius)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(mid_x + 9.15 * np.cos(t), mid_y + 9.15 * np.sin(t),
            color=line, lw=0.8, alpha=faint)
    ax.scatter([mid_x], [mid_y], s=18, c=line, zorder=3, alpha=0.7)

    # penalty areas (16.5 m from goal line, 40.3 m wide)
    pa_w = 40.32
    pa_d = 16.5
    pa_top = mid_y - pa_w / 2
    for gx in (0.0, pitch_length):
        sign = 1 if gx == 0.0 else -1
        x0 = gx
        x1 = gx + sign * pa_d
        ax.plot([x0, x1, x1, x0], [pa_top, pa_top, pa_top + pa_w, pa_top + pa_w],
                color=line, lw=0.9, alpha=faint)

    # goal areas (5.5 m from goal line, 18.32 m wide)
    ga_w = 18.32
    ga_d = 5.5
    ga_top = mid_y - ga_w / 2
    for gx in (0.0, pitch_length):
        sign = 1 if gx == 0.0 else -1
        x0 = gx
        x1 = gx + sign * ga_d
        ax.plot([x0, x1, x1, x0], [ga_top, ga_top, ga_top + ga_w, ga_top + ga_w],
                color=line, lw=0.7, alpha=faint * 0.8)

    # penalty spots (11 m from goal line)
    ax.scatter([11.0, pitch_length - 11.0], [mid_y, mid_y],
              s=14, c=line, zorder=3, alpha=0.6)

    # goals (centered, 7.32 m wide, drawn as thick bars)
    goal_hw = 7.32 / 2
    for gx in (0.0, pitch_length):
        ax.plot([gx, gx], [mid_y - goal_hw, mid_y + goal_hw],
                color="#fef9c3", lw=3.0, alpha=0.7, solid_capstyle="round", zorder=2)

    # corner arcs (1 m radius quarter-circles)
    arc_t = np.linspace(0, np.pi / 2, 30)
    for cx, cy, a0 in [
        (0, 0, 0), (0, pitch_width, -np.pi / 2),
        (pitch_length, 0, np.pi / 2), (pitch_length, pitch_width, np.pi),
    ]:
        ax.plot(cx + np.cos(arc_t + a0), cy + np.sin(arc_t + a0),
                color=line, lw=0.6, alpha=faint * 0.7)

    # direction-of-play arrow
    arr_y = pitch_width + 1.6
    ax.annotate(
        "", xy=(mid_x + 12, arr_y), xytext=(mid_x - 12, arr_y),
        arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.2,
                        mutation_scale=10),
    )
    ax.text(mid_x, arr_y + 0.4, "Direction of play \u2192",
            ha="center", va="bottom", fontsize=7, color="#94a3b8")

    ax.set_xlim(-3.5, pitch_length + 3.5)
    ax.set_ylim(-3.5, pitch_width + 4.5)
    ax.set_aspect("equal")
    ax.axis("off")


def render_possession_animation(
    config: Config,
    graph_index: int,
    player_ids: Sequence[int],
    output_path: Path,
    fps: float = _DEFAULT_ANIMATION_FPS,
    dpi: int = 120,
    *,
    graphs: Optional[List] = None,
) -> Path:
    """Build MP4 (or GIF if ffmpeg unavailable) of one possession.

    Parameters
    ----------
    graphs : list[HeteroData], optional
        Pre-loaded possession graphs.  When *None* they are loaded from
        ``config.data.output_dir / config.graphs_filename``.
    """
    out_dir = Path(config.data.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if graphs is None:
        graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
        validate_graph_config_match(graphs, config.graph.split_context_edges)

    meta = pd.read_parquet(out_dir / "event_metadata.parquet")
    event_features = np.load(out_dir / "event_features.npy")
    if len(meta) != len(event_features):
        warnings.warn(
            f"event_metadata rows ({len(meta)}) != event_features rows ({len(event_features)}); "
            "time/observed_action labels may be wrong for some global_idx.",
            stacklevel=2,
        )

    emb_dir = Path(config.inference.embedding_output_dir)
    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    comp = SituationComparator(model, device, graphs, Z, player_info, event_features)
    pids = [int(p) for p in player_ids][:2]
    if len(pids) < 2:
        raise ValueError("Need at least two player IDs for overlay comparison.")
    if len(set(pids)) < 2:
        raise ValueError("Need two distinct player IDs (same ID twice is not a comparison).")
    result = comp.compare_possession(graph_index, pids, meta)
    if result is None or not result.get("events"):
        raise RuntimeError("compare_possession returned no events (bad graph index?).")

    pl, pw = float(PITCH_LENGTH), float(PITCH_WIDTH)
    arrow_colors = ["#22d3ee", "#fb923c"]
    tm_color, opp_color = "#60a5fa", "#f87171"
    names = {int(r["player_id"]): str(r.get("player_name", "")) for _, r in player_info.iterrows()}

    g = graphs[graph_index]
    events = result["events"]
    traj_x = [float(ev["situation"]["location"][0]) for ev in events]
    traj_y = [float(ev["situation"]["location"][1]) for ev in events]

    fig = plt.figure(figsize=(13.5, 7.5))
    fig.patch.set_facecolor("#0f1a12")
    gs = GridSpec(1, 2, width_ratios=[4.25, 1.0], wspace=0.14)
    ax = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])
    title = (
        f"Pipeline: {config.tag or 'baseline'}  |  graph_index={graph_index}  |  "
        f"pos_key={result.get('pos_key')}"
    )
    fig.suptitle(title, fontsize=10, color="#e2e8f0")

    p_names = {pid: names.get(pid, str(pid))[:20] for pid in pids}

    def _frame(fi: int):
        ax.clear()
        _draw_pitch(ax, pl, pw)
        ax.invert_yaxis()
        ev = events[fi]
        sit = ev["situation"]
        gx = float(sit["location"][0])
        gy = float(sit["location"][1])

        # freeze-frame players
        tm_xy, opp_xy = _get_event_context_positions(g, fi, gx, gy, pl, pw)
        if tm_xy:
            ax.scatter(
                [p[0] for p in tm_xy], [p[1] for p in tm_xy],
                s=100, c=tm_color, marker="o", alpha=0.75,
                edgecolors="white", linewidths=0.6, zorder=3,
            )
        if opp_xy:
            ax.scatter(
                [p[0] for p in opp_xy], [p[1] for p in opp_xy],
                s=100, c=opp_color, marker="o", alpha=0.75,
                edgecolors="white", linewidths=0.6, zorder=3,
            )

        # ball trail + current ball
        if fi > 0:
            _draw_ball_trail(ax, traj_x[: fi + 1], traj_y[: fi + 1])
        ax.scatter([gx], [gy], s=280, c="white", edgecolors="#0f172a",
                   zorder=5, linewidths=1.5)
        ax.text(gx, gy - 2.2, "BALL", ha="center", va="top",
                fontsize=5.5, color="white", fontweight="bold", alpha=0.7)

        # ── prominent event title (center-top of pitch) ──
        obs = sit.get("observed_action", "")
        n_ev = len(events)
        event_title = f"EVENT {fi + 1}/{n_ev}  \u2014  {obs}"
        ax.text(
            0.50, 0.99, event_title,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#fbbf24", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a",
                      edgecolor="#fbbf24", alpha=0.92, linewidth=1.2),
            zorder=10,
        )

        # ── structured event info (top-left) ──
        gt_line = _gt_label(sit)
        third = sit.get("third", "")
        lane = sit.get("lane", "")
        pres = sit.get("under_pressure")
        minute = sit.get("minute", -1)
        second = sit.get("second", -1)
        time_str = f"{minute}:{second:02d}" if minute >= 0 else "?"
        loc = sit.get("location", ("?", "?"))
        info_lines = [
            f"Time: {time_str}   Loc: ({loc[0]}, {loc[1]})",
            f"Zone: {third} / {lane}" + ("   [PRESS]" if pres else ""),
            gt_line,
        ]
        ax.text(
            0.01, 0.99,
            "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="white", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a",
                      edgecolor="#334155", alpha=0.85),
        )

        # ── player name tags (top-right of pitch) ──
        for k, pid in enumerate(pids):
            tag_txt = f"P{k + 1}: {p_names[pid]}"
            ax.text(
                0.99, 0.99 - k * 0.05,
                tag_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=arrow_colors[k % len(arrow_colors)],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a",
                          edgecolor=arrow_colors[k % len(arrow_colors)],
                          alpha=0.8, linewidth=0.8),
            )

        # ── prediction arrows (dashed) with action-type labels ──
        preds = ev["predictions"]
        role_tags_cf = [f"P1 \u2014 {p_names[pids[0]]}", f"P2 \u2014 {p_names[pids[1]]}"]
        drawn = [(k, pid) for k, pid in enumerate(pids) if pid in preds]
        n_draw = len(drawn)
        for i, (k, pid) in enumerate(drawn):
            pr = preds[pid]
            off_x = (
                (i - (n_draw - 1) / 2.0) * _ARROW_ORIGIN_SEP_M if n_draw > 1 else 0.0
            )
            x0, y0 = gx + off_x, gy
            raw_x, raw_y = _arrow_end(
                pr["angle_bin"], pr["length_bin"],
                gx, gy, pl, pw,
            )
            ddx, ddy = raw_x - gx, raw_y - gy
            x1_raw, y1_raw = x0 + ddx, y0 + ddy
            x2, y2, clipped = _cap_arrow_display(x0, y0, x1_raw, y1_raw)
            clr = arrow_colors[k % len(arrow_colors)]
            ax.annotate(
                "",
                xy=(x2, y2), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=clr,
                    lw=2.2, shrinkA=0, shrinkB=0,
                    mutation_scale=12, linestyle="--",
                ),
                zorder=6,
            )
            act_lbl = _top_action_label(pr, max_chars=10)
            cap = "*" if clipped else ""
            ax.text(
                x2 + 0.7, y2 + 0.7,
                f"P{k + 1}: {act_lbl}{cap}",
                color=clr, fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f172a",
                          edgecolor="none", alpha=0.6),
            )

        # ── right-hand comparison panel ──
        _draw_comparison_panel(
            ax_panel, pids, preds, names,
            role_tags=role_tags_cf,
            subtitle="Counterfactual: same state, two z_p",
            situation=sit,
        )

        # ── legend (all visual elements explained) ──
        legend_items = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="white", markersize=8,
                   markeredgecolor="#0f172a", markeredgewidth=1,
                   label="Ball (current position)"),
            Line2D([0], [0], marker="o", color="#fef9c3",
                   markerfacecolor="#facc15", markersize=6,
                   markeredgecolor="#422006", markeredgewidth=0.4,
                   lw=1, alpha=0.7,
                   label="Prior event locations (may jump far)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=tm_color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label="Teammate (360 freeze-frame)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=opp_color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label="Opponent (360 freeze-frame)"),
            Line2D([0], [0], color=arrow_colors[0], lw=2, linestyle="--",
                   label="P1 predicted action (dir+len)"),
            Line2D([0], [0], color=arrow_colors[1], lw=2, linestyle="--",
                   label="P2 predicted action (dir+len)"),
        ]
        ax.legend(
            handles=legend_items, loc="upper left",
            bbox_to_anchor=(0.0, 0.85),
            fontsize=6, framealpha=0.85,
            facecolor="#0f172a", edgecolor="#334155",
            labelcolor="white", handlelength=2.0,
        )

        # ── progress bar ──
        _draw_progress_bar(ax, fi, len(events))

    n = len(events)
    return _save_smooth_animation(fig, _frame, n, output_path, fps, dpi)


def _find_top1_substitutes(
    Z: np.ndarray,
    player_info: pd.DataFrame,
    actor_pids: Sequence[int],
    gender_map: Optional[dict] = None,
) -> dict:
    """For each unique on-ball actor, find the top-1 cosine-similar player.

    Returns ``{actor_pid: substitute_pid}``.  Actors without an embedding
    row are silently skipped.
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    pids = player_info["player_id"].values
    pid_to_idx = {int(p): i for i, p in enumerate(pids)}
    sim = _cos_sim(Z)
    np.fill_diagonal(sim, -np.inf)

    gender_arr = None
    if gender_map:
        gender_arr = np.array([gender_map.get(int(p), "unknown") for p in pids])

    result: dict = {}
    for apid in dict.fromkeys(actor_pids):
        idx = pid_to_idx.get(int(apid))
        if idx is None:
            continue
        scores = sim[idx].copy()
        if gender_arr is not None:
            mask = gender_arr != gender_arr[idx]
            scores[mask] = -np.inf
        best = int(np.argmax(scores))
        result[int(apid)] = int(pids[best])
    return result


def find_consecutive_possessions(
    graphs: List,
    start_gidx: int,
    num_possessions: int,
) -> List[int]:
    """Return up to *num_possessions* graph indices from the same match,
    starting at *start_gidx* and continuing in possession-number order.

    The starting graph is always included as the first element.  Subsequent
    graphs are those from the same ``match_id`` whose ``pos_key[1]``
    (possession_number) is strictly greater, taken in ascending order.
    """
    g0 = graphs[start_gidx]
    mid = getattr(g0, "match_id", None)
    pn0 = getattr(g0, "pos_key", (None, None, None))[1]
    if mid is None or pn0 is None:
        return [start_gidx]

    candidates: List[Tuple[int, int]] = []   # (possession_number, graph_idx)
    for i, g in enumerate(graphs):
        if getattr(g, "match_id", None) != mid:
            continue
        pk = getattr(g, "pos_key", (None, None, None))
        pn = pk[1]
        if pn is None:
            continue
        if pn > pn0:
            candidates.append((pn, i))
    candidates.sort(key=lambda x: x[0])

    result = [start_gidx]
    for _, gidx in candidates:
        if len(result) >= num_possessions:
            break
        result.append(gidx)
    return result


def render_dynamic_substitute_animation(
    config: Config,
    graph_indices: Sequence[int],
    output_path: Path,
    fps: float = _DEFAULT_ANIMATION_FPS,
    dpi: int = 120,
    *,
    graphs: Optional[List] = None,
    gender_map: Optional[dict] = None,
) -> Path:
    """Animate one or more consecutive possessions with dynamic substitutes.

    For every event the on-ball actor is read from the graph.  That actor's
    top-1 cosine-similarity neighbor (from the pipeline's embeddings) is
    overlaid as the "substitute".  When the on-ball actor changes, the
    substitute automatically switches.  When a new possession begins the ball
    trail resets and a header shows which possession we are in.

    Parameters
    ----------
    graph_indices : sequence of int
        One or more graph indices to animate in order.  Typically produced
        by :func:`find_consecutive_possessions`.

    Returns the saved file path (.mp4 or .gif).
    """
    if not graph_indices:
        raise ValueError("graph_indices must not be empty.")

    out_dir = Path(config.data.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if graphs is None:
        graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
        validate_graph_config_match(graphs, config.graph.split_context_edges)

    meta = pd.read_parquet(out_dir / "event_metadata.parquet")
    event_features = np.load(out_dir / "event_features.npy")
    if len(meta) != len(event_features):
        warnings.warn(
            f"event_metadata rows ({len(meta)}) != event_features rows ({len(event_features)}); "
            "time/observed_action labels may be wrong for some global_idx.",
            stacklevel=2,
        )

    emb_dir = Path(config.inference.embedding_output_dir)
    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # ── collect actors across ALL possessions ──
    all_actor_lists: List[List[int]] = []
    all_unique_actors: List[int] = []
    for gidx in graph_indices:
        actors = graphs[gidx].event_player_ids.tolist()
        all_actor_lists.append(actors)
        for a in actors:
            if a >= 0 and a not in all_unique_actors:
                all_unique_actors.append(a)
    if not all_unique_actors:
        raise RuntimeError("No valid on-ball actors across the selected possessions.")

    print("Finding top-1 substitute for each on-ball actor …")
    sub_map = _find_top1_substitutes(Z, player_info, all_unique_actors, gender_map)
    all_pids = list(set(all_unique_actors) | set(sub_map.values()))

    names = {int(r["player_id"]): str(r.get("player_name", ""))
             for _, r in player_info.iterrows()}
    for apid in all_unique_actors:
        spid = sub_map.get(apid)
        if spid is not None:
            print(f"  {names.get(apid, str(apid)):30s} → {names.get(spid, str(spid))}")

    # ── run compare_possession for each graph ──
    comp = SituationComparator(model, device, graphs, Z, player_info, event_features)

    # flat lists that the animation iterates over
    flat_events: List[dict] = []
    flat_actors: List[int] = []
    # per-frame metadata: which possession segment this frame belongs to
    frame_poss_idx: List[int] = []
    # boundary: first flat-frame index of each possession
    poss_start_frame: List[int] = []
    poss_labels: List[str] = []
    seg_graphs: List = []
    # StatsBomb possession_team_id per segment (pos_key[2]) — for turnover cues
    seg_team_ids: List[Optional[int]] = []

    for seg_i, gidx in enumerate(graph_indices):
        pk = getattr(graphs[gidx], "pos_key", None)
        result = comp.compare_possession(gidx, all_pids, meta)
        if result is None or not result.get("events"):
            print(f"  Warning: graph {gidx} returned no events; skipping.")
            continue
        poss_start_frame.append(len(flat_events))
        poss_labels.append(f"(graph={gidx}, key={pk})")
        seg_graphs.append(graphs[gidx])
        if pk is not None and len(pk) >= 3:
            seg_team_ids.append(int(pk[2]))
        else:
            seg_team_ids.append(None)
        actors = all_actor_lists[seg_i]
        for ev_i, ev in enumerate(result["events"]):
            flat_events.append(ev)
            actor = int(actors[ev_i]) if ev_i < len(actors) else -1
            flat_actors.append(actor)
            frame_poss_idx.append(len(poss_start_frame) - 1)

    if not flat_events:
        raise RuntimeError("All selected possessions returned no events.")
    # sentinel for slicing convenience
    poss_start_frame.append(len(flat_events))

    print(f"Total frames: {len(flat_events)} across {len(poss_labels)} possession(s)")

    # ── pre-compute trajectories per possession for ball trail ──
    poss_traj_x: List[List[float]] = []
    poss_traj_y: List[List[float]] = []
    for pi in range(len(poss_labels)):
        s, e = poss_start_frame[pi], poss_start_frame[pi + 1]
        poss_traj_x.append([float(flat_events[f]["situation"]["location"][0]) for f in range(s, e)])
        poss_traj_y.append([float(flat_events[f]["situation"]["location"][1]) for f in range(s, e)])

    pl, pw = float(PITCH_LENGTH), float(PITCH_WIDTH)
    actor_color = "#22d3ee"
    sub_color = "#fb923c"
    tm_color, opp_color = "#60a5fa", "#f87171"

    n_seg_anim = len(poss_labels)
    fig = plt.figure(figsize=(13.5, 7.5))
    fig.patch.set_facecolor("#0f1a12")
    gs = GridSpec(1, 2, width_ratios=[4.25, 1.0], wspace=0.14)
    ax = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])
    supt = (
        f"Dynamic Substitute Overlay  |  {config.tag or 'baseline'}"
        + (
            f"  |  {n_seg_anim} possession segments (ball may change hands)"
            if n_seg_anim > 1
            else ""
        )
    )
    fig.suptitle(supt, fontsize=10, color="#e2e8f0")

    def _frame(fi: int):
        ax.clear()
        _draw_pitch(ax, pl, pw)
        ax.invert_yaxis()

        pi = frame_poss_idx[fi]
        local_fi = fi - poss_start_frame[pi]
        tx = poss_traj_x[pi]
        ty = poss_traj_y[pi]

        ev = flat_events[fi]
        sit = ev["situation"]
        gx, gy = float(sit["location"][0]), float(sit["location"][1])

        cur_graph = seg_graphs[pi]
        tm_xy, opp_xy = _get_event_context_positions(
            cur_graph, local_fi, gx, gy, pl, pw,
        )
        if tm_xy:
            ax.scatter(
                [p[0] for p in tm_xy], [p[1] for p in tm_xy],
                s=100, c=tm_color, marker="o", alpha=0.75,
                edgecolors="white", linewidths=0.6, zorder=3,
            )
        if opp_xy:
            ax.scatter(
                [p[0] for p in opp_xy], [p[1] for p in opp_xy],
                s=100, c=opp_color, marker="o", alpha=0.75,
                edgecolors="white", linewidths=0.6, zorder=3,
            )

        if local_fi > 0:
            _draw_ball_trail(ax, tx[: local_fi + 1], ty[: local_fi + 1])

        ax.scatter([gx], [gy], s=280, c="white", edgecolors="#0f172a",
                   zorder=5, linewidths=1.5)
        ax.text(gx, gy - 2.2, "BALL", ha="center", va="top",
                fontsize=5.5, color="white", fontweight="bold", alpha=0.7)

        actor = flat_actors[fi]
        sub = sub_map.get(actor)
        actor_nm = names.get(actor, str(actor))[:22]
        sub_nm = names.get(sub, str(sub))[:22] if sub else "?"

        # ── prominent event title (center-top of pitch) ──
        obs = sit.get("observed_action", "")
        n_seg = len(poss_labels)
        n_total = len(flat_events)
        seg_tid = seg_team_ids[pi] if pi < len(seg_team_ids) else None
        event_title = f"EVENT {fi + 1}/{n_total}  \u2014  {obs}"
        ax.text(
            0.50, 0.99, event_title,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#fbbf24", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a",
                      edgecolor="#fbbf24", alpha=0.92, linewidth=1.2),
            zorder=10,
        )

        # ── possession boundary (multi-segment only): turnover vs same team ──
        if n_seg > 1 and local_fi == 0 and pi > 0:
            prev_tid = seg_team_ids[pi - 1] if pi - 1 < len(seg_team_ids) else None
            ax.text(
                0.50, 0.825,
                _possession_change_caption(prev_tid, seg_tid),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.2, color="#f5f3ff", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#4c1d95",
                          edgecolor="#a78bfa", alpha=0.96, linewidth=1.1),
                zorder=11,
            )

        # ── structured event info (top-left) ──
        gt_line = _gt_label(sit)
        third = sit.get("third", "")
        lane = sit.get("lane", "")
        pres = sit.get("under_pressure")
        minute = sit.get("minute", -1)
        second = sit.get("second", -1)
        time_str = f"{minute}:{second:02d}" if minute >= 0 else "?"
        loc = sit.get("location", ("?", "?"))
        team_note = (
            f"Controlling team ID: {seg_tid}"
            if seg_tid is not None
            else "Controlling team ID: (unknown)"
        )
        info_lines = [
            f"Possession {pi + 1}/{n_seg}   {team_note}   Time: {time_str}   Loc: ({loc[0]}, {loc[1]})",
            f"Zone: {third} / {lane}" + ("   [PRESS]" if pres else ""),
            gt_line,
            f"On-ball: {actor_nm}  \u2192  Sub: {sub_nm}",
        ]
        ax.text(
            0.01, 0.99,
            "\n".join(info_lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="white", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a",
                      edgecolor="#334155", alpha=0.85),
        )

        # ── player name tags (top-right) ──
        ax.text(
            0.99, 0.99,
            f"A (on-ball): {actor_nm}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=actor_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a",
                      edgecolor=actor_color, alpha=0.8, linewidth=0.8),
        )
        ax.text(
            0.99, 0.94,
            f"B (substitute): {sub_nm}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=sub_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a",
                      edgecolor=sub_color, alpha=0.8, linewidth=0.8),
        )

        # ── prediction arrows (dashed) with action-type labels ──
        preds = ev["predictions"]
        pair_meta = [
            (actor, actor_color, "On-ball", "A"),
            (sub, sub_color, "Substitute", "B"),
        ]
        drawn_sub = [
            (orig_i, pid, clr, short)
            for orig_i, (pid, clr, _tag, short) in enumerate(pair_meta)
            if pid is not None and pid in preds
        ]
        n_sub = len(drawn_sub)
        for i, (_orig_i, pid, clr, short) in enumerate(drawn_sub):
            pr = preds[pid]
            off_x = (
                (i - (n_sub - 1) / 2.0) * _ARROW_ORIGIN_SEP_M if n_sub > 1 else 0.0
            )
            x0, y0 = gx + off_x, gy
            raw_x, raw_y = _arrow_end(pr["angle_bin"], pr["length_bin"], gx, gy, pl, pw)
            ddx, ddy = raw_x - gx, raw_y - gy
            x1_raw, y1_raw = x0 + ddx, y0 + ddy
            x2, y2, clipped = _cap_arrow_display(x0, y0, x1_raw, y1_raw)
            act_lbl = _top_action_label(pr, max_chars=10)
            cap = "*" if clipped else ""
            ax.annotate(
                "", xy=(x2, y2), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=clr, lw=2.2,
                                shrinkA=0, shrinkB=0, mutation_scale=12,
                                linestyle="--"),
                zorder=6,
            )
            ax.text(
                x2 + 0.7, y2 + 0.7,
                f"{short}: {act_lbl}{cap}",
                color=clr, fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f172a",
                          edgecolor="none", alpha=0.6),
            )

        # ── comparison panel ──
        panel_pids = [pid for _oi, pid, _c, _s in drawn_sub]
        panel_tags = []
        for _oi, pid, _c, short in drawn_sub:
            nm = names.get(pid, str(pid))[:20]
            if short == "A":
                panel_tags.append(f"A (on-ball) \u2014 {nm}")
            else:
                panel_tags.append(f"B (substitute) \u2014 {nm}")
        if panel_pids:
            panel_sub = (
                f"Segment {pi + 1}/{n_seg}  |  On-ball vs top-1 cosine neighbor"
                if n_seg > 1
                else "On-ball vs top-1 cosine neighbor"
            )
            _draw_comparison_panel(
                ax_panel, panel_pids, preds, names,
                role_tags=panel_tags,
                subtitle=panel_sub,
                situation=sit,
                actor_pid=actor,
            )
        else:
            ax_panel.clear()
            ax_panel.set_facecolor("#0f172a")
            ax_panel.axis("off")

        # ── legend (all visual elements explained) ──
        legend_items = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="white", markersize=8,
                   markeredgecolor="#0f172a", markeredgewidth=1,
                   label="Ball (current position)"),
            Line2D([0], [0], marker="o", color="#fef9c3",
                   markerfacecolor="#facc15", markersize=6,
                   markeredgecolor="#422006", markeredgewidth=0.4,
                   lw=1, alpha=0.7,
                   label="Prior event locations (may jump far)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=tm_color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label="Teammate (360 freeze-frame)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=opp_color, markersize=7,
                   markeredgecolor="white", markeredgewidth=0.5,
                   label="Opponent (360 freeze-frame)"),
            Line2D([0], [0], color=actor_color, lw=2, linestyle="--",
                   label="A: on-ball predicted (dir+len)"),
            Line2D([0], [0], color=sub_color, lw=2, linestyle="--",
                   label="B: substitute predicted (dir+len)"),
        ]
        ax.legend(
            handles=legend_items, loc="upper left",
            bbox_to_anchor=(0.0, 0.82),
            fontsize=6, framealpha=0.85,
            facecolor="#0f172a", edgecolor="#334155",
            labelcolor="white", handlelength=2.0,
        )

        # ── progress bar (segment index + highlight first event after boundary) ──
        _draw_progress_bar(
            ax, fi, len(flat_events),
            possession_segment=(pi + 1, n_seg) if n_seg > 1 else None,
            segment_boundary_highlight=(
                n_seg > 1 and local_fi == 0 and pi > 0
            ),
        )

    n = len(flat_events)
    return _save_smooth_animation(fig, _frame, n, output_path, fps, dpi)


def default_players_from_graph(graphs: List, graph_index: int) -> List[int]:
    """First two distinct on-ball player IDs in chronological event order."""
    g = graphs[graph_index]
    seen: List[int] = []
    for pid in g.event_player_ids.tolist():
        if pid < 0:
            continue
        if pid not in seen:
            seen.append(int(pid))
        if len(seen) >= 2:
            return [seen[0], seen[1]]
    if len(seen) == 1:
        raise ValueError(
            "Only one distinct actor on this graph; pass two --players with embeddings."
        )
    raise ValueError("No actor player_ids on this graph; pass --players explicitly.")


def find_graph_index(
    graphs: List,
    match_id: int,
    possession_number: int,
    possession_team_id: int,
) -> Optional[int]:
    key = (match_id, possession_number, possession_team_id)
    for i, g in enumerate(graphs):
        if getattr(g, "pos_key", None) == key:
            return i
    return None


def list_possession_summaries(graphs: List, limit: int = 30) -> None:
    print(f"Total graphs: {len(graphs):,}  (showing first {limit})")
    print(f"{'idx':>5}  {'n_ev':>4}  {'match_id':>10}  pos_key")
    for i, g in enumerate(graphs[:limit]):
        T = g["event"].x.shape[0]
        pk = getattr(g, "pos_key", None)
        mid = getattr(g, "match_id", None)
        print(f"{i:5d}  {T:4d}  {mid!s:>10}  {pk!s}")


_MAX_POSSESSIONS = 5


def run_possession_animation_cli(
    config: Config,
    graph_index: Optional[int],
    match_id: Optional[int],
    possession_number: Optional[int],
    possession_team_id: Optional[int],
    players_csv: Optional[str],
    output: Path,
    fps: float,
    list_only: bool,
    *,
    random_possession: bool = False,
    dynamic_substitute: bool = False,
    random_seed: Optional[int] = None,
    num_possessions: int = 1,
) -> None:
    num_possessions = max(1, min(num_possessions, _MAX_POSSESSIONS))

    out_dir = Path(config.data.output_dir)
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)

    if list_only:
        list_possession_summaries(graphs, limit=40)
        return

    gidx: Optional[int] = graph_index

    if random_possession:
        rng = np.random.default_rng(random_seed)
        gidx = int(rng.integers(0, len(graphs)))
        print(f"Randomly selected graph index: {gidx}")

    if gidx is None and match_id is not None and possession_number is not None and possession_team_id is not None:
        gidx = find_graph_index(graphs, match_id, possession_number, possession_team_id)
        if gidx is None:
            raise SystemExit(
                f"No graph with pos_key=({match_id}, {possession_number}, {possession_team_id})"
            )
    if gidx is None:
        raise SystemExit(
            "Set --graph-index, (--match-id, --possession-number, --possession-team-id), or --random."
        )

    if dynamic_substitute:
        graph_indices = find_consecutive_possessions(graphs, gidx, num_possessions)
        print(f"Possessions to animate: {len(graph_indices)}  (graph indices: {graph_indices})")

        gender_map = None
        try:
            from ..phase6_inference import build_gender_map
            gender_map = build_gender_map(
                config.data.output_dir, config.data.statsbomb_base_path,
            )
        except (FileNotFoundError, ImportError):
            pass
        path = render_dynamic_substitute_animation(
            config, graph_indices, output,
            fps=fps, graphs=graphs, gender_map=gender_map,
        )
        print(f"Saved: {path.resolve()}")
        return

    if players_csv:
        pids = [int(x.strip()) for x in players_csv.split(",") if x.strip()][:2]
    else:
        pids = default_players_from_graph(graphs, gidx)

    if len(pids) < 2:
        raise SystemExit("Need at least two --players (comma-separated).")
    if len(set(pids)) < 2:
        raise SystemExit("Need two distinct player IDs for the overlay.")

    path = render_possession_animation(config, gidx, pids, output, fps=fps, graphs=graphs)
    print(f"Saved: {path.resolve()}")
