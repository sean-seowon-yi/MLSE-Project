"""
graph/visible_area.py — Visible-area polygon utilities.

Converts the StatsBomb `visible_area` polygon into:
  - scalar global features (area, centroid, aspect ratio)
  - arc-length resampled boundary nodes for the graph
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from config import PITCH_LEN, PITCH_WID, PITCH_AREA, N_CAM_PTS


# ── Public API ─────────────────────────────────────────────────────────────────

def global_features(visible_area) -> np.ndarray:
    """
    Compute 4 scalar features from a visible_area polygon.

    Returns
    -------
    feats : (4,) float32
        [visible_area_pct, vis_centroid_x, vis_centroid_y, vis_aspect_ratio]
    """
    feats = np.zeros(4, dtype=np.float32)
    pts = _parse(visible_area)
    if pts is None:
        return feats

    feats[0] = float(np.clip(_shoelace(pts) / PITCH_AREA, 0.0, 1.0))
    c = pts.mean(axis=0)
    feats[1] = float(np.clip(c[0] / PITCH_LEN, 0.0, 1.0))
    feats[2] = float(np.clip(c[1] / PITCH_WID, 0.0, 1.0))
    bb_w = pts[:, 0].max() - pts[:, 0].min()
    bb_h = pts[:, 1].max() - pts[:, 1].min() + 1e-6
    feats[3] = float(np.clip(bb_w / bb_h, 0.0, 5.0) / 5.0)
    return feats


def resample_boundary_nodes(
    visible_area, n_pts: int = N_CAM_PTS
) -> Optional[np.ndarray]:
    """
    Resample the visible_area polygon to exactly `n_pts` evenly-spaced
    boundary points using arc-length parameterisation.

    Returns (n_pts, 2) float32 array in pitch coordinates,
    or None if the polygon is missing / degenerate.
    """
    pts = _parse(visible_area)
    if pts is None:
        return None

    # Close polygon if needed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    if total < 1e-6:
        return None

    sample_arc = np.linspace(0.0, total, n_pts, endpoint=False)
    sampled = np.zeros((n_pts, 2), dtype=np.float32)
    for k, s in enumerate(sample_arc):
        idx = int(np.searchsorted(arc, s, side="right")) - 1
        idx = np.clip(idx, 0, len(seg_len) - 1)
        t = (s - arc[idx]) / (seg_len[idx] + 1e-8)
        sampled[k] = pts[idx] + t * diffs[idx]
    return sampled


def fallback_boundary_nodes(n_pts: int = N_CAM_PTS) -> np.ndarray:
    """Return a default (pitch-centre) boundary when visible_area is absent."""
    return np.full((n_pts, 2), [PITCH_LEN / 2, PITCH_WID / 2], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _parse(visible_area) -> Optional[np.ndarray]:
    """Return (N, 2) float32 array or None if missing / malformed."""
    if visible_area is None:
        return None
    pts = np.array(visible_area, dtype=np.float32)
    if pts.ndim == 1:
        if len(pts) < 6 or len(pts) % 2 != 0:
            return None
        pts = pts.reshape(-1, 2)
    return pts if len(pts) >= 3 else None
