"""
Phase 7 – Counterfactual situation comparison.

Core idea
---------
Given a concrete game situation (an event node inside a possession graph),
extract the state embedding ``h_event`` from the GNN, then pair it with
the *global* ``z_p`` of different players and run the action heads.

This answers: "In **this** situation, what would player A do vs player B?"
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ..config import Config, EVENT_TYPES, PITCH_LENGTH, PITCH_WIDTH, POSITIONS
from ..phase3_graph import PossessionGraphBuilder
from ..phase3_graph.masking import mask_future_info
from ..phase4_model import PlayerSimilarityModel

_POS_NAME_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}

_ANGLE_BIN_LABELS = [
    "Forward", "Fwd-Right", "Right", "Back-Right",
    "Backward", "Back-Left", "Left", "Fwd-Left",
    "No angle",
]
_LENGTH_BIN_LABELS = [
    "Very short", "Short", "Medium", "Long", "Very long",
]


class SituationComparator:
    """
    Runs the trained model's action heads on a real game state while
    substituting different players' global ``z_p``.
    """

    def __init__(
        self,
        model: PlayerSimilarityModel,
        device: torch.device,
        graphs: List,
        Z: np.ndarray,
        player_info: pd.DataFrame,
        event_features: np.ndarray,
    ):
        self.model = model
        self.device = device
        self.graphs = graphs
        self.Z = Z
        self.event_features = event_features

        self._pid_to_z_idx: Dict[int, int] = {
            int(row["player_id"]): i
            for i, (_, row) in enumerate(player_info.iterrows())
        }
        self._names: Dict[int, str] = {
            int(row["player_id"]): row.get("player_name", "")
            for _, row in player_info.iterrows()
        }
        self._pid_to_pos_idx: Dict[int, int] = {
            int(row["player_id"]): _POS_NAME_TO_IDX.get(
                row.get("position_name", "Unknown"),
                _POS_NAME_TO_IDX.get("Unknown", 0),
            )
            for _, row in player_info.iterrows()
        }

        # global_event_idx → (graph_idx, local_event_idx)
        self._event_lookup: Dict[int, Tuple[int, int]] = {}
        for g_idx, g in enumerate(graphs):
            if not hasattr(g, "event_indices_global"):
                continue
            for local, global_idx in enumerate(g.event_indices_global.tolist()):
                self._event_lookup[int(global_idx)] = (g_idx, local)

    # ── helpers ──────────────────────────────────────────────────────

    def _get_z(self, pid: int) -> Optional[np.ndarray]:
        idx = self._pid_to_z_idx.get(pid)
        if idx is None:
            return None
        return self.Z[idx]

    def _describe_situation(self, global_idx: int, meta: pd.DataFrame) -> Dict:
        """Build a human-friendly dict describing an event's situation."""
        raw = self.event_features[global_idx]
        loc_x = float(raw[14]) * PITCH_LENGTH
        loc_y = float(raw[15]) * PITCH_WIDTH

        third = "defensive"
        if loc_x > 80:
            third = "attacking"
        elif loc_x > 40:
            third = "middle"

        lane = "center"
        if loc_y < 26.7:
            lane = "left"
        elif loc_y > 53.3:
            lane = "right"

        under_pressure = bool(raw[96] > 0.5)

        info: Dict = {
            "global_idx": global_idx,
            "location": (round(loc_x, 1), round(loc_y, 1)),
            "third": third,
            "lane": lane,
            "under_pressure": under_pressure,
        }
        if global_idx in meta.index:
            row = meta.loc[global_idx]
            info["period"] = int(row.get("period", -1)) if "period" in row.index else -1
            info["minute"] = int(row.get("minute", -1)) if "minute" in row.index else -1
            info["second"] = int(row.get("second", -1)) if "second" in row.index else -1
            info["observed_action"] = row.get("event_type", "")
        return info

    # ── core comparison ──────────────────────────────────────────────

    def compare(
        self,
        global_event_idx: int,
        player_ids: List[int],
        meta: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        For a single event, compute action distributions for every player
        in *player_ids* by substituting their global ``z_p``.

        Returns
        -------
        dict with:
          "situation"   : description dict
          "predictions" : {pid: {"action_type": ndarray, "angle_bin": ndarray,
                                  "length_bin": ndarray}}
        or None if the event is not in any graph.
        """
        if global_event_idx not in self._event_lookup:
            return None

        graph_idx, local_ev_idx = self._event_lookup[global_event_idx]
        g = self.graphs[graph_idx]

        g_masked = g.clone()
        masked_x = mask_future_info(g_masked["event"].x.numpy())
        g_masked["event"].x = torch.tensor(masked_x, dtype=torch.float32)
        g_dev = g_masked.to(self.device)

        with torch.no_grad():
            out_dict = self.model.encode_possession_counterfactual(g_dev)
            h_event = out_dict["event"]          # (T, d)
            h_ev = h_event[local_ev_idx]         # (d,)

        situation = self._describe_situation(global_event_idx, meta)
        predictions: Dict[int, Dict[str, np.ndarray]] = {}

        with torch.no_grad():
            for pid in player_ids:
                z_p_np = self._get_z(pid)
                if z_p_np is None:
                    continue
                z_p = torch.tensor(z_p_np, dtype=torch.float32, device=self.device)
                pos_idx = torch.tensor(
                    [self._pid_to_pos_idx.get(pid, 0)],
                    dtype=torch.long, device=self.device,
                )
                h_cond = self.model.film_condition(
                    h_ev.unsqueeze(0), z_p.unsqueeze(0), pos_idx=pos_idx,
                )  # (1, d)

                at = F.softmax(self.model.action_type_head(h_cond).squeeze(0), dim=-1).cpu().numpy()
                ang = F.softmax(self.model.angle_bin_head(h_cond).squeeze(0), dim=-1).cpu().numpy()
                ln = F.softmax(self.model.length_bin_head(h_cond).squeeze(0), dim=-1).cpu().numpy()

                predictions[pid] = {"action_type": at, "angle_bin": ang, "length_bin": ln}

        return {"situation": situation, "predictions": predictions}

    # ── text formatting ──────────────────────────────────────────────

    def format_comparison(self, result: Dict) -> List[str]:
        """Turn a ``compare`` result into human-readable lines."""
        sit = result["situation"]
        lines: List[str] = []
        lines.append("=" * 62)
        lines.append("SITUATION")
        lines.append(f"  Location : ({sit['location'][0]}, {sit['location'][1]})")
        lines.append(f"  Zone     : {sit['third']} third, {sit['lane']} lane")
        lines.append(f"  Pressure : {'yes' if sit['under_pressure'] else 'no'}")
        if sit.get("period", -1) >= 0:
            lines.append(f"  Time     : period {sit['period']}, {sit['minute']:02d}:{sit['second']:02d}")
        if sit.get("observed_action"):
            lines.append(f"  Observed : {sit['observed_action']}")
        lines.append("")

        lines.append("PREDICTED ACTION DISTRIBUTIONS")
        for pid, preds in result["predictions"].items():
            name = self._names.get(pid, str(pid))
            lines.append(f"  Player: {name} (id={pid})")

            at = preds["action_type"]
            top3 = np.argsort(at)[::-1][:4]
            lines.append("    action_type (top-4):")
            for i in top3:
                lines.append(f"      {EVENT_TYPES[i]:20s} {at[i]:.3f}")

            ang = preds["angle_bin"]
            top2a = np.argsort(ang)[::-1][:3]
            lines.append("    direction (top-3):")
            for i in top2a:
                lbl = _ANGLE_BIN_LABELS[i] if i < len(_ANGLE_BIN_LABELS) else f"bin {i}"
                lines.append(f"      {lbl:15s} {ang[i]:.3f}")

            ln = preds["length_bin"]
            top2l = np.argsort(ln)[::-1][:3]
            lines.append("    length (top-3):")
            for i in top2l:
                lbl = _LENGTH_BIN_LABELS[i] if i < len(_LENGTH_BIN_LABELS) else f"bin {i}"
                lines.append(f"      {lbl:15s} {ln[i]:.3f}")
            lines.append("")

        return lines

    # ── visualisation ────────────────────────────────────────────────

    def plot_comparison(
        self,
        result: Dict,
        output_dir: Path,
        filename: str,
    ) -> None:
        """
        Grouped bar chart of action_type probabilities for all players
        in a single situation.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        preds = result["predictions"]
        if not preds:
            return

        sit = result["situation"]
        all_probs = np.stack([p["action_type"] for p in preds.values()], axis=0)
        top_k = 6
        mean_probs = all_probs.mean(axis=0)
        top_idx = np.argsort(mean_probs)[::-1][:top_k]
        top_labels = [EVENT_TYPES[i] for i in top_idx]

        rows = []
        for pid, p in preds.items():
            name = self._names.get(pid, str(pid))
            for j, idx in enumerate(top_idx):
                rows.append({
                    "player": name,
                    "action": top_labels[j],
                    "prob": float(p["action_type"][idx]),
                })
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.8 / len(preds)
        x_base = np.arange(len(top_labels))
        for k, (pid, p) in enumerate(preds.items()):
            name = self._names.get(pid, str(pid))
            vals = [float(p["action_type"][idx]) for idx in top_idx]
            ax.bar(x_base + k * bar_width, vals, bar_width, label=name)

        ax.set_xticks(x_base + bar_width * (len(preds) - 1) / 2)
        ax.set_xticklabels(top_labels, rotation=30, ha="right")
        ax.set_ylabel("Probability")
        zone = f"{sit['third']} third, {sit['lane']}"
        pressure = " under pressure" if sit['under_pressure'] else ""
        ax.set_title(f"Action predictions — {zone}{pressure}")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    def plot_direction_comparison(
        self,
        result: Dict,
        output_dir: Path,
        filename: str,
    ) -> None:
        """Bar chart comparing angle-bin distributions across players."""
        output_dir.mkdir(parents=True, exist_ok=True)
        preds = result["predictions"]
        if not preds:
            return

        n_bins = len(_ANGLE_BIN_LABELS)
        fig, ax = plt.subplots(figsize=(9, 4))
        x_base = np.arange(n_bins)
        bar_w = 0.8 / len(preds)
        for k, (pid, p) in enumerate(preds.items()):
            name = self._names.get(pid, str(pid))
            vals = p["angle_bin"][:n_bins].tolist()
            ax.bar(x_base + k * bar_w, vals, bar_w, label=name)
        ax.set_xticks(x_base + bar_w * (len(preds) - 1) / 2)
        ax.set_xticklabels(_ANGLE_BIN_LABELS, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Probability")
        ax.set_title("Direction predictions")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)
