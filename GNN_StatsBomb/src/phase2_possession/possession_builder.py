"""
Phase 2: Possession Construction.

Groups Phase 1 event-level outputs into possession sequences using
StatsBomb's ``possession_number`` and ``possession_team_id`` fields.

Each possession is a continuous spell where one team controls the ball.
Events within a possession are sorted temporally, and optional
possession-level labels (shot, goal, xG) are computed for downstream
value/outcome prediction heads.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import PossessionConfig, get_config


def _parse_timestamp(ts: str) -> float:
    """Parse StatsBomb timestamp ``"HH:MM:SS.mmm"`` into seconds (float).

    The timestamp is period-relative (resets each half). Since possessions
    never span periods, only relative deltas within a possession matter.
    """
    try:
        parts = ts.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    except (ValueError, IndexError, AttributeError):
        return 0.0


@dataclass
class Possession:
    """A single possession sequence with indices into the global event arrays."""

    pos_key: Tuple[int, int, int]  # (match_id, possession_number, possession_team_id)
    event_indices: List[int]       # row indices into event_features.npy

    # Per-event metadata (parallel to event_indices)
    player_ids: List[int]
    team_ids: List[int]
    event_types: List[str]
    position_names: List[str]

    # Per-event timestamps (seconds within the period, millisecond precision)
    # for time-delta edge attributes in Phase 3
    timestamps_sec: List[float] = field(default_factory=list)

    # Possession-level labels for outcome heads
    ends_in_shot: bool = False
    ends_in_goal: bool = False
    total_xg: float = 0.0

    match_id: int = 0
    possession_team_id: int = 0


class PossessionBuilder:
    """
    Builds possession sequences from Phase 1 outputs.

    Parameters
    ----------
    config : PossessionConfig
        min/max events per possession.
    """

    def __init__(self, config: PossessionConfig):
        self.config = config

    def build(
        self,
        metadata: pd.DataFrame,
        event_features: Optional[np.ndarray] = None,
    ) -> List[Possession]:
        """
        Group events into possessions.

        Parameters
        ----------
        metadata : DataFrame
            Must contain: match_id, possession_number, possession_team_id,
            player_id, team_id, event_type, position_name, period, minute, second.
        event_features : ndarray, optional
            (n_events, 126) array — not modified, but its length is used for
            sanity-checking alignment with *metadata*.

        Returns
        -------
        list[Possession]
        """
        required = [
            "match_id", "possession_number", "possession_team_id",
            "player_id", "team_id", "event_type", "position_name",
            "period", "minute", "second",
        ]
        for col in required:
            if col not in metadata.columns:
                raise ValueError(f"Metadata missing required column: {col}")

        if event_features is not None and len(metadata) != event_features.shape[0]:
            raise ValueError(
                f"Metadata rows ({len(metadata)}) != event_features rows "
                f"({event_features.shape[0]})"
            )

        metadata = metadata.copy()
        metadata["_orig_idx"] = np.arange(len(metadata))

        grouped = metadata.groupby(
            ["match_id", "possession_number", "possession_team_id"],
            sort=False,
        )

        possessions: List[Possession] = []

        for pos_key, group in tqdm(grouped, desc="Building possessions", leave=False):
            group = group.sort_values(
                ["period", "minute", "second", "_orig_idx"]
            )

            n = len(group)
            if n < self.config.min_events or n > self.config.max_events:
                continue

            indices = group["_orig_idx"].tolist()
            player_ids = group["player_id"].tolist()
            team_ids = group["team_id"].tolist()
            event_types = group["event_type"].tolist()
            position_names = group["position_name"].tolist()

            if "timestamp" in group.columns:
                timestamps_sec = group["timestamp"].apply(_parse_timestamp).tolist()
            else:
                timestamps_sec = (
                    group["minute"].astype(float) * 60.0
                    + group["second"].astype(float)
                ).tolist()

            # "Shot" anywhere in the possession — in StatsBomb data, shots
            # almost always terminate the possession (save/block/goal resets
            # play), so "contains shot" ≈ "ends in shot" in practice.
            ends_in_shot = "Shot" in event_types
            ends_in_goal = False
            total_xg = 0.0

            # Scan for shot outcomes if available
            if "shot_outcome" in group.columns:
                shot_rows = group[group["event_type"] == "Shot"]
                for _, sr in shot_rows.iterrows():
                    outcome = sr.get("shot_outcome")
                    if outcome == "Goal":
                        ends_in_goal = True
            if "shot_xg" in group.columns:
                xg_vals = group.loc[group["event_type"] == "Shot", "shot_xg"]
                total_xg = float(xg_vals.sum()) if len(xg_vals) > 0 else 0.0

            match_id, _, poss_team = pos_key

            possessions.append(Possession(
                pos_key=pos_key,
                event_indices=indices,
                player_ids=player_ids,
                team_ids=team_ids,
                event_types=event_types,
                position_names=position_names,
                timestamps_sec=timestamps_sec,
                ends_in_shot=ends_in_shot,
                ends_in_goal=ends_in_goal,
                total_xg=total_xg,
                match_id=int(match_id),
                possession_team_id=int(poss_team),
            ))

        return possessions

    @staticmethod
    def save(possessions: List[Possession], path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(possessions, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> List[Possession]:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    config = get_config()
    out_dir = Path(config.data.output_dir)

    meta = pd.read_parquet(out_dir / "event_metadata.parquet")
    features = np.load(out_dir / "event_features.npy")

    builder = PossessionBuilder(config.possession)
    possessions = builder.build(meta, features)

    print(f"Total possessions: {len(possessions):,}")
    if possessions:
        p = possessions[0]
        print(f"  First: key={p.pos_key}, events={len(p.event_indices)}, "
              f"shot={p.ends_in_shot}, goal={p.ends_in_goal}")

    save_path = out_dir / "possessions.pkl"
    PossessionBuilder.save(possessions, str(save_path))
    print(f"Saved to {save_path}")
