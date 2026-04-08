"""
data/loader.py — StatsBomb data loading and preprocessing.

Supports loading from multiple competition-seasons in a single call.
Each (competition_id, season_id) pair is fetched and concatenated into
one unified events + frames DataFrame.

Performance notes
-----------------
- Location unpacking uses vectorised numpy stacking, not apply(pd.Series).
- Score lookup is fully vectorised — no iterrows().
- Match loading shows a tqdm progress bar per competition.
- match_meta is scoped to the matches actually loaded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm.auto import tqdm

from config import COMPETITIONS, MAX_MATCHES_PER_COMP


# ── Public API ─────────────────────────────────────────────────────────────────

def load_competition_data(
    competitions: List[Tuple[int, int]] = COMPETITIONS,
    max_matches_per_comp: Optional[int] = MAX_MATCHES_PER_COMP,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Load events and 360 frames for one or more competition-seasons.

    Parameters
    ----------
    competitions          : list of (competition_id, season_id) pairs
    max_matches_per_comp  : cap per competition — None = load all

    Returns
    -------
    events      : DataFrame — one row per event, across all competitions
    frames      : DataFrame — one row per player per frame
    match_meta  : dict  match_id -> {home_team, away_team, competition, season}
    score_lookup: dict  event_id -> (home_goals, away_goals) before this event
    """
    all_events:     List[pd.DataFrame] = []
    all_frames:     List[pd.DataFrame] = []
    all_match_meta: Dict               = {}
    total_matches                      = 0

    for comp_id, season_id in competitions:
        e, f, meta = _load_one_competition(comp_id, season_id, max_matches_per_comp)
        all_events.append(e)
        all_frames.append(f)
        all_match_meta.update(meta)
        total_matches += len(meta)

    print(f"\nConcatenating {total_matches} matches across {len(competitions)} competitions...")
    events = pd.concat(all_events, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)

    print("Building score lookup...")
    score_lookup = _build_score_lookup(events, all_match_meta)

    print(f"\n{'─'*50}")
    print(f"Total events  : {len(events):,}")
    print(f"Total frames  : {len(frames):,}")
    print(f"Total matches : {total_matches:,}")
    print(f"Score lookup  : {len(score_lookup):,} entries")
    print(f"{'─'*50}")

    return events, frames, all_match_meta, score_lookup


def list_available_360_competitions() -> pd.DataFrame:
    """
    Query the StatsBomb API and return a DataFrame of all competition-seasons
    that have confirmed 360 freeze-frame data available (match_available_360
    is non-null). Useful for discovering new releases.

    Returns DataFrame with columns:
        competition_id, season_id, competition_name, season_name,
        country_name, match_available_360
    """
    comps = sb.competitions()
    has_360 = comps[comps["match_available_360"].notna()].copy()
    cols = [
        "competition_id", "season_id", "competition_name",
        "season_name", "country_name", "match_available_360",
    ]
    cols = [c for c in cols if c in has_360.columns]
    return (
        has_360[cols]
        .sort_values(["competition_name", "season_name"])
        .reset_index(drop=True)
    )


def build_player_vocab(events: pd.DataFrame) -> Dict[int, int]:
    """
    Map each player_id that made a completed pass to a 1-based integer index.
    Index 0 is reserved for <UNK>.
    """
    pass_player_ids = (
        events[(events["type"] == "Pass") & (events["pass_outcome"].isna())]
        ["player_id"]
        .dropna()
        .unique()
    )
    vocab = {pid: idx + 1 for idx, pid in enumerate(sorted(pass_player_ids))}
    print(f"Player vocab: {len(vocab) + 1} total ({len(vocab)} known + 1 UNK)")
    return vocab


def filter_pass_events(events: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only completed passes that have a 360 frame AND a valid end location.
    """
    frame_event_ids = set(frames["id"].unique())
    mask = (
        (events["type"] == "Pass")
        & (events["id"].isin(frame_event_ids))
        & (events["pass_end_x"].notna())
        & (events["event_x"].notna())
        & (events["pass_outcome"].isna())
    )
    result = events[mask].copy()
    print(f"Pass events with 360 + valid end location: {len(result):,}")
    return result


def build_player_labels(events: pd.DataFrame) -> Dict[int, str]:
    """
    Returns dict: player_id -> "First Last (Position)" using most-frequent position.
    """
    labels: Dict[int, str] = {}
    has_name = "player" in events.columns
    has_pos  = "position"    in events.columns

    for pid, grp in events.dropna(subset=["player_id"]).groupby("player_id"):
        name = grp["player"].iloc[0] if has_name else str(pid)
        if has_pos:
            pos_counts = grp["position"].dropna().value_counts()
            pos = pos_counts.index[0] if len(pos_counts) else "Unknown"
        else:
            pos = "Unknown"
        labels[pid] = f"{name} ({pos})"
    return labels


# ── Single-competition loader ──────────────────────────────────────────────────

def _load_one_competition(
    comp_id: int,
    season_id: int,
    max_matches: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch events and frames for one competition-season."""
    matches     = sb.matches(competition_id=comp_id, season_id=season_id)
    matches_360 = matches[matches["match_status_360"].notna()].reset_index(drop=True)
    match_ids   = matches_360["match_id"].tolist()

    # Identify competition + season name for logging
    comp_name   = matches_360["competition"].apply(
        lambda x: x.get("competition_name", "") if isinstance(x, dict) else ""
    ).iloc[0] if len(matches_360) else f"comp_{comp_id}"
    season_name = matches_360["season"].apply(
        lambda x: x.get("season_name", "") if isinstance(x, dict) else ""
    ).iloc[0] if len(matches_360) else f"season_{season_id}"

    print(f"\n{comp_name} {season_name}  — {len(match_ids)} matches with 360 data")

    if max_matches is not None:
        match_ids = match_ids[:max_matches]
        print(f"  (capped at {max_matches})")

    # Build match_meta only for the matches we load
    loaded_mask = matches_360["match_id"].isin(match_ids)
    meta_df     = matches_360[loaded_mask].set_index("match_id")

    match_meta: Dict = {}
    for mid, row in meta_df.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")
        match_meta[mid] = {
            "home_team":   home.get("home_team_name") if isinstance(home, dict) else str(home),
            "away_team":   away.get("away_team_name") if isinstance(away, dict) else str(away),
            "competition": comp_name,
            "season":      season_name,
        }

    ev_list: List[pd.DataFrame] = []
    fr_list: List[pd.DataFrame] = []

    for mid in tqdm(match_ids, desc=f"  Loading {comp_name}", unit="match", leave=False):
        e = sb.events(match_id=mid)
        try:
            f = sb.frames(match_id=mid, fmt="dataframe")
        except Exception as ex:
            print(f"Error loading frames for comp {comp_id}, season {season_id}, match {mid}: {ex}")
            continue
        e["match_id"] = mid
        f["match_id"] = mid
        ev_list.append(e)
        fr_list.append(f)
    if len(ev_list) != 0:
        events = pd.concat(ev_list, ignore_index=True)
    if len(fr_list) != 0:
        frames = pd.concat(fr_list, ignore_index=True)
    print(f"  Loaded: {len(events):,} events, {len(frames):,} frame rows")

    events = _unpack_locations(events)
    frames = _unpack_frame_locations(frames)

    return events, frames, match_meta


# ── Vectorised helpers ─────────────────────────────────────────────────────────

def _unpack_list_column(series: pd.Series, names: List[str]) -> pd.DataFrame:
    """
    Vectorised unpacking of a column containing [x, y] lists.
    ~30-50× faster than apply(pd.Series) on large DataFrames.
    """
    arr        = series.to_numpy(dtype=object)
    valid_mask = pd.notna(series)
    out        = np.full((len(arr), len(names)), np.nan, dtype=np.float64)
    if valid_mask.any():
        out[valid_mask] = np.array(arr[valid_mask].tolist(), dtype=np.float64)
    return pd.DataFrame(out, index=series.index, columns=names)


def _unpack_locations(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if "location" in events.columns:
        events[["event_x", "event_y"]] = _unpack_list_column(
            events["location"], ["event_x", "event_y"]
        )
    if "pass_end_location" in events.columns:
        events[["pass_end_x", "pass_end_y"]] = _unpack_list_column(
            events["pass_end_location"], ["pass_end_x", "pass_end_y"]
        )
    return events


def _unpack_frame_locations(frames: pd.DataFrame) -> pd.DataFrame:
    loc_df = _unpack_list_column(frames["location"], ["player_x", "player_y"])
    return pd.concat([frames.drop(columns=["location"]), loc_df], axis=1)


def _build_score_lookup(
    events: pd.DataFrame,
    match_meta: Dict,
) -> Dict[str, Tuple[int, int]]:
    """
    Vectorised score lookup: event_id -> (home_goals, away_goals) before the event.
    """
    ev = events[["id", "match_id", "index", "type", "team", "shot_outcome"]].copy()

    home_map = {mid: m["home_team"] for mid, m in match_meta.items()}
    away_map = {mid: m["away_team"] for mid, m in match_meta.items()}
    ev["home_team"] = ev["match_id"].map(home_map)
    ev["away_team"] = ev["match_id"].map(away_map)

    is_goal         = (ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")
    ev["home_goal"] = (is_goal & (ev["team"] == ev["home_team"])).astype(int)
    ev["away_goal"] = (is_goal & (ev["team"] == ev["away_team"])).astype(int)

    ev = ev.sort_values(["match_id", "index"])
    ev["home_goals_before"] = (
        ev.groupby("match_id")["home_goal"].cumsum() - ev["home_goal"]
    )
    ev["away_goals_before"] = (
        ev.groupby("match_id")["away_goal"].cumsum() - ev["away_goal"]
    )

    return dict(zip(
        ev["id"],
        zip(ev["home_goals_before"].astype(int), ev["away_goals_before"].astype(int)),
    ))
