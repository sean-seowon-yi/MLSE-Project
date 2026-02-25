"""
Main entry point for GNN StatsBomb Player Similarity System.

Uses StatsBomb 360 data (events + spatial context from freeze frames). Only
matches with three-sixty/{match_id}.json are loaded; only events with a 360
frame are kept.

Phase 1 — Data Preparation & Feature Encoding
    python main.py --mode prepare
    python main.py --mode prepare --competition 11 --season 1

The ``prepare`` mode:
1. Loads StatsBomb events and 360 freeze frames for the selected competitions/seasons.
2. Keeps only events that have a 360 frame; attaches freeze_frame to each.
3. Encodes each event as a fixed-length vector (event features + 360 spatial context;
   left/right roles kept distinct by default, configurable via config).
4. Saves the feature matrix, metadata, and a summary report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config, Config
from src.data_preparation import StatsBombDataLoader
from src.feature_encoder import EventFeatureEncoder


def prepare_data(config: Config) -> str:
    """
    Phase 1: Load → filter → encode → save.

    Returns path to saved feature matrix.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION & FEATURE ENCODING")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────
    loader = StatsBombDataLoader(config.data)

    if config.data.use_360:
        print("\nUsing StatsBomb 360: only matches with three-sixty data are loaded; only events with a 360 frame are kept.")

    comps = loader.load_competitions()
    print(f"\nCompetitions/seasons selected: {len(comps)}")

    matches = loader.load_matches(comps)
    print(f"Matches found: {len(matches)}")

    print("\nLoading events …")
    df = loader.load_all_events(matches)
    print(f"Total on-ball events loaded: {len(df):,}")

    if df.empty:
        print("No events loaded — check paths and filters.")
        return ""

    # ── 2. Quick data summary ────────────────────────────────────────
    print("\n--- Event-type distribution ---")
    type_counts = df["event_type"].value_counts()
    for etype, count in type_counts.items():
        print(f"  {etype:25s} {count:>8,}")

    n_players = df["player_id"].nunique()
    n_teams = df["team_id"].nunique()
    n_matches = df["match_id"].nunique()
    print(f"\nUnique players : {n_players:,}")
    print(f"Unique teams   : {n_teams:,}")
    print(f"Unique matches : {n_matches:,}")

    # ── 3. Encode features ───────────────────────────────────────────
    print("\nEncoding features …")
    encoder = EventFeatureEncoder(config.feature, mirror=config.feature.mirror_sides)
    X, meta = encoder.encode_dataframe(df)
    print(f"Feature matrix shape: {X.shape}  (events × features)")
    print(f"Feature dimension   : {encoder.feature_dim}")

    # Sanity checks
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN cells: {nan_count}  |  Inf cells: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  WARNING — cleaning NaN/Inf → 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 4. Save ──────────────────────────────────────────────────────
    out_dir = Path(config.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "event_features.npy"
    meta_path = out_dir / "event_metadata.parquet"
    names_path = out_dir / "feature_names.json"
    stats_path = out_dir / "data_stats.json"

    np.save(str(features_path), X)
    meta_out = meta.drop(columns=["freeze_frame"], errors="ignore")
    meta_out.to_parquet(str(meta_path), index=False)

    with open(names_path, "w") as f:
        json.dump(encoder.get_feature_names(), f, indent=2)

    stats = {
        "n_events": int(len(df)),
        "n_players": int(n_players),
        "n_teams": int(n_teams),
        "n_matches": int(n_matches),
        "n_competitions": len(comps),
        "feature_dim": int(encoder.feature_dim),
        "event_type_counts": {k: int(v) for k, v in type_counts.items()},
        "mirror_sides": bool(config.feature.mirror_sides),
        "use_360": bool(config.data.use_360),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  {features_path.name}  ({X.nbytes / 1e6:.1f} MB)")
    print(f"  {meta_path.name}")
    print(f"  {names_path.name}")
    print(f"  {stats_path.name}")

    return str(features_path)


def main():
    parser = argparse.ArgumentParser(
        description="GNN StatsBomb Player Similarity System"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prepare",
        choices=["prepare"],
        help="Pipeline phase to run (Phase 1 only for now)",
    )
    parser.add_argument(
        "--competition",
        type=int,
        nargs="*",
        default=None,
        help="Competition ID(s) to include (default: all)",
    )
    parser.add_argument(
        "--season",
        type=int,
        nargs="*",
        default=None,
        help="Season ID(s) to include (default: all)",
    )
    args = parser.parse_args()

    config = get_config()

    if args.competition:
        config.data.competition_ids = args.competition
    if args.season:
        config.data.season_ids = args.season

    print("=" * 60)
    print("GNN STATSBOMB PLAYER SIMILARITY SYSTEM")
    print("=" * 60)
    print(f"Mode         : {args.mode}")
    print(f"Competitions : {config.data.competition_ids or 'all'}")
    print(f"Seasons      : {config.data.season_ids or 'all'}")

    if args.mode == "prepare":
        prepare_data(config)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
