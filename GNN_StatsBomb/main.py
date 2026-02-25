"""
Main entry point for GNN StatsBomb Player Similarity System.

Modes
─────
  prepare            Phase 1 — load events, encode features, save artifacts.
  build_possessions  Phase 2 — group events into possession sequences.
  build_graphs       Phase 3 — build per-possession heterogeneous graphs.
  train              Phase 5 — train the GNN model (creates Phase 4 model).
  inference          Phase 6 — generate player embeddings from trained model.
  search             Phase 6 — find similar players to a query.
  full_pipeline      Run Phases 1→6A end-to-end (no search).
  analyze            Phase 7 — interpret embeddings and similarities.

Usage
─────
  python main.py --mode prepare
  python main.py --mode build_possessions
  python main.py --mode build_graphs
  python main.py --mode train
  python main.py --mode inference
  python main.py --mode search --player_id 5503
  python main.py --mode full_pipeline
  python main.py --mode analyze
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config, Config


# ── Phase 1 ──────────────────────────────────────────────────────────────

def prepare_data(config: Config) -> str:
    """Phase 1: Load → filter → encode → save."""
    from src.data_preparation import StatsBombDataLoader
    from src.feature_encoder import EventFeatureEncoder

    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION & FEATURE ENCODING")
    print("=" * 60)

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

    print("\nEncoding features …")
    encoder = EventFeatureEncoder(config.feature, mirror=config.feature.mirror_sides)
    X, meta = encoder.encode_dataframe(df)
    print(f"Feature matrix shape: {X.shape}  (events × features)")
    print(f"Feature dimension   : {encoder.feature_dim}")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN cells: {nan_count}  |  Inf cells: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  WARNING — cleaning NaN/Inf → 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    out_dir = Path(config.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "event_features.npy"
    meta_path = out_dir / "event_metadata.parquet"
    names_path = out_dir / "feature_names.json"
    stats_path = out_dir / "data_stats.json"

    np.save(str(features_path), X)
    meta_out = meta.drop(columns=["freeze_frame"], errors="ignore")
    meta_out.to_parquet(str(meta_path), index=False)

    # Save freeze frames separately for Phase 3 graph construction
    freeze_path = out_dir / "freeze_frames.pkl"
    if "freeze_frame" in df.columns:
        freeze_data = df["freeze_frame"].tolist()
        with open(freeze_path, "wb") as f:
            pickle.dump(freeze_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {freeze_path.name}  (freeze frames for {len(freeze_data):,} events)")

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


# ── Phase 2 ──────────────────────────────────────────────────────────────

def build_possessions(config: Config) -> None:
    """Phase 2: Group events into possession sequences."""
    from src.phase2_possession import PossessionBuilder

    print("\n" + "=" * 60)
    print("PHASE 2: POSSESSION CONSTRUCTION")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    meta = pd.read_parquet(out_dir / "event_metadata.parquet")
    features = np.load(out_dir / "event_features.npy")

    builder = PossessionBuilder(config.possession)
    possessions = builder.build(meta, features)

    print(f"\nTotal possessions: {len(possessions):,}")
    n_shot = sum(1 for p in possessions if p.ends_in_shot)
    n_goal = sum(1 for p in possessions if p.ends_in_goal)
    print(f"  ending in shot: {n_shot:,}")
    print(f"  ending in goal: {n_goal:,}")

    save_path = out_dir / "possessions.pkl"
    PossessionBuilder.save(possessions, str(save_path))
    print(f"\nSaved to {save_path}")


# ── Phase 3 ──────────────────────────────────────────────────────────────

def build_graphs(config: Config) -> None:
    """Phase 3: Build per-possession heterogeneous graphs."""
    from src.phase2_possession import PossessionBuilder
    from src.phase3_graph import PossessionGraphBuilder

    print("\n" + "=" * 60)
    print("PHASE 3: GRAPH CONSTRUCTION (HeteroData)")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    features = np.load(out_dir / "event_features.npy")
    possessions = PossessionBuilder.load(str(out_dir / "possessions.pkl"))

    freeze_path = out_dir / "freeze_frames.pkl"
    with open(freeze_path, "rb") as f:
        freeze_frames = pickle.load(f)

    print(f"\nPossessions: {len(possessions):,}")
    print(f"Events: {features.shape[0]:,}")
    print(f"Freeze frames: {len(freeze_frames):,}")

    builder = PossessionGraphBuilder(config.graph)
    graphs = builder.build_graphs(possessions, features, freeze_frames)

    print(f"\nGraphs built: {len(graphs):,}")
    if graphs:
        g = graphs[0]
        print(f"  Sample graph — event nodes: {g['event'].x.shape[0]}, "
              f"player nodes: {g['player'].x.shape[0]}")

    save_path = out_dir / "possession_graphs.pkl"
    PossessionGraphBuilder.save(graphs, str(save_path))
    print(f"\nSaved to {save_path}")


# ── Phase 5 (includes Phase 4 model creation) ───────────────────────────

def train_model(config: Config) -> None:
    """Phase 5: Train the GNN model."""
    import torch
    from torch.utils.data import DataLoader
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase5_training import (
        PossessionGraphDataset, collate_fn, train_val_test_split, Trainer,
    )

    print("\n" + "=" * 60)
    print("PHASE 5: TRAINING")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    features = np.load(out_dir / "event_features.npy")
    graphs = PossessionGraphBuilder.load(str(out_dir / "possession_graphs.pkl"))

    print(f"\nGraphs loaded: {len(graphs):,}")

    train_g, val_g, test_g = train_val_test_split(
        graphs,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
    )
    print(f"Train: {len(train_g):,}  Val: {len(val_g):,}  Test: {len(test_g):,}")

    train_ds = PossessionGraphDataset(train_g, features, config.model)
    val_ds = PossessionGraphDataset(val_g, features, config.model)

    train_loader = DataLoader(
        train_ds, batch_size=config.training.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = PlayerSimilarityModel(config.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    trainer = Trainer(model, config.training, train_loader, val_loader)
    trainer.train()

    print("\nTraining complete.")


# ── Phase 6: Inference ───────────────────────────────────────────────────

def run_inference(config: Config, max_graphs: int | None = None) -> None:
    """Phase 6-A: Generate player embeddings."""
    import torch
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase6_inference import EmbeddingGenerator

    print("\n" + "=" * 60)
    print("PHASE 6: INFERENCE — EMBEDDING GENERATION")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    graphs = PossessionGraphBuilder.load(str(out_dir / "possession_graphs.pkl"))
    meta = pd.read_parquet(out_dir / "event_metadata.parquet")

    if max_graphs is not None and max_graphs < len(graphs):
        print(f"Using subset: {max_graphs:,} / {len(graphs):,} graphs")
        graphs = graphs[:max_graphs]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PlayerSimilarityModel(config.model)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    generator = EmbeddingGenerator(model, device, config.inference)
    Z, player_info = generator.generate(graphs, meta)

    print(f"\nEmbedding matrix: {Z.shape}")
    print(f"Players: {len(player_info)}")

    generator.save(Z, player_info, config.inference.embedding_output_dir)


# ── Phase 6: Search ──────────────────────────────────────────────────────

def search_similar(config: Config, player_id: int) -> None:
    """Phase 6-B: Find similar players."""
    from src.phase6_inference import SimilaritySearcher

    print("\n" + "=" * 60)
    print(f"PHASE 6: SIMILARITY SEARCH — Player {player_id}")
    print("=" * 60)

    emb_dir = Path(config.inference.embedding_output_dir)
    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    searcher = SimilaritySearcher(config.inference)
    sim_matrix = searcher.compute_similarity_matrix(Z)

    results = searcher.find_similar_players(
        query_player_id=player_id,
        Z=Z,
        player_info=player_info,
        similarity_matrix=sim_matrix,
    )

    query_row = player_info[player_info["player_id"] == player_id]
    if len(query_row) > 0:
        q = query_row.iloc[0]
        print(f"\nQuery: {q['player_name']} ({q['position_name']}, "
              f"{q['n_possessions']} possessions)")

    print(f"\nTop-{config.inference.top_k} similar players:")
    print(results.to_string(index=False))


# ── Phase 7: Analysis / interpretation ───────────────────────────────────

def analyze_results(config: Config, num_queries: int = 5) -> None:
    """
    Phase 7: Situation-level interpretation of player similarity.

    For a small set of query players, sample real game situations and
    predict what the query AND each nearest neighbour would do in those
    exact states.  Generates text reports, per-situation action bar
    charts, direction charts, and PCA neighbourhood plots.
    """
    from src.phase7_analysis import ReportBuilder

    print("\n" + "=" * 60)
    print("PHASE 7: ANALYSIS — SITUATION-LEVEL SIMILARITY INTERPRETATION")
    print("=" * 60)

    builder = ReportBuilder(config)
    builder.run()


# ── Full pipeline helper ─────────────────────────────────────────────────

def run_full_pipeline(config: Config) -> None:
    """
    Run the full pipeline from raw StatsBomb data to player embeddings:

      Phase 1  prepare          — encode events + save artifacts
      Phase 2  build_possessions
      Phase 3  build_graphs
      Phase 5  train            — fit GNN model
      Phase 6A inference        — generate player embeddings

    Search (Phase 6B) is left as a separate CLI call since it requires a
    specific query player_id.
    """
    print("\n" + "=" * 60)
    print("FULL PIPELINE: PREPARE → POSSESSIONS → GRAPHS → TRAIN → INFERENCE")
    print("=" * 60)

    features_path = prepare_data(config)
    if not features_path:
        print("Aborting full pipeline: Phase 1 produced no features.")
        return

    build_possessions(config)
    build_graphs(config)
    train_model(config)
    run_inference(config)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN StatsBomb Player Similarity System"
    )
    parser.add_argument(
        "--mode", type=str, default="prepare",
        choices=[
            "prepare", "build_possessions", "build_graphs",
            "train", "inference", "search", "full_pipeline", "analyze",
        ],
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--competition", type=int, nargs="*", default=None,
        help="Competition ID(s) to include (default: all).",
    )
    parser.add_argument(
        "--season", type=int, nargs="*", default=None,
        help="Season ID(s) to include (default: all).",
    )
    parser.add_argument(
        "--player_id", type=int, default=None,
        help="Player ID for similarity search.",
    )
    parser.add_argument(
        "--max_graphs", type=int, default=None,
        help="Limit number of possession graphs for inference (faster validation).",
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
    elif args.mode == "build_possessions":
        build_possessions(config)
    elif args.mode == "build_graphs":
        build_graphs(config)
    elif args.mode == "train":
        train_model(config)
    elif args.mode == "inference":
        run_inference(config, max_graphs=args.max_graphs)
    elif args.mode == "full_pipeline":
        run_full_pipeline(config)
    elif args.mode == "analyze":
        analyze_results(config)
    elif args.mode == "search":
        if args.player_id is None:
            parser.error("--player_id is required for search mode.")
        search_similar(config, args.player_id)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
