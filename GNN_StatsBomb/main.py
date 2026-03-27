"""
Main entry point for GNN StatsBomb Player Similarity System.

Modes
─────
  prepare            Phase 1 — load events, encode features, save artifacts.
  build_possessions  Phase 2 — group events into possession sequences.
  build_graphs       Phase 3 — build per-possession heterogeneous graphs.
  train              Phase 5 — train the GNN model (creates Phase 4 model).
  evaluate           Evaluate best model on test set (metrics + plots).
  inference          Phase 6 — generate player embeddings from trained model.
  search             Phase 6 — find similar players to a query.
  ground_truth       Evaluate pseudo ground-truth player pairs (rank check).
  self_consistency   Intrinsic self-consistency test (cross-competition + random-half).
  policy_diagnostic  Policy distance correlation + substitute quality diagnostic.
  fifa_comparison    FIFA stat comparison test (sample players, compare stats).
  analyze            Phase 7 — interpret embeddings and similarities.
  full_pipeline      Run Phases 1→6A end-to-end (no search).
  full_eval          Run ALL post-training evaluations for one pipeline.
  full_eval_all      Run ALL post-training evaluations for ALL 4 pipelines.

Usage
─────
  python main.py --mode prepare
  python main.py --mode build_possessions
  python main.py --mode build_graphs
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode inference
  python main.py --mode search --player_id 5503
  python main.py --mode ground_truth
  python main.py --mode self_consistency
  python main.py --mode full_pipeline
  python main.py --mode analyze

  # Run all evaluations for one pipeline
  python main.py --mode full_eval
  python main.py --mode full_eval --tag split_ctx --split_context_edges
  python main.py --mode full_eval --eval_output_dir ./my_evals

  # Run all evaluations for ALL 4 pipelines
  python main.py --mode full_eval_all

Experiment tagging (ablation)
─────────────────────────────
  python main.py --mode build_graphs --tag split_ctx --split_context_edges
  python main.py --mode train --tag split_ctx --split_context_edges
  python main.py --mode evaluate --tag split_ctx --split_context_edges

  python main.py --mode train --player_sampling --tag player_samp
  python main.py --mode train --split_context_edges --player_sampling --tag split_ctx_ps

  --tag namespaces Phase 3+ outputs (graphs, checkpoints, embeddings)
  while sharing Phase 1-2 data.  Omit --tag to use the baseline
  defaults (checkpoints/baseline/, embeddings/baseline/).
  --split_context_edges must be passed consistently for every phase of
  the same experiment (it changes graph structure and model architecture).
  --player_sampling enables player-aware contrastive batching (training only).
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import get_config, validate_graph_config_match, Config
from src.phase6_inference.provenance import load_manifest


# ── Pipeline registry ─────────────────────────────────────────────────────
# Each entry defines the CLI flags needed to reproduce one pipeline variant.
# Used by full_eval_all to iterate over all experiments.

PIPELINE_REGISTRY = [
    {
        "name": "baseline",
        "tag": "",
        "split_context_edges": False,
        "player_sampling": False,
    },
    {
        "name": "player_samp",
        "tag": "player_samp",
        "split_context_edges": False,
        "player_sampling": True,
    },
    {
        "name": "split_ctx",
        "tag": "split_ctx",
        "split_context_edges": True,
        "player_sampling": False,
    },
    {
        "name": "split_ctx_ps",
        "tag": "split_ctx_ps",
        "split_context_edges": True,
        "player_sampling": True,
    },
]


def _configure_pipeline(pipe_def: dict) -> Config:
    """Build a Config from a PIPELINE_REGISTRY entry."""
    config = get_config()
    if pipe_def["split_context_edges"]:
        config.graph.split_context_edges = True
    if pipe_def["player_sampling"]:
        config.training.player_sampling = True
        config.training.contrastive_temperature = 0.15
        config.training.lambda_contrast = 0.15
        config.training.lambda_pooled_contrast = 0.5
        config.training.anneal_temperature = True
        config.training.temperature_start = 0.15
        config.training.temperature_end = 0.02
        config.training.anneal_pooled_weight = True
        config.training.pooled_weight_start = 0.10
        config.training.pooled_weight_end = 0.50
    if pipe_def["tag"]:
        config.apply_tag(pipe_def["tag"])
    return config


def _build_run_info(
    config: Config,
    checkpoint_path: Optional[Path] = None,
    embedding_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_info: Dict[str, Any] = {
        "tag": config.tag,
        "graphs_filename": config.graphs_filename,
        "split_context_edges": bool(config.graph.split_context_edges),
        "checkpoint_dir": str(Path(config.training.checkpoint_dir).resolve()),
        "embedding_output_dir": str(Path(config.inference.embedding_output_dir).resolve()),
    }
    if checkpoint_path is not None:
        run_info["checkpoint_path"] = str(checkpoint_path.resolve())
    if embedding_dir is not None:
        manifest = load_manifest(embedding_dir)
        if manifest is not None:
            run_info["embedding_manifest"] = manifest
    if extra:
        run_info.update(extra)
    return run_info


def _augment_json_report(json_path: Path, run_info: Dict[str, Any]) -> None:
    if not json_path.exists():
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["run_info"] = run_info
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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

    save_path = out_dir / config.graphs_filename
    if save_path.exists():
        existing_graphs = PossessionGraphBuilder.load(str(save_path))
        try:
            validate_graph_config_match(existing_graphs, config.graph.split_context_edges)
        except ValueError as e:
            print(
                "\nRefusing to overwrite existing graph file with a different "
                "graph architecture."
            )
            print(f"Existing file: {save_path}")
            print(f"Reason: {e}")
            print("Use a different --tag (recommended) or remove the existing graph file.")
            return

    builder = PossessionGraphBuilder(config.graph)
    graphs = builder.build_graphs(possessions, features, freeze_frames)

    print(f"\nGraphs built: {len(graphs):,}")
    if graphs:
        g = graphs[0]
        print(f"  Sample graph — event nodes: {g['event'].x.shape[0]}, "
              f"player nodes: {g['player'].x.shape[0]}")

    PossessionGraphBuilder.save(graphs, str(save_path))
    print(f"\nSaved to {save_path}")


# ── Phase 5 (includes Phase 4 model creation) ───────────────────────────

def train_model(config: Config, resume_from: str = None) -> None:
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
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)

    print(f"\nGraphs loaded: {len(graphs):,}  ({config.graphs_filename})")

    train_g, val_g, test_g = train_val_test_split(
        graphs,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=42,
    )
    print(f"Train: {len(train_g):,}  Val: {len(val_g):,}  Test: {len(test_g):,}")

    train_ds = PossessionGraphDataset(train_g, features, config.model)
    val_ds = PossessionGraphDataset(val_g, features, config.model)

    if config.training.player_sampling:
        from src.phase5_training.sampler import PlayerAwareBatchSampler
        player_index = train_ds.build_player_index()
        sampler = PlayerAwareBatchSampler(
            player_index=player_index,
            players_per_batch=config.training.players_per_batch,
            possessions_per_player=config.training.possessions_per_player,
            total_possessions=len(train_ds),
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            collate_fn=collate_fn, num_workers=0,
        )
        print(f"Player-aware sampling: K={config.training.players_per_batch}, "
              f"M={config.training.possessions_per_player}, "
              f"{len(player_index)} unique players, "
              f"{len(sampler)} batches/epoch")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0,
        )

    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    trainer = Trainer(model, config.training, train_loader, val_loader)
    trainer.train(resume_from=resume_from)

    print("\nTraining complete.")


# ── Test-set evaluation ───────────────────────────────────────────────────

def run_evaluate(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """
    Evaluate the trained model on the held-out test set.

    Loads the same train/val/test split as training, runs the best checkpoint
    on the test set, and reports accuracy, macro F1, and outcome metrics.
    When *eval_output_dir* is given, saves outputs there; otherwise defaults
    to ``evaluations/{tag}/test_metrics/``.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase5_training import (
        PossessionGraphDataset,
        collate_fn,
        train_val_test_split,
    )
    from src.phase5_training.evaluator import evaluate

    print("\n" + "=" * 60)
    print("TEST-SET EVALUATION")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    features = np.load(out_dir / "event_features.npy")
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)

    train_g, val_g, test_g = train_val_test_split(
        graphs,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=42,
    )
    print(f"Test graphs: {len(test_g):,}")

    test_ds = PossessionGraphDataset(test_g, features, config.model)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run --mode train first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    eval_dir = eval_output_dir or _default_eval_dir(config, "test_metrics")
    metrics = evaluate(model, test_loader, device, eval_dir, save_plots=True)

    print("\n--- Test set metrics ---")
    print(f"  Action type  — accuracy: {metrics['action_type']['accuracy']:.4f}  macro F1: {metrics['action_type']['macro_f1']:.4f}")
    print(f"  Angle bin    — accuracy: {metrics['angle_bin']['accuracy']:.4f}  macro F1: {metrics['angle_bin']['macro_f1']:.4f}")
    print(f"  Length bin   — accuracy: {metrics['length_bin']['accuracy']:.4f}  macro F1: {metrics['length_bin']['macro_f1']:.4f}")
    print("  Outcome:")
    print(f"    ends_in_shot — accuracy: {metrics['outcome']['ends_in_shot']['accuracy']:.4f}  BCE: {metrics['outcome']['ends_in_shot']['bce']:.4f}  AUC: {metrics['outcome']['ends_in_shot']['auc_roc']:.4f}")
    print(f"    ends_in_goal — accuracy: {metrics['outcome']['ends_in_goal']['accuracy']:.4f}  BCE: {metrics['outcome']['ends_in_goal']['bce']:.4f}  AUC: {metrics['outcome']['ends_in_goal']['auc_roc']:.4f}")
    print(f"\nMetrics and plots saved to: {eval_dir}")


# ── Phase 6: Inference ───────────────────────────────────────────────────

def run_inference(
    config: Config,
    max_graphs: int | None = None,
    inference_split: str = "all",
) -> None:
    """Phase 6-A: Generate player embeddings."""
    import torch
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase6_inference import EmbeddingGenerator
    from src.phase5_training.dataset import train_val_test_split

    print("\n" + "=" * 60)
    print("PHASE 6: INFERENCE — EMBEDDING GENERATION")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)
    meta = pd.read_parquet(out_dir / "event_metadata.parquet")

    if inference_split != "all":
        train_g, val_g, test_g = train_val_test_split(
            graphs,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            test_ratio=config.training.test_ratio,
            seed=42,
        )
        split_map = {"train": train_g, "val": val_g, "test": test_g}
        graphs = split_map[inference_split]
        print(
            f"Using split='{inference_split}' graphs only: "
            f"{len(graphs):,} / {len(train_g) + len(val_g) + len(test_g):,}"
        )

    if max_graphs is not None and max_graphs < len(graphs):
        print(f"Using subset: {max_graphs:,} / {len(graphs):,} graphs")
        graphs = graphs[:max_graphs]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run --mode train first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    generator = EmbeddingGenerator(model, device, config.inference)
    Z, player_info = generator.generate(graphs, meta)

    print(f"\nEmbedding matrix: {Z.shape}")
    print(f"Players: {len(player_info)}")

    generator.save(
        Z,
        player_info,
        config.inference.embedding_output_dir,
        checkpoint_path=ckpt_path,
        graphs_filename=config.graphs_filename,
        split_context_edges=config.graph.split_context_edges,
        tag=config.tag,
        inference_split=inference_split,
    )


# ── Phase 6: Search ──────────────────────────────────────────────────────

def search_similar(config: Config, player_id: int) -> None:
    """Phase 6-B: Find similar players."""
    from src.phase6_inference import SimilaritySearcher, build_gender_map
    from src.phase6_inference.provenance import validate_manifest

    print("\n" + "=" * 60)
    print(f"PHASE 6: SIMILARITY SEARCH — Player {player_id}")
    print("=" * 60)

    emb_dir = Path(config.inference.embedding_output_dir)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    ok, reason = validate_manifest(
        output_dir=emb_dir,
        checkpoint_path=ckpt_path,
        graphs_filename=config.graphs_filename,
        split_context_edges=config.graph.split_context_edges,
    )
    if not ok:
        print(f"Embedding provenance check failed: {reason}")
        print("Re-run --mode inference for this pipeline/tag before search.")
        return

    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    try:
        gender_map = build_gender_map(
            config.data.output_dir, config.data.statsbomb_base_path,
        )
    except FileNotFoundError:
        gender_map = None

    searcher = SimilaritySearcher(config.inference)
    sim_matrix = searcher.compute_similarity_matrix(Z)

    results = searcher.find_similar_players(
        query_player_id=player_id,
        Z=Z,
        player_info=player_info,
        similarity_matrix=sim_matrix,
        gender_map=gender_map,
    )

    query_row = player_info[player_info["player_id"] == player_id]
    if len(query_row) > 0:
        q = query_row.iloc[0]
        print(f"\nQuery: {q['player_name']} ({q['position_name']}, "
              f"{q['n_possessions']} possessions)")

    print(f"\nTop-{config.inference.top_k} similar players:")
    print(results.to_string(index=False))


# ── Ground-truth evaluation ───────────────────────────────────────────────

def run_ground_truth(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """Evaluate pseudo ground-truth player similarity pairs.

    Loads Phase 6 embeddings, computes the rank at which each expected
    partner appears in the query player's neighbour list.  When
    *eval_output_dir* is given, saves outputs there; otherwise defaults
    to ``evaluations/{tag}/ground_truth/``.
    """
    from src.phase6_inference.ground_truth import evaluate_ground_truth, GROUND_TRUTH_PAIRS
    from src.phase6_inference.provenance import validate_manifest
    from src.phase6_inference import build_gender_map

    print("\n" + "=" * 60)
    print("PSEUDO GROUND-TRUTH EVALUATION")
    print("=" * 60)

    emb_dir = Path(config.inference.embedding_output_dir)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    ok, reason = validate_manifest(
        output_dir=emb_dir,
        checkpoint_path=ckpt_path,
        graphs_filename=config.graphs_filename,
        split_context_edges=config.graph.split_context_edges,
    )
    if not ok:
        print(f"Embedding provenance check failed: {reason}")
        print("Re-run --mode inference for this pipeline/tag before ground_truth.")
        return

    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    print(f"\nEmbeddings loaded: {Z.shape[0]} players, {Z.shape[1]}-D")
    print(f"Ground-truth pairs defined: {len(GROUND_TRUTH_PAIRS)}")

    print("Building gender map ...")
    try:
        gender_map = build_gender_map(
            config.data.output_dir, config.data.statsbomb_base_path,
        )
        n_m = sum(1 for v in gender_map.values() if v == "male")
        n_f = sum(1 for v in gender_map.values() if v == "female")
        print(f"  Gender map: {n_m} male, {n_f} female")
    except FileNotFoundError as e:
        print(f"  Warning: could not build gender map ({e}); skipping gender filter.")
        gender_map = None

    out = eval_output_dir or _default_eval_dir(config, "ground_truth")

    result = evaluate_ground_truth(
        Z,
        player_info,
        output_dir=out,
        similarity_metric=config.inference.similarity_metric,
        gender_map=gender_map,
    )
    _augment_json_report(
        out / "ground_truth_results.json",
        _build_run_info(
            config,
            checkpoint_path=Path(config.training.checkpoint_dir) / "best_model.pt",
            embedding_dir=emb_dir,
            extra={"mode": "ground_truth"},
        ),
    )

    summary = result["summary"]
    n_eval = summary["n_evaluated"]
    n_miss = summary["n_missing"]

    print(f"\nPairs evaluated : {n_eval}")
    print(f"Pairs skipped   : {n_miss} (player missing from embeddings)")

    if n_eval > 0:
        sim_metric = summary.get("similarity_metric", config.inference.similarity_metric)
        print(f"\n--- Per-pair results ---")
        for p in result["pairs"]:
            tier = f"[T{p['tier']}]"
            if p["status"] == "MISSING":
                print(f"  {tier} {p['player_a']} <-> {p['player_b']}  "
                      f"SKIPPED (missing: {', '.join(p['missing'])})")
            else:
                r_ab = p["rank_b_in_a"]
                r_ba = p["rank_a_in_b"]
                sim = p["similarity"]
                print(f"  {tier} {p['player_a']:25s} <-> {p['player_b']:25s}  "
                      f"{sim_metric}={sim:.4f}  rank(A→B)={r_ab:>4d}  rank(B→A)={r_ba:>4d}")

        print(f"\n--- Aggregate ---")
        print(f"  Mean {sim_metric}  : {summary['mean_similarity']:.4f}")
        print(f"  Mean rank    : {summary['mean_rank']:.1f}")
        print(f"  Median rank  : {summary['median_rank']}")
        print(f"  Hit@5        : {summary['hit_at_5']:.1%}")
        print(f"  Hit@10       : {summary['hit_at_10']:.1%}")
        print(f"  Hit@20       : {summary['hit_at_20']:.1%}")
        print(f"  Hit@50       : {summary['hit_at_50']:.1%}")

        for tier_key, ts in summary.get("by_tier", {}).items():
            print(f"  {tier_key.replace('_', ' ').title():10s}: "
                  f"mean_rank={ts['mean_rank']:.1f}  "
                  f"hit@5={ts['hit_at_5']:.1%}  "
                  f"hit@10={ts['hit_at_10']:.1%}  "
                  f"hit@20={ts['hit_at_20']:.1%}")

    print(f"\nReport saved to: {out / 'ground_truth_report.txt'}")
    print(f"JSON saved to:   {out / 'ground_truth_results.json'}")


# ── Self-consistency evaluation ──────────────────────────────────────────

def run_self_consistency(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """Intrinsic self-consistency test for player embeddings.

    For players appearing in multiple competitions, split their
    possession-level GNN embeddings by competition, pool each split
    independently, and check if the player's own alternate-competition
    embedding ranks highest.  Also runs a random-half stability test.
    When *eval_output_dir* is given, saves outputs there; otherwise
    defaults to ``evaluations/{tag}/self_consistency/``.
    """
    import torch
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase6_inference.self_consistency import SelfConsistencyEvaluator
    from src.phase6_inference import build_gender_map

    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY EVALUATION")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)
    meta = pd.read_parquet(out_dir / "event_metadata.parquet")

    print(f"\nGraphs loaded: {len(graphs):,}  ({config.graphs_filename})")
    print(f"Metadata rows: {len(meta):,}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run --mode train first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    emb_dir = Path(config.inference.embedding_output_dir)

    print("Building gender map ...")
    try:
        gender_map = build_gender_map(
            config.data.output_dir, config.data.statsbomb_base_path,
        )
        n_m = sum(1 for v in gender_map.values() if v == "male")
        n_f = sum(1 for v in gender_map.values() if v == "female")
        print(f"  Gender map: {n_m} male, {n_f} female")
    except FileNotFoundError as e:
        print(f"  Warning: could not build gender map ({e}); skipping gender filter.")
        gender_map = None

    out = eval_output_dir or _default_eval_dir(config, "self_consistency")

    evaluator = SelfConsistencyEvaluator(
        model=model,
        device=device,
        config=config.inference,
        min_poss_per_split=config.inference.min_samples_per_player,
        gender_map=gender_map,
    )
    result = evaluator.run(graphs, meta, output_dir=out)
    _augment_json_report(
        out / "self_consistency_results.json",
        _build_run_info(
            config,
            checkpoint_path=ckpt_path,
            embedding_dir=emb_dir,
            extra={"mode": "self_consistency"},
        ),
    )

    for test_key, test_label in [
        ("competition_split", "Competition-split"),
        ("random_half", "Random-half"),
    ]:
        data = result[test_key]
        if "summary" not in data:
            print(f"\n{test_label}: no testable players.")
            continue
        s = data["summary"]
        eff = s.get("effective_gallery_size", s["gallery_size"])
        print(f"\n--- {test_label} ({s['n_players']} players, effective gallery ~{eff}) ---")
        print(f"  Self-cosine  : {s['self_cosine_mean']:.4f} (margin over cross: {s['cosine_margin']:.4f})")
        print(f"  Mean rank    : {s['mean_rank_all']:.1f} / ~{eff}")
        print(f"  Median rank  : {s['median_rank_all']} / ~{eff}")
        print(f"  Hit@1={s['hit_at_1']:.1%}  Hit@5={s['hit_at_5']:.1%}  "
              f"Hit@10={s['hit_at_10']:.1%}  Hit@20={s['hit_at_20']:.1%}  "
              f"Hit@50={s['hit_at_50']:.1%}")

    print(f"\nReport saved to: {out / 'self_consistency_report.txt'}")
    print(f"JSON saved to:   {out / 'self_consistency_results.json'}")


# ── Phase 7: Analysis / interpretation ───────────────────────────────────

def analyze_results(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """
    Phase 7: Situation-level interpretation of player similarity.

    For a small set of query players, sample real game situations and
    predict what the query AND each nearest neighbour would do in those
    exact states.  Generates text reports, per-situation action bar
    charts, direction charts, and PCA neighbourhood plots.
    When *eval_output_dir* is given, saves outputs there; otherwise
    defaults to ``evaluations/{tag}/analysis/``.
    """
    from src.phase7_analysis import ReportBuilder

    print("\n" + "=" * 60)
    print("PHASE 7: ANALYSIS — SITUATION-LEVEL SIMILARITY INTERPRETATION")
    print("=" * 60)

    builder = ReportBuilder(config)
    builder.run(output_dir=eval_output_dir)


# ── Policy diagnostic ─────────────────────────────────────────────────────

def run_policy_diagnostic(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """Run policy-distance and substitute-quality diagnostics.

    Tests whether cosine similarity in the learned embedding space
    corresponds to actual behavioral similarity (JS divergence of
    predicted action distributions in identical game situations).
    When *eval_output_dir* is given, saves outputs there; otherwise
    defaults to ``evaluations/{tag}/policy_diagnostic/``.
    """
    import torch
    from src.phase3_graph import PossessionGraphBuilder
    from src.phase4_model import PlayerSimilarityModel
    from src.phase6_inference.policy_diagnostic import PolicyDiagnostic
    from src.phase6_inference.provenance import validate_manifest
    from src.phase6_inference import build_gender_map

    print("\n" + "=" * 60)
    print("POLICY DIAGNOSTIC")
    print("=" * 60)

    out_dir = Path(config.data.output_dir)
    graphs = PossessionGraphBuilder.load(str(out_dir / config.graphs_filename))
    validate_graph_config_match(graphs, config.graph.split_context_edges)
    meta = pd.read_parquet(out_dir / "event_metadata.parquet")

    print(f"\nGraphs loaded: {len(graphs):,}  ({config.graphs_filename})")

    emb_dir = Path(config.inference.embedding_output_dir)
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run --mode train first.")
        return
    ok, reason = validate_manifest(
        output_dir=emb_dir,
        checkpoint_path=ckpt_path,
        graphs_filename=config.graphs_filename,
        split_context_edges=config.graph.split_context_edges,
    )
    if not ok:
        print(f"Embedding provenance check failed: {reason}")
        print("Re-run --mode inference for this pipeline/tag before policy_diagnostic.")
        return

    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")

    print(f"Embeddings loaded: {Z.shape[0]} players, {Z.shape[1]}-D")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PlayerSimilarityModel(config.model, graph_config=config.graph)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Building gender map ...")
    try:
        gender_map = build_gender_map(
            config.data.output_dir, config.data.statsbomb_base_path,
        )
        n_m = sum(1 for v in gender_map.values() if v == "male")
        n_f = sum(1 for v in gender_map.values() if v == "female")
        print(f"  Gender map: {n_m} male, {n_f} female")
    except FileNotFoundError as e:
        print(f"  Warning: could not build gender map ({e}); skipping gender filter.")
        gender_map = None

    diagnostic = PolicyDiagnostic(
        model=model,
        device=device,
        Z=Z,
        player_info=player_info,
        graphs=graphs,
        metadata=meta,
        gender_map=gender_map,
    )
    out = eval_output_dir or _default_eval_dir(config, "policy_diagnostic")

    result = diagnostic.run(output_dir=out)
    _augment_json_report(
        out / "policy_diagnostic_results.json",
        _build_run_info(
            config,
            checkpoint_path=ckpt_path,
            embedding_dir=emb_dir,
            extra={"mode": "policy_diagnostic"},
        ),
    )

    corr = result.get("correlation", {})
    if "overall" in corr:
        ov = corr["overall"]
        print(f"\n--- Policy Distance Correlation ---")
        print(f"  Overall Spearman rho: {ov['spearman_rho']:.4f}  "
              f"({ov['n_pairs']:,} pooled pairs)")
        for grp in sorted(k for k in corr if k != "overall"):
            g = corr[grp]
            p_perm = g.get("p_value_permutation", None)
            p_str = f"  (p_perm={p_perm:.4f})" if p_perm is not None else ""
            print(f"  {grp:25s}: rho = {g['spearman_rho']:+.4f}{p_str}")

    sub = result.get("substitute_quality", {})
    if sub.get("n_queries", 0) > 0:
        print(f"\n--- Substitute Quality ---")
        print(f"  JS(top-K) = {sub['aggregate_mean_js_top_k']:.6f}  "
              f"JS(random) = {sub['aggregate_mean_js_random_k']:.6f}  "
              f"ratio = {sub['aggregate_ratio']:.2f}")

    film = result.get("film_sensitivity", {})
    if film:
        print(f"\n--- FiLM Sensitivity ---")
        print(f"  Overall: mean JS(correct vs shuffled) = {film['overall_mean_js']:.6f}  "
              f"median = {film['overall_median_js']:.6f}")
        for grp in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
            if grp in film.get("per_group", {}):
                g = film["per_group"][grp]
                print(f"  {grp:12s}: mean = {g['mean_js']:.6f}")

    print(f"\nReport: {out / 'policy_diagnostic_report.txt'}")
    print(f"JSON:   {out / 'policy_diagnostic_results.json'}")


# ── FIFA comparison ──────────────────────────────────────────────────────

def run_fifa_comparison(config: Config, eval_output_dir: Optional[Path] = None) -> None:
    """Run FIFA stat comparison test using the pipeline's embeddings.

    Imports the parameterised ``run_comparison`` from
    ``test_fifa_comparison.py`` and passes config-derived paths.
    When *eval_output_dir* is given, saves outputs there; otherwise
    defaults to ``evaluations/{tag}/fifa_comparison/``.
    """
    from test_fifa_comparison import run_comparison

    print("\n" + "=" * 60)
    print("FIFA STAT COMPARISON TEST")
    print("=" * 60)

    emb_dir = Path(config.inference.embedding_output_dir)
    if not (emb_dir / "player_embeddings.npy").exists():
        print(f"  Embeddings not found in {emb_dir} — skipping FIFA comparison.")
        return

    project_root = Path(__file__).resolve().parent.parent
    fifa_dir = project_root / "FIFA_data"
    male_csv = fifa_dir / "statsbomb_male_players_fifa.csv"
    female_csv = fifa_dir / "statsbomb_female_players_fifa.csv"

    if not male_csv.exists() or not female_csv.exists():
        print(f"  FIFA CSVs not found in {fifa_dir} — skipping FIFA comparison.")
        return

    if eval_output_dir is None:
        eval_output_dir = _default_eval_dir(config, "fifa_comparison")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_output_dir / "test_fifa_comparison.txt"

    run_comparison(
        embeddings_dir=emb_dir,
        processed_dir=Path(config.data.output_dir),
        statsbomb_base=Path(config.data.statsbomb_base_path),
        male_fifa_csv=male_csv,
        female_fifa_csv=female_csv,
        output_file=out_file,
        seed=42,
    )


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


# ── Full evaluation orchestrator ─────────────────────────────────────────

_EVAL_STEPS = [
    ("inference",          "Phase 6 — Embedding generation"),
    ("test_metrics",       "Test-set evaluation (accuracy, F1, AUC)"),
    ("ground_truth",       "Pseudo ground-truth pair ranking"),
    ("self_consistency",   "Self-consistency (competition-split + random-half)"),
    ("policy_diagnostic",  "Policy diagnostic (JS correlation, substitute quality, FiLM)"),
    ("analysis",           "Phase 7 — Situation-level similarity analysis"),
    ("fifa_comparison",    "FIFA stat comparison test"),
]


def run_full_eval(config: Config, eval_root: Path) -> None:
    """Run all post-training evaluations for one pipeline.

    Results are saved under ``eval_root/{pipeline_name}/`` with one
    subdirectory per evaluation step.  Embeddings remain in the normal
    ``embeddings/{tag}/`` location (read-only input).
    """
    import time

    name = config.tag or "baseline"
    pipe_dir = eval_root / name
    pipe_dir.mkdir(parents=True, exist_ok=True)

    total = len(_EVAL_STEPS)
    t0 = time.time()

    print("\n" + "#" * 70)
    print(f"#  FULL EVALUATION: {name}")
    print(f"#  Output directory: {pipe_dir}")
    print("#" * 70)
    print(f"\nSteps to run: {total}")
    for i, (step_id, desc) in enumerate(_EVAL_STEPS, 1):
        print(f"  {i}. [{step_id}] {desc}")
    print()

    def _step_banner(idx: int, step_id: str, desc: str) -> None:
        print(f"\n{'='*70}")
        print(f"  [{idx}/{total}] {desc}")
        print(f"  → {pipe_dir / step_id}")
        print(f"{'='*70}")

    # Step 1: Inference (embeddings stay in embeddings/{tag}/)
    _step_banner(1, "inference", "Embedding generation")
    run_inference(config)

    # Step 2: Test-set metrics
    _step_banner(2, "test_metrics", "Test-set evaluation")
    run_evaluate(config, eval_output_dir=pipe_dir / "test_metrics")

    # Step 3: Ground truth
    _step_banner(3, "ground_truth", "Pseudo ground-truth evaluation")
    run_ground_truth(config, eval_output_dir=pipe_dir / "ground_truth")

    # Step 4: Self-consistency
    _step_banner(4, "self_consistency", "Self-consistency evaluation")
    run_self_consistency(config, eval_output_dir=pipe_dir / "self_consistency")

    # Step 5: Policy diagnostic
    _step_banner(5, "policy_diagnostic", "Policy diagnostic")
    run_policy_diagnostic(config, eval_output_dir=pipe_dir / "policy_diagnostic")

    # Step 6: Phase 7 analysis
    _step_banner(6, "analysis", "Situation-level analysis (Phase 7)")
    analyze_results(config, eval_output_dir=pipe_dir / "analysis")

    # Step 7: FIFA comparison
    _step_banner(7, "fifa_comparison", "FIFA stat comparison test")
    run_fifa_comparison(config, eval_output_dir=pipe_dir / "fifa_comparison")

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'#'*70}")
    print(f"#  FULL EVALUATION COMPLETE: {name}")
    print(f"#  Time elapsed: {minutes}m {seconds}s")
    print(f"#  All outputs: {pipe_dir}")
    print(f"{'#'*70}")


def run_full_eval_all(eval_root: Path) -> None:
    """Run ``run_full_eval`` for every pipeline in PIPELINE_REGISTRY."""
    import time

    eval_root.mkdir(parents=True, exist_ok=True)
    n = len(PIPELINE_REGISTRY)
    t0 = time.time()

    print("\n" + "=" * 70)
    print("  FULL EVALUATION — ALL PIPELINES")
    print("=" * 70)
    for i, pipe_def in enumerate(PIPELINE_REGISTRY, 1):
        print(f"  {i}. {pipe_def['name']}")
    print(f"\nOutput root: {eval_root}")
    print()

    for i, pipe_def in enumerate(PIPELINE_REGISTRY, 1):
        print(f"\n{'*'*70}")
        print(f"*  PIPELINE {i}/{n}: {pipe_def['name']}")
        print(f"{'*'*70}")
        config = _configure_pipeline(pipe_def)
        run_full_eval(config, eval_root)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'='*70}")
    print(f"  ALL {n} PIPELINES EVALUATED")
    print(f"  Time elapsed: {minutes}m {seconds}s")
    print(f"  Results in:   {eval_root}")
    print("=" * 70)


# ── Eval path helper ─────────────────────────────────────────────────────

def _default_eval_dir(config: "Config", step: str) -> Path:
    """Return ``evaluations/{tag}/{step}`` for standalone evaluation modes."""
    tag = config.tag or "baseline"
    return Path("evaluations") / tag / step


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN StatsBomb Player Similarity System"
    )
    parser.add_argument(
        "--mode", type=str, default="prepare",
        choices=[
            "prepare", "build_possessions", "build_graphs",
            "train", "evaluate", "inference", "search", "ground_truth",
            "self_consistency", "full_pipeline", "analyze",
            "policy_diagnostic", "fifa_comparison",
            "full_eval", "full_eval_all",
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
    parser.add_argument(
        "--inference_split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help=(
            "Subset of graphs to use during --mode inference. "
            "'all' uses full corpus; train/val/test uses the same match-level "
            "split (seed=42) as training/evaluation."
        ),
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help=(
            "Experiment tag.  Replaces the default 'baseline' leaf "
            "directory for Phase 3+ outputs (e.g. --tag split_ctx "
            "→ checkpoints/split_ctx/).  Phase 1-2 outputs remain shared."
        ),
    )
    parser.add_argument(
        "--split_context_edges", action="store_true", default=False,
        help=(
            "Split 360 context edges into teammate/opponent relation "
            "types.  Must be used consistently across build_graphs, "
            "train, evaluate, inference, and analyze for the same "
            "experiment."
        ),
    )
    parser.add_argument(
        "--player_sampling", action="store_true", default=False,
        help=(
            "Enable player-aware batch sampling during training.  "
            "Guarantees multiple possessions per player in every batch "
            "so the contrastive loss receives positive pairs.  "
            "Training-phase only; does not affect graphs or inference."
        ),
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Resume training from a checkpoint file (e.g. "
            "'checkpoint_epoch_130.pt').  Loads model weights, optimizer, "
            "scheduler, and best_val_loss.  Only applies to --mode train."
        ),
    )
    parser.add_argument(
        "--eval_output_dir", type=str, default="./evaluations",
        help=(
            "Root directory for full_eval / full_eval_all outputs. "
            "Each pipeline gets a subdirectory with organised results. "
            "Default: ./evaluations"
        ),
    )
    args = parser.parse_args()

    config = get_config()
    if args.split_context_edges:
        config.graph.split_context_edges = True
    if args.player_sampling:
        config.training.player_sampling = True
        config.training.contrastive_temperature = 0.15
        config.training.lambda_contrast = 0.15
        config.training.lambda_pooled_contrast = 0.5
        config.training.anneal_temperature = True
        config.training.temperature_start = 0.15
        config.training.temperature_end = 0.02
        config.training.anneal_pooled_weight = True
        config.training.pooled_weight_start = 0.10
        config.training.pooled_weight_end = 0.50
    if args.tag:
        config.apply_tag(args.tag)
    if args.competition:
        config.data.competition_ids = args.competition
    if args.season:
        config.data.season_ids = args.season

    print("=" * 60)
    print("GNN STATSBOMB PLAYER SIMILARITY SYSTEM")
    print("=" * 60)
    print(f"Mode         : {args.mode}")
    if config.graph.split_context_edges:
        print(f"Context edges: split (teammate / opponent)")
    if config.training.player_sampling:
        print(f"Sampling     : player-aware (K={config.training.players_per_batch}, "
              f"M={config.training.possessions_per_player})")
        print(f"  temperature: {config.training.contrastive_temperature}  "
              f"λ_contrast: {config.training.lambda_contrast}  "
              f"λ_pooled: {config.training.lambda_pooled_contrast}")
        if config.training.anneal_temperature or config.training.anneal_pooled_weight:
            parts = []
            if config.training.anneal_temperature:
                parts.append(f"temp {config.training.temperature_start}->{config.training.temperature_end}")
            if config.training.anneal_pooled_weight:
                parts.append(f"λ_pooled {config.training.pooled_weight_start}->{config.training.pooled_weight_end}")
            print(f"  annealing  : {', '.join(parts)}")
    print(f"Tag          : {config.tag or 'baseline'}")
    print(f"  graphs     : {config.data.output_dir}/{config.graphs_filename}")
    print(f"  checkpoints: {config.training.checkpoint_dir}")
    print(f"  embeddings : {config.inference.embedding_output_dir}")
    print(f"Competitions : {config.data.competition_ids or 'all'}")
    print(f"Seasons      : {config.data.season_ids or 'all'}")

    if args.mode == "prepare":
        prepare_data(config)
    elif args.mode == "build_possessions":
        build_possessions(config)
    elif args.mode == "build_graphs":
        build_graphs(config)
    elif args.mode == "train":
        train_model(config, resume_from=args.resume)
    elif args.mode == "evaluate":
        eval_dir = _default_eval_dir(config, "test_metrics")
        run_evaluate(config, eval_output_dir=eval_dir)
    elif args.mode == "inference":
        run_inference(
            config,
            max_graphs=args.max_graphs,
            inference_split=args.inference_split,
        )
    elif args.mode == "ground_truth":
        eval_dir = _default_eval_dir(config, "ground_truth")
        run_ground_truth(config, eval_output_dir=eval_dir)
    elif args.mode == "self_consistency":
        eval_dir = _default_eval_dir(config, "self_consistency")
        run_self_consistency(config, eval_output_dir=eval_dir)
    elif args.mode == "full_pipeline":
        run_full_pipeline(config)
    elif args.mode == "analyze":
        eval_dir = _default_eval_dir(config, "analysis")
        analyze_results(config, eval_output_dir=eval_dir)
    elif args.mode == "policy_diagnostic":
        eval_dir = _default_eval_dir(config, "policy_diagnostic")
        run_policy_diagnostic(config, eval_output_dir=eval_dir)
    elif args.mode == "fifa_comparison":
        eval_dir = _default_eval_dir(config, "fifa_comparison")
        run_fifa_comparison(config, eval_output_dir=eval_dir)
    elif args.mode == "search":
        if args.player_id is None:
            parser.error("--player_id is required for search mode.")
        search_similar(config, args.player_id)
    elif args.mode == "full_eval":
        run_full_eval(config, eval_root=Path(args.eval_output_dir))
    elif args.mode == "full_eval_all":
        run_full_eval_all(eval_root=Path(args.eval_output_dir))

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
