"""
Main Entry Point for GNN Player Similarity System.

This script orchestrates the entire pipeline:
1. Data Preparation - Load and preprocess SkillCorner data
2. Graph Assembly - Convert frames to graph structures
3. Training - Train the GATv2 autoencoder
4. Inference - Generate embeddings and similarity search
5. Validation - Verify role consistency

Usage:
    python main.py --mode all          # Run full pipeline
    python main.py --mode prepare      # Only prepare data
    python main.py --mode train        # Only train model
    python main.py --mode inference    # Only run inference
    python main.py --mode validate     # Only run validation
"""

import argparse
from pathlib import Path
import json
import torch

from config import get_config, Config
from data_preparation import SkillCornerDataLoader
from graph_assembly import GraphAssembler, save_graphs, load_graphs
from dataset import train_val_test_split, create_data_loaders
from model import PlayerSimilarityAutoencoder
from train import Trainer, compute_reconstruction_metrics
from inference import (
    EmbeddingGenerator, 
    PlayerProfileBuilder, 
    SimilaritySearcher,
    RoleValidator,
    save_profiles,
    load_profiles
)
from utils import (
    set_seed, 
    get_device, 
    print_model_summary,
    plot_training_history,
    plot_embedding_tsne,
    plot_embedding_tsne_detailed,
    plot_similarity_matrix_by_group,
    plot_role_match_rates,
    plot_embedding_statistics,
    save_metrics
)


def prepare_data(config: Config) -> str:
    """
    Phase 1 & 2: Data Preparation and Graph Assembly.
    
    Returns:
        Path to saved graphs file
    """
    print("\n" + "=" * 60)
    print("PHASE 1 & 2: DATA PREPARATION AND GRAPH ASSEMBLY")
    print("=" * 60)
    
    # Load and process SkillCorner data
    print("\nLoading SkillCorner data...")
    loader = SkillCornerDataLoader(config.data)
    
    all_frames = []
    for match_info in loader.matches_info:
        match_id = match_info['id']
        print(f"\nProcessing match {match_id}...")
        try:
            frames = loader.process_match(match_id)
            all_frames.extend(frames)
            print(f"  Got {len(frames)} valid frames")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nTotal frames collected: {len(all_frames)}")
    
    # Build graphs
    print("\nBuilding graph snapshots...")
    assembler = GraphAssembler(config.graph)
    graphs = assembler.frames_to_graphs(all_frames)
    
    print(f"Total graphs created: {len(graphs)}")
    
    # Save graphs
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_path = output_dir / "graphs.pkl"
    save_graphs(graphs, str(graphs_path))
    
    # Save some statistics
    stats = {
        'num_matches': len(loader.matches_info),
        'num_frames': len(all_frames),
        'num_graphs': len(graphs),
        'sampling_rate': config.data.sampling_rate
    }
    with open(output_dir / "data_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nData saved to {output_dir}")
    
    return str(graphs_path)


def train_model(config: Config, graphs_path: str) -> str:
    """
    Phase 3 & 4: Model Training.
    
    Returns:
        Path to best model checkpoint
    """
    print("\n" + "=" * 60)
    print("PHASE 3 & 4: MODEL TRAINING")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(config.data.random_seed)
    
    # Load graphs
    print("\nLoading graphs...")
    graphs = load_graphs(graphs_path)
    print(f"Loaded {len(graphs)} graphs")
    
    # Split data
    print("\nSplitting data...")
    train_graphs, val_graphs, test_graphs = train_val_test_split(
        graphs,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.random_seed,
        split_by_match=True
    )
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=config.training.batch_size,
        mask_all_players=False
    )
    
    # Create model
    print("\nCreating model...")
    model = PlayerSimilarityAutoencoder(config.model)
    print_model_summary(model)
    
    # Create trainer
    device = get_device()
    print(f"\nUsing device: {device}")
    
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device)
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    # Plot training history
    checkpoint_dir = Path(config.training.checkpoint_dir)
    plot_training_history(history, str(checkpoint_dir / "training_history.png"))
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    trainer.load_checkpoint('best_model.pt')
    metrics = compute_reconstruction_metrics(model, test_loader, trainer.device)
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test X MAE: {metrics['x_mae']:.4f}")
    print(f"Test Y MAE: {metrics['y_mae']:.4f}")
    
    # Save test metrics
    with open(checkpoint_dir / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return str(checkpoint_dir / "best_model.pt")


def run_inference(config: Config, model_path: str, graphs_path: str) -> str:
    """
    Phase 5: Generate Embeddings and Build Profiles.
    
    Returns:
        Path to saved profiles
    """
    print("\n" + "=" * 60)
    print("PHASE 5: INFERENCE AND SIMILARITY SEARCH")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    model = PlayerSimilarityAutoencoder(config.model)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load graphs
    print("\nLoading graphs...")
    graphs = load_graphs(graphs_path)
    print(f"Loaded {len(graphs)} graphs")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    generator = EmbeddingGenerator(model, device)
    player_embeddings = generator.generate_embeddings(graphs, batch_size=64)
    print(f"Generated embeddings for {len(player_embeddings)} players")
    
    # Build profiles
    print("\nBuilding player profiles...")
    profile_builder = PlayerProfileBuilder(config.inference)
    profiles = profile_builder.build_profiles(
        player_embeddings, 
        min_samples=config.inference.min_samples_per_player
    )
    print(f"Built profiles for {len(profiles)} players")
    
    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    searcher = SimilaritySearcher(config.inference)
    similarity_matrix = searcher.compute_similarity_matrix(profiles)
    
    # Save results
    output_dir = Path(config.inference.embedding_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profiles_path = output_dir / "player_profiles.json"
    save_profiles(profiles, str(profiles_path))
    
    # Save similarity matrix
    import numpy as np
    np.save(str(output_dir / "similarity_matrix.npy"), similarity_matrix)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Similarity matrix by position group (most interpretable)
    plot_similarity_matrix_by_group(
        similarity_matrix, profiles,
        save_path=str(output_dir / "similarity_matrix_by_group.png")
    )
    
    # 2. t-SNE embedding visualization (grouped by position type)
    if len(profiles) >= 5:  # Need enough points for t-SNE
        plot_embedding_tsne(
            profiles, perplexity=min(30, len(profiles) - 1),
            save_path=str(output_dir / "embedding_tsne_grouped.png")
        )
        
        # 3. t-SNE with detailed/specific roles
        plot_embedding_tsne_detailed(
            profiles, perplexity=min(30, len(profiles) - 1),
            save_path=str(output_dir / "embedding_tsne_detailed.png")
        )
    
    # 4. Embedding statistics
    plot_embedding_statistics(
        profiles,
        save_path=str(output_dir / "embedding_statistics.png")
    )
    
    print(f"\nResults saved to {output_dir}")
    
    return str(profiles_path)


def run_validation(config: Config, profiles_path: str):
    """
    Validate role consistency of learned embeddings.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: ROLE CONSISTENCY CHECK")
    print("=" * 60)
    
    # Load profiles
    print("\nLoading profiles...")
    profiles = load_profiles(profiles_path)
    print(f"Loaded {len(profiles)} profiles")
    
    # Load similarity matrix
    output_dir = Path(config.inference.embedding_output_dir)
    import numpy as np
    similarity_matrix = np.load(str(output_dir / "similarity_matrix.npy"))
    
    # Run validation
    print("\nValidating role consistency...")
    validator = RoleValidator()
    metrics = validator.validate_role_consistency(
        profiles, similarity_matrix, top_k=5
    )
    validator.print_validation_report(metrics)
    
    # Save validation metrics
    serializable = {
        'overall_exact_match_rate': float(metrics['overall_exact_match_rate']),
        'overall_group_match_rate': float(metrics['overall_group_match_rate']),
        'per_role_metrics': {
            role: {
                'exact_match_rate': float(data['exact_match_rate']),
                'group_match_rate': float(data['group_match_rate']),
                'count': int(data['count'])
            }
            for role, data in metrics['per_role_metrics'].items()
        }
    }
    save_metrics(serializable, str(output_dir / "validation_metrics.json"))
    
    # Plot role match rates
    print("\nGenerating validation visualizations...")
    plot_role_match_rates(serializable, str(output_dir / "role_match_rates.png"))
    
    # Example similarity searches
    print("\n" + "=" * 60)
    print("EXAMPLE SIMILARITY SEARCHES")
    print("=" * 60)
    
    searcher = SimilaritySearcher(config.inference)
    
    # Find a few players of different roles
    example_roles = ['Left Winger', 'Right Back', 'Center Forward', 'Goalkeeper']
    
    for role in example_roles:
        role_players = profiles[profiles['role'] == role]
        if len(role_players) > 0:
            target_id = role_players.iloc[0]['player_id']
            print(f"\n=== Similar players to Player {target_id} ({role}) ===")
            similar = searcher.find_similar_players(
                target_id, profiles, similarity_matrix, top_k=5
            )
            print(similar[['player_id', 'role', 'similarity']].to_string(index=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GNN Player Similarity System')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['all', 'prepare', 'train', 'inference', 'validate'],
        help='Which phase to run'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--graphs-path',
        type=str,
        default=None,
        help='Path to graphs file (for train/inference modes)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model (for inference mode)'
    )
    parser.add_argument(
        '--profiles-path',
        type=str,
        default=None,
        help='Path to profiles file (for validate mode)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    print("=" * 60)
    print("GNN PLAYER SIMILARITY SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    if args.lr:
        print(f"Learning rate: {args.lr}")
    
    if args.mode == 'all':
        # Run full pipeline
        graphs_path = prepare_data(config)
        model_path = train_model(config, graphs_path)
        profiles_path = run_inference(config, model_path, graphs_path)
        run_validation(config, profiles_path)
        
    elif args.mode == 'prepare':
        prepare_data(config)
        
    elif args.mode == 'train':
        graphs_path = args.graphs_path or str(Path(config.data.output_dir) / "graphs.pkl")
        train_model(config, graphs_path)
        
    elif args.mode == 'inference':
        graphs_path = args.graphs_path or str(Path(config.data.output_dir) / "graphs.pkl")
        model_path = args.model_path or str(Path(config.training.checkpoint_dir) / "best_model.pt")
        run_inference(config, model_path, graphs_path)
        
    elif args.mode == 'validate':
        profiles_path = args.profiles_path or str(Path(config.inference.embedding_output_dir) / "player_profiles.json")
        run_validation(config, profiles_path)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
