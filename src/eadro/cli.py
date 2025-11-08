"""Command-line interface for training Eadro models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eadro.config import TrainingConfig
from eadro.data.dataset import ChunkGraphDataset, collate_graph_batches
from eadro.data.io import load_preprocessed_dataset
from eadro.preprocessing.pipeline import preprocess_dataset
from eadro.training import Trainer
from eadro.utils import configure_experiment, record_scores, seed_everything


def get_device(use_gpu: bool) -> torch.device:
    """Select computation device.

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        torch.device instance
    """
    if use_gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Eadro: Microservice troubleshooting toolkit"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess raw dataset"
    )
    preprocess_parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
        help="Path to raw dataset directory",
    )
    preprocess_parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset identifier (e.g., TT, SN)",
    )
    preprocess_parser.add_argument(
        "--output_root",
        type=str,
        default="./preprocessed",
        help="Root directory for preprocessed output",
    )
    preprocess_parser.add_argument(
        "--benchmark",
        type=str,
        default="TrainTicket",
        choices=["TrainTicket", "SocialNetwork"],
        help="Benchmark system name",
    )
    preprocess_parser.add_argument(
        "--chunk_length",
        type=int,
        default=60,
        help="Chunk duration in seconds",
    )
    preprocess_parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    preprocess_parser.add_argument(
        "--overlap_threshold",
        type=int,
        default=1,
        help="Minimum fault overlap for labeling",
    )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train Eadro model")

    # Training parameters
    train_parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--gpu", type=bool, default=True, help="Use GPU if available"
    )
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    # Model architecture
    train_parser.add_argument(
        "--self_attn", type=bool, default=True, help="Use self-attention"
    )
    train_parser.add_argument(
        "--fuse_dim", type=int, default=128, help="Fusion dimension"
    )
    train_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Loss weight for detection vs localization",
    )
    train_parser.add_argument(
        "--locate_hiddens",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes for localization head",
    )
    train_parser.add_argument(
        "--detect_hiddens",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes for detection head",
    )

    # Source-specific parameters
    train_parser.add_argument(
        "--log_dim", type=int, default=16, help="Log embedding dimension"
    )
    train_parser.add_argument(
        "--trace_kernel_sizes",
        type=int,
        nargs="+",
        default=[2],
        help="Trace TCN kernel sizes",
    )
    train_parser.add_argument(
        "--trace_hiddens",
        type=int,
        nargs="+",
        default=[64],
        help="Trace TCN hidden sizes",
    )
    train_parser.add_argument(
        "--metric_kernel_sizes",
        type=int,
        nargs="+",
        default=[2],
        help="Metric TCN kernel sizes",
    )
    train_parser.add_argument(
        "--metric_hiddens",
        type=int,
        nargs="+",
        default=[64],
        help="Metric TCN hidden sizes",
    )
    train_parser.add_argument(
        "--graph_hiddens",
        type=int,
        nargs="+",
        default=[64],
        help="Graph GAT hidden sizes",
    )
    train_parser.add_argument(
        "--attn_head", type=int, default=4, help="GAT attention heads"
    )
    train_parser.add_argument(
        "--activation", type=float, default=0.2, help="LeakyReLU negative slope"
    )

    # Data parameters
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., TT, SN)",
    )
    train_parser.add_argument(
        "--data_root",
        type=str,
        default="./chunks",
        help="Root directory for datasets",
    )
    train_parser.add_argument(
        "--result_dir", type=str, default="./result", help="Directory to save results"
    )
    train_parser.add_argument(
        "--evaluation_epoch",
        type=int,
        default=10,
        help="Evaluate every N epochs",
    )

    return parser.parse_args()


def run_preprocessing(args: argparse.Namespace) -> int:
    """Run preprocessing pipeline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting preprocessing pipeline...")

    try:
        chunks_dir = preprocess_dataset(
            raw_data_root=Path(args.raw_data),
            dataset_name=args.dataset_name,
            output_root=Path(args.output_root),
            benchmark=args.benchmark,
            chunk_length=args.chunk_length,
            test_ratio=args.test_ratio,
            overlap_threshold=args.overlap_threshold,
        )

        logging.info(f"\n{'=' * 80}")
        logging.info(f"Preprocessing completed successfully!")
        logging.info(f"Output directory: {chunks_dir}")
        logging.info(f"{'=' * 80}\n")

        return 0
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_training(args: argparse.Namespace) -> int:
    """Run model training.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Convert to config object
    config = TrainingConfig(
        random_seed=args.random_seed,
        gpu=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        self_attention=args.self_attn,
        fuse_dim=args.fuse_dim,
        alpha=args.alpha,
        locate_hiddens=tuple(args.locate_hiddens),
        detect_hiddens=tuple(args.detect_hiddens),
        log_dim=args.log_dim,
        trace_kernel_sizes=tuple(args.trace_kernel_sizes),
        trace_hiddens=tuple(args.trace_hiddens),
        metric_kernel_sizes=tuple(args.metric_kernel_sizes),
        metric_hiddens=tuple(args.metric_hiddens),
        graph_hiddens=tuple(args.graph_hiddens),
        attn_head=args.attn_head,
        activation=args.activation,
        data_root=Path(args.data_root),
        dataset_name=args.dataset,
        result_dir=Path(args.result_dir),
        evaluation_epoch=args.evaluation_epoch,
    )

    # Setup experiment
    hash_id, experiment_dir = configure_experiment(
        config.to_logging_dict(), config.result_dir
    )
    seed_everything(config.random_seed)
    device = get_device(config.gpu)

    # Load dataset
    data_dir = config.data_dir()
    logging.info(f"Loading dataset from {data_dir}")
    dataset = load_preprocessed_dataset(data_dir)

    # Create data loaders
    train_dataset = ChunkGraphDataset(dataset.train.chunks, dataset.metadata)
    test_dataset = ChunkGraphDataset(dataset.test.chunks, dataset.metadata)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graph_batches,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batches,
        pin_memory=True,
    )

    # Initialize trainer
    trainer = Trainer(
        event_num=dataset.metadata.event_dim,
        metric_num=dataset.metadata.metric_dim,
        node_num=dataset.metadata.node_count,
        device=device,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        patience=config.patience,
        result_dir=config.result_dir,
        hash_id=hash_id,
        # Model kwargs
        self_attn=config.self_attention,
        fuse_dim=config.fuse_dim,
        alpha=config.alpha,
        locate_hiddens=list(config.locate_hiddens),
        detect_hiddens=list(config.detect_hiddens),
        log_dim=config.log_dim,
        trace_kernel_sizes=list(config.trace_kernel_sizes),
        trace_hiddens=list(config.trace_hiddens),
        metric_kernel_sizes=list(config.metric_kernel_sizes),
        metric_hiddens=list(config.metric_hiddens),
        graph_hiddens=list(config.graph_hiddens),
        attn_head=config.attn_head,
        activation=config.activation,
        chunk_lenth=dataset.metadata.chunk_length,
    )

    # Train and evaluate
    logging.info("Starting training...")
    scores, converge_epoch = trainer.fit(
        train_loader, test_loader, evaluation_epoch=config.evaluation_epoch
    )

    # Record results
    record_scores(config.result_dir, hash_id, scores, converge_epoch)
    logging.info(f"Experiment hash ID: {hash_id}")
    logging.info("Training completed successfully")

    return 0


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success)
    """
    args = parse_args()

    if args.command == "preprocess":
        return run_preprocessing(args)
    elif args.command == "train":
        return run_training(args)
    else:
        print("Error: No command specified. Use 'preprocess' or 'train'.")
        print("Run with --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
