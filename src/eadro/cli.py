"""Command-line interface for training Eadro models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import typer
from rich.console import Console
from torch.utils.data import DataLoader

from eadro.config import TrainingConfig
from eadro.data.dataset import ChunkGraphDataset, collate_graph_batches
from eadro.data.io import load_preprocessed_dataset
from eadro.preprocessing.pipeline import preprocess_dataset
from eadro.training import Trainer
from eadro.utils import configure_experiment, record_scores, seed_everything

app = typer.Typer(
    name="eadro",
    help="🔧 Eadro: End-to-End Troubleshooting Framework for Microservices",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


def setup_logging():
    """Configure logging with rich formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


@app.command()
def preprocess(
    raw_data: str = typer.Option(
        ...,
        help="Path to raw dataset directory",
    ),
    dataset_name: str = typer.Option(
        ...,
        help="Dataset identifier (e.g., TT, SN)",
    ),
    output_root: str = typer.Option(
        "./preprocessed",
        help="Root directory for preprocessed output",
    ),
    benchmark: str = typer.Option(
        "TrainTicket",
        help="Benchmark system name (TrainTicket or SocialNetwork)",
    ),
    chunk_length: int = typer.Option(
        60,
        help="Chunk duration in seconds",
    ),
    test_ratio: float = typer.Option(
        0.2,
        help="Test set proportion",
    ),
    overlap_threshold: int = typer.Option(
        1,
        help="Minimum fault overlap for labeling",
    ),
) -> None:
    """Preprocess raw dataset."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting preprocessing pipeline...")

    try:
        chunks_dir = preprocess_dataset(
            raw_data_root=Path(raw_data),
            dataset_name=dataset_name,
            output_root=Path(output_root),
            benchmark=benchmark,
            chunk_length=chunk_length,
            test_ratio=test_ratio,
            overlap_threshold=overlap_threshold,
        )

        logging.info(f"\n{'=' * 80}")
        logging.info("Preprocessing completed successfully!")
        logging.info(f"Output directory: {chunks_dir}")
        logging.info(f"{'=' * 80}\n")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def convert_rcabench(
    rcabench_root: str = typer.Option(
        ...,
        help="Path to RCABench dataset directory (e.g., __dev__rcabench_test_r1)",
    ),
    output_root: str = typer.Option(
        ...,
        help="Output directory for converted data",
    ),
    chunk_length: int = typer.Option(
        60,
        help="Chunk duration in seconds",
    ),
    test_ratio: float = typer.Option(
        0.2,
        help="Test set proportion",
    ),
) -> None:
    """Convert RCABench dataset to Eadro format."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting RCABench dataset conversion...")

    try:
        from eadro.preprocessing.rcabench_adapter import RcaBenchAdapter

        adapter = RcaBenchAdapter(
            chunk_length=chunk_length,
            test_ratio=test_ratio,
        )
        dataset = adapter.convert_dataset(
            rcabench_root=Path(rcabench_root),
            output_dir=Path(output_root),
        )

        logging.info(f"\n{'=' * 80}")
        logging.info("Conversion completed successfully!")
        logging.info(f"Train chunks: {len(dataset.train)}")
        logging.info(f"Test chunks: {len(dataset.test)}")
        logging.info(f"Output directory: {output_root}")
        logging.info(f"{'=' * 80}\n")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def train(
    # Data parameters
    dataset: str = typer.Option(
        ...,
        help="Dataset name (e.g., TT, SN)",
    ),
    data_root: str = typer.Option(
        ".",
        help="Root directory for datasets",
    ),
    result_dir: str = typer.Option(
        "./result",
        help="Directory to save results",
    ),
    # Training parameters
    random_seed: int = typer.Option(
        42,
        help="Random seed",
    ),
    gpu: bool = typer.Option(
        True,
        help="Use GPU if available",
    ),
    epochs: int = typer.Option(
        50,
        help="Number of epochs",
    ),
    batch_size: int = typer.Option(
        256,
        help="Batch size",
    ),
    lr: float = typer.Option(
        0.001,
        help="Learning rate",
    ),
    patience: int = typer.Option(
        10,
        help="Early stopping patience",
    ),
    evaluation_epoch: int = typer.Option(
        10,
        help="Evaluate every N epochs",
    ),
    # Model architecture
    self_attn: bool = typer.Option(
        True,
        help="Use self-attention",
    ),
    fuse_dim: int = typer.Option(
        128,
        help="Fusion dimension",
    ),
    alpha: float = typer.Option(
        0.5,
        help="Loss weight for detection vs localization",
    ),
    locate_hiddens: List[int] = typer.Option(
        [64],
        help="Hidden sizes for localization head",
    ),
    detect_hiddens: List[int] = typer.Option(
        [64],
        help="Hidden sizes for detection head",
    ),
    # Source-specific parameters
    log_dim: int = typer.Option(
        16,
        help="Log embedding dimension",
    ),
    trace_kernel_sizes: List[int] = typer.Option(
        [2],
        help="Trace TCN kernel sizes",
    ),
    trace_hiddens: List[int] = typer.Option(
        [64],
        help="Trace TCN hidden sizes",
    ),
    metric_kernel_sizes: List[int] = typer.Option(
        [2],
        help="Metric TCN kernel sizes",
    ),
    metric_hiddens: List[int] = typer.Option(
        [64],
        help="Metric TCN hidden sizes",
    ),
    graph_hiddens: List[int] = typer.Option(
        [64],
        help="Graph GAT hidden sizes",
    ),
    attn_head: int = typer.Option(
        4,
        help="GAT attention heads",
    ),
    activation: float = typer.Option(
        0.2,
        help="LeakyReLU negative slope",
    ),
) -> None:
    """Train Eadro model."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Convert to config object
    config = TrainingConfig(
        random_seed=random_seed,
        gpu=gpu,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        patience=patience,
        self_attention=self_attn,
        fuse_dim=fuse_dim,
        alpha=alpha,
        locate_hiddens=tuple(locate_hiddens),
        detect_hiddens=tuple(detect_hiddens),
        log_dim=log_dim,
        trace_kernel_sizes=tuple(trace_kernel_sizes),
        trace_hiddens=tuple(trace_hiddens),
        metric_kernel_sizes=tuple(metric_kernel_sizes),
        metric_hiddens=tuple(metric_hiddens),
        graph_hiddens=tuple(graph_hiddens),
        attn_head=attn_head,
        activation=activation,
        data_root=Path(data_root),
        dataset_name=dataset,
        result_dir=Path(result_dir),
        evaluation_epoch=evaluation_epoch,
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
    dataset_obj = load_preprocessed_dataset(data_dir)

    # Create data loaders
    train_dataset = ChunkGraphDataset(dataset_obj.train.chunks, dataset_obj.metadata)
    test_dataset = ChunkGraphDataset(dataset_obj.test.chunks, dataset_obj.metadata)

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
        event_num=dataset_obj.metadata.event_dim,
        metric_num=dataset_obj.metadata.metric_dim,
        node_num=dataset_obj.metadata.node_count,
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
        chunk_lenth=dataset_obj.metadata.chunk_length,
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


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    app()
