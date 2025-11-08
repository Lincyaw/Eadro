"""End-to-end preprocessing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..data.io import save_preprocessed_dataset
from .chunk_aligner import ChunkAligner
from .processors import (
    LogTemplateExtractor,
    MetricsProcessor,
    RecordsProcessor,
    ServiceInfo,
    TracesProcessor,
)

__all__ = ["preprocess_dataset"]


def preprocess_dataset(
    raw_data_root: Path,
    dataset_name: str,
    output_root: Path,
    benchmark: str = "TrainTicket",
    chunk_length: int = 60,
    test_ratio: float = 0.2,
    overlap_threshold: int = 1,
    drain_config: Optional[Path] = None,
) -> Path:
    """Run complete preprocessing pipeline.

    Args:
        raw_data_root: Root directory containing raw dataset
        dataset_name: Dataset identifier (e.g., 'TT', 'SN')
        output_root: Root directory for all outputs
        benchmark: Benchmark system name
        chunk_length: Chunk duration in seconds
        test_ratio: Test set proportion
        overlap_threshold: Minimum fault overlap for labeling
        drain_config: Optional Drain3 config path

    Returns:
        Path to the final chunks directory

    Example:
        >>> from pathlib import Path
        >>> chunks_dir = preprocess_dataset(
        ...     raw_data_root=Path("/data/TT_Dataset"),
        ...     dataset_name="TT",
        ...     output_root=Path("./preprocessed"),
        ...     chunk_length=60,
        ... )
        >>> print(f"Chunks saved to: {chunks_dir}")
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Setup paths
    fault_data_path = raw_data_root / "data"
    fault_free_path = raw_data_root / "no_fault"

    parsed_dir = output_root / "parsed_data" / dataset_name
    chunks_dir = output_root / "chunks" / dataset_name

    logging.info(f"Starting preprocessing for {dataset_name}")
    logging.info(f"Raw data: {raw_data_root}")
    logging.info(f"Output: {chunks_dir}")

    # Initialize processors
    service_info = ServiceInfo(benchmark)

    # Step 1: Process records
    logging.info("\n" + "=" * 80)
    logging.info("Step 1: Processing fault records")
    logging.info("=" * 80)
    records_processor = RecordsProcessor(dataset_name)
    records_processor.process(fault_data_path, parsed_dir)

    # Step 2: Extract log templates
    logging.info("\n" + "=" * 80)
    logging.info("Step 2: Extracting log templates")
    logging.info("=" * 80)
    log_extractor = LogTemplateExtractor(drain_config)
    templates = log_extractor.extract_templates(
        fault_free_path, fault_data_path, parsed_dir
    )

    # Step 3: Process metrics
    logging.info("\n" + "=" * 80)
    logging.info("Step 3: Processing metrics")
    logging.info("=" * 80)
    metrics_processor = MetricsProcessor(service_info)
    fault_folders = sorted([f for f in fault_data_path.iterdir() if f.is_dir()])
    metrics_processor.process(fault_folders, parsed_dir)

    # Step 4: Process traces
    logging.info("\n" + "=" * 80)
    logging.info("Step 4: Processing traces")
    logging.info("=" * 80)
    traces_processor = TracesProcessor(service_info)
    span_files = [f / "spans.json" for f in fault_folders]
    traces_processor.process(span_files, parsed_dir)

    # Step 5: Align chunks
    logging.info("\n" + "=" * 80)
    logging.info("Step 5: Aligning multi-source data into chunks")
    logging.info("=" * 80)
    aligner = ChunkAligner(
        service_info,
        chunk_length=chunk_length,
        overlap_threshold=overlap_threshold,
    )

    dataset = aligner.process_all_records(
        records_dir=parsed_dir,
        logs_dir=parsed_dir,
        metrics_dir=parsed_dir,
        traces_dir=parsed_dir,
        templates=templates,
        output_dir=chunks_dir,
        test_ratio=test_ratio,
    )

    # Step 6: Save dataset
    logging.info("\n" + "=" * 80)
    logging.info("Step 6: Saving dataset")
    logging.info("=" * 80)
    save_preprocessed_dataset(dataset, chunks_dir)

    # Also save legacy format for compatibility
    ChunkAligner.save_legacy_format(dataset, chunks_dir)

    logging.info(f"\n{'=' * 80}")
    logging.info(f"Preprocessing complete! Output: {chunks_dir}")
    logging.info(f"{'=' * 80}\n")

    return chunks_dir
