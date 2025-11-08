"""Adapter to convert RCABench dataset to Eadro format.

This module directly converts RCABench parquet data into EadroDataset structure,
bypassing intermediate json/csv formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ..data.schema import ChunkPayload, DatasetMetadata, DatasetSplit, EadroDataset
from .processors import ServiceInfo

__all__ = ["RcaBenchAdapter"]


def _z_score_normalize(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Z-score normalization with safety check."""
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


class RcaBenchAdapter:
    """Convert RCABench dataset to Eadro format.

    Args:
        chunk_length: Duration of each chunk in seconds (default: 60)
        test_ratio: Proportion of data for test set (default: 0.2)
        overlap_threshold: Minimum fault overlap in seconds to mark chunk as faulty (default: 1)
    """

    # Metrics to extract from RCABench (mapping to Eadro's expected metrics)
    METRIC_MAPPING = {
        "container.cpu.usage": "cpu_usage_total",
        "k8s.pod.cpu.usage": "cpu_usage_system",
        "jvm.cpu.recent_utilization": "cpu_usage_user",
        "container.memory.usage": "memory_usage",
        "container.memory.working_set": "memory_working_set",
        "k8s.pod.network.io": "rx_bytes",  # Will be derived
        "k8s.pod.network.io_tx": "tx_bytes",  # Will be derived
    }

    def __init__(
        self,
        chunk_length: int = 60,
        test_ratio: float = 0.2,
        overlap_threshold: int = 1,
        drain_config: Optional[Path] = None,
    ):
        self.chunk_length = chunk_length
        self.test_ratio = test_ratio
        self.overlap_threshold = overlap_threshold
        self.drain_config = drain_config

        # Will be initialized when processing data
        self.service_info: Optional[ServiceInfo] = None
        self.service2nid: Dict[str, int] = {}
        self.log_templates: List[str] = []
        self.event2id: Dict[str, int] = {}
        self.template_miner: Optional[TemplateMiner] = None  # Reuse miner

    def convert_dataset(
        self,
        rcabench_root: Path,
        output_dir: Optional[Path] = None,
    ) -> EadroDataset:
        """Convert entire RCABench dataset to Eadro format.

        IMPORTANT: Train/test split is done at CASE level to avoid data leakage.
        Each case is processed independently into chunks, and cases are split
        into train/test sets before chunking.

        Args:
            rcabench_root: Root directory containing RCABench case folders
            output_dir: Optional directory to save the converted dataset

        Returns:
            EadroDataset ready for training
        """
        logging.info(f"Converting RCABench dataset from {rcabench_root}")

        # Get all case directories
        case_dirs = sorted([d for d in rcabench_root.iterdir() if d.is_dir()])
        if not case_dirs:
            raise ValueError(f"No case directories found in {rcabench_root}")

        logging.info(f"Found {len(case_dirs)} cases")

        # Step 1: Split cases into train/test BEFORE processing (avoid data leakage)
        np.random.seed(42)
        indices = np.random.permutation(len(case_dirs))
        n_test = max(1, int(len(case_dirs) * self.test_ratio))
        n_train = len(case_dirs) - n_test

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_cases = [case_dirs[i] for i in train_indices]
        test_cases = [case_dirs[i] for i in test_indices]

        logging.info(f"Split: {n_train} train cases, {n_test} test cases")

        # Step 2: Initialize service topology from first case
        self._init_service_info(case_dirs[0])

        # Step 3: Extract log templates from TRAIN cases only (avoid test leakage)
        self._extract_log_templates(train_cases)

        # Step 4: Process train cases into chunks
        logging.info(f"\n{'=' * 80}")
        logging.info("Processing TRAIN cases")
        logging.info(f"{'=' * 80}")
        train_chunks = self._process_cases(train_cases, "train")

        # Step 5: Process test cases into chunks
        logging.info(f"\n{'=' * 80}")
        logging.info("Processing TEST cases")
        logging.info(f"{'=' * 80}")
        test_chunks = self._process_cases(test_cases, "test")

        if not train_chunks and not test_chunks:
            raise ValueError("No chunks generated from dataset")

        logging.info(
            f"\nTotal: {len(train_chunks)} train chunks, {len(test_chunks)} test chunks"
        )

        # Step 6: Create metadata
        reference_chunk = train_chunks[0] if train_chunks else test_chunks[0]
        metadata = self._create_metadata(reference_chunk)

        # Step 7: Create dataset
        dataset = EadroDataset(
            metadata=metadata,
            train=DatasetSplit(chunks=tuple(train_chunks)),
            test=DatasetSplit(chunks=tuple(test_chunks)),
        )
        dataset.validate()

        # Print statistics
        self._print_statistics(dataset)

        # Step 8: Save if output_dir provided
        if output_dir:
            from ..data.io import save_preprocessed_dataset

            output_dir.mkdir(parents=True, exist_ok=True)
            save_preprocessed_dataset(dataset, output_dir)
            logging.info(f"Saved dataset to {output_dir}")

        return dataset

    def _init_service_info(self, sample_case: Path) -> None:
        """Initialize service topology from a sample case."""
        logging.info("Initializing service topology...")

        # Read service names from logs
        logs_df = pd.read_parquet(sample_case / "abnormal_logs.parquet")
        services = sorted(logs_df["service_name"].unique())

        # Filter out empty service names
        services = [s for s in services if s and s.strip()]

        logging.info(f"Found {len(services)} services")

        # Detect benchmark type
        if any("ts-" in s for s in services):
            benchmark = "TrainTicket"
        elif any("social" in s.lower() for s in services):
            benchmark = "SocialNetwork"
        else:
            benchmark = "TrainTicket"  # Default

        logging.info(f"Detected benchmark: {benchmark}")

        # Use ServiceInfo for topology
        self.service_info = ServiceInfo(benchmark)
        self.service2nid = self.service_info.service2nid

        logging.info(f"Initialized {self.service_info.node_num} nodes")

    def _extract_log_templates(self, case_dirs: List[Path]) -> None:
        """Extract log templates using Drain3."""
        logging.info("\nExtracting log templates from all cases...")

        # Collect sample log messages (limit to avoid memory issues)
        all_logs: List[str] = []
        max_logs_per_case = 5000  # Limit logs per case

        for case_dir in case_dirs:
            logs_path = case_dir / "abnormal_logs.parquet"
            if not logs_path.exists():
                continue

            df = pd.read_parquet(logs_path)
            logs_sample = (
                df["message"].dropna().astype(str).head(max_logs_per_case).tolist()
            )
            all_logs.extend(logs_sample)

        logging.info(f"Collected {len(all_logs)} log messages")

        # Train Drain3
        config = TemplateMinerConfig()
        if self.drain_config and self.drain_config.exists():
            config.load(str(self.drain_config))
        config.profiling_enabled = False

        self.template_miner = TemplateMiner(config=config)
        for idx, log_msg in enumerate(all_logs):
            if idx % 10000 == 0 and idx > 0:
                logging.info(f"Processed {idx}/{len(all_logs)} logs")
            self.template_miner.add_log_message(log_msg)

        # Extract templates
        self.log_templates = []
        for cluster in self.template_miner.drain.clusters:
            self.log_templates.append(cluster.get_template())

        # Build event vocabulary
        self.event2id = {temp: idx + 1 for idx, temp in enumerate(self.log_templates)}
        self.event2id["Unseen"] = 0

        logging.info(f"Extracted {len(self.log_templates)} log templates")

    def _process_cases(
        self, case_dirs: List[Path], split_name: str
    ) -> List[ChunkPayload]:
        """Process multiple cases into chunks.

        Args:
            case_dirs: List of case directories to process
            split_name: Name of split ('train' or 'test') for logging

        Returns:
            List of ChunkPayload objects
        """
        all_chunks: List[ChunkPayload] = []

        for idx, case_dir in enumerate(case_dirs):
            logging.info(f"\n{'-' * 60}")
            logging.info(
                f"[{split_name.upper()}] Processing case {idx + 1}/{len(case_dirs)}: {case_dir.name}"
            )
            logging.info(f"{'-' * 60}")

            try:
                chunks = self._process_case(case_dir, case_dir.name)
                all_chunks.extend(chunks)
                logging.info(f"Generated {len(chunks)} chunks from {case_dir.name}")
            except Exception as e:
                logging.error(f"Failed to process {case_dir.name}: {e}", exc_info=True)
                continue

        return all_chunks

    def _process_case(self, case_dir: Path, case_id: str) -> List[ChunkPayload]:
        """Process a single RCABench case into chunks.

        Modified to generate NON-OVERLAPPING chunks to avoid data leakage.
        """

        # Load injection metadata
        with open(case_dir / "injection.json") as f:
            injection = json.load(f)

        # Parse timestamps
        start_time = pd.to_datetime(injection["start_time"])
        end_time = pd.to_datetime(injection["end_time"])

        # Get root cause service
        ground_truth = injection.get("ground_truth", {})
        root_cause_services = ground_truth.get("service", [])

        if root_cause_services:
            root_cause = root_cause_services[0]
            culprit_nid = self.service2nid.get(root_cause, -1)
        else:
            culprit_nid = -1  # No fault

        logging.info(f"Root cause: {root_cause_services} -> node {culprit_nid}")
        logging.info(f"Time range: {start_time} to {end_time}")

        # Load data
        logs_df = pd.read_parquet(case_dir / "abnormal_logs.parquet")
        metrics_df = pd.read_parquet(case_dir / "abnormal_metrics.parquet")
        traces_df = pd.read_parquet(case_dir / "abnormal_traces.parquet")

        # Normalize timestamps to seconds from start
        logs_df["timestamp"] = (
            pd.to_datetime(logs_df["time"]) - start_time
        ).dt.total_seconds()
        metrics_df["timestamp"] = (
            pd.to_datetime(metrics_df["time"]) - start_time
        ).dt.total_seconds()
        traces_df["timestamp"] = (
            pd.to_datetime(traces_df["time"]) - start_time
        ).dt.total_seconds()

        # Generate NON-OVERLAPPING time windows
        # This reduces data leakage risk compared to sliding windows
        duration = int((end_time - start_time).total_seconds())
        intervals = [
            (s, s + self.chunk_length - 1)
            for s in range(
                0, duration - self.chunk_length + 1, self.chunk_length
            )  # Step by chunk_length (no overlap)
        ]

        logging.info(f"Generated {len(intervals)} time intervals")

        # Process each data modality
        logs_features = self._process_logs(logs_df, intervals)
        metrics_features = self._process_metrics(metrics_df, intervals)
        traces_features = self._process_traces(traces_df, intervals)

        # Create chunks
        chunks = []
        for idx, (start, end) in enumerate(intervals):
            chunk_id = f"{case_id}_chunk_{idx}"

            chunk = ChunkPayload(
                chunk_id=chunk_id,
                logs=logs_features[idx],
                metrics=metrics_features[idx],
                traces=traces_features[idx],
                culprit=culprit_nid,
            )
            chunk.validate()
            chunks.append(chunk)

        return chunks

    def _process_logs(
        self,
        logs_df: pd.DataFrame,
        intervals: List[Tuple[int, int]],
    ) -> npt.NDArray[np.float32]:
        """Process logs using simplified event counting (faster than Hawkes).

        Instead of fitting Hawkes process (very slow), we use simple event counts
        as features, which is much faster and still informative.

        Returns:
            Array of shape (num_chunks, num_nodes, event_dim)
        """
        logging.info("Processing logs...")

        event_num = len(self.log_templates) + 1
        node_num = self.service_info.node_num
        result = np.zeros((len(intervals), node_num, event_num), dtype=np.float32)

        # Pre-match all log messages (vectorized, do once)
        if "event_id" not in logs_df.columns:
            logging.info("Pre-matching log templates (one-time operation)...")
            miner = self.template_miner

            event_ids = []
            for msg in logs_df["message"]:
                match = miner.match(str(msg))
                if match:
                    template = match.get_template()
                    eid = self.event2id.get(template, 0)
                else:
                    eid = 0  # Unseen
                event_ids.append(eid)

            logs_df["event_id"] = event_ids
            logging.info("Template matching complete")

        # Process each chunk (vectorized)
        for chunk_idx, (start, end) in enumerate(intervals):
            if chunk_idx % 5 == 0:
                logging.info(f"Processing chunk {chunk_idx + 1}/{len(intervals)}")

            chunk_logs = logs_df[
                (logs_df["timestamp"] >= start) & (logs_df["timestamp"] <= end)
            ]

            # Group by service and count events (vectorized)
            for service, group in chunk_logs.groupby("service_name"):
                if service not in self.service2nid:
                    continue

                nid = self.service2nid[service]

                # Count each event type (vectorized)
                event_counts = group["event_id"].value_counts()

                # Normalize by log(count + 1) to reduce scale
                for eid, count in event_counts.items():
                    if 0 <= eid < event_num:
                        result[chunk_idx, nid, eid] = np.log1p(count)

        return result

    def _process_metrics(
        self,
        metrics_df: pd.DataFrame,
        intervals: List[Tuple[int, int]],
    ) -> npt.NDArray[np.float32]:
        """Process system metrics (optimized with vectorization).

        Returns:
            Array of shape (num_chunks, num_nodes, chunk_length, metric_dim)
        """
        logging.info("Processing metrics...")

        metric_names = self.service_info.metric_names
        metric_num = len(metric_names)
        node_num = self.service_info.node_num

        result = np.zeros(
            (len(intervals), node_num, self.chunk_length, metric_num),
            dtype=np.float32,
        )

        # Pre-process: add time_offset column
        # (will be updated per chunk, but column structure is ready)

        # Map RCABench metric names to Eadro indices
        rcabench_to_eadro = {
            "container.cpu.usage": 0,  # cpu_usage_total
            "k8s.pod.cpu.usage": 1,  # cpu_usage_system
            "jvm.cpu.recent_utilization": 2,  # cpu_usage_user
            "container.memory.usage": 3,  # memory_usage
            "container.memory.working_set": 4,  # memory_working_set
        }

        # Process each chunk
        for chunk_idx, (start, end) in enumerate(intervals):
            if chunk_idx % 5 == 0:
                logging.info(f"Processing chunk {chunk_idx + 1}/{len(intervals)}")

            chunk_metrics = metrics_df[
                (metrics_df["timestamp"] >= start) & (metrics_df["timestamp"] <= end)
            ].copy()

            if chunk_metrics.empty:
                continue

            # Add time offset
            chunk_metrics["time_offset"] = (chunk_metrics["timestamp"] - start).astype(
                int
            )
            chunk_metrics = chunk_metrics[
                chunk_metrics["time_offset"] < self.chunk_length
            ]

            # Group by service and metric type (vectorized)
            for (service, metric_type), group in chunk_metrics.groupby(
                ["service_name", "metric"]
            ):
                if service not in self.service2nid:
                    continue
                if metric_type not in rcabench_to_eadro:
                    continue

                nid = self.service2nid[str(service)]
                metric_idx = rcabench_to_eadro[metric_type]

                # Aggregate by time_offset (average if multiple values per second)
                time_values = group.groupby("time_offset")["value"].mean().to_dict()

                for time_offset, value in time_values.items():
                    if 0 <= time_offset < self.chunk_length:
                        result[chunk_idx, nid, time_offset, metric_idx] = float(value)

        # Normalize each metric across all chunks (vectorized)
        logging.info("Normalizing metrics...")
        for metric_idx in range(metric_num):
            for nid in range(node_num):
                data = result[:, nid, :, metric_idx]
                result[:, nid, :, metric_idx] = _z_score_normalize(data)

        return result

    def _process_traces(
        self,
        traces_df: pd.DataFrame,
        intervals: List[Tuple[int, int]],
    ) -> npt.NDArray[np.float32]:
        """Process distributed traces.

        Returns:
            Array of shape (num_chunks, num_nodes, chunk_length, 2)
            Channel 0: average latency, Channel 1: invocation count
        """
        logging.info("Processing traces...")

        node_num = self.service_info.node_num
        result = np.zeros(
            (len(intervals), node_num, self.chunk_length, 2),
            dtype=np.float32,
        )

        # Calculate span duration (end_time - start_time)
        if (
            "attr.span_start_time" in traces_df.columns
            and "attr.span_end_time" in traces_df.columns
        ):
            traces_df["duration"] = pd.to_datetime(
                traces_df["attr.span_end_time"]
            ) - pd.to_datetime(traces_df["attr.span_start_time"])
            traces_df["duration_ms"] = traces_df["duration"].dt.total_seconds() * 1000
        else:
            # Fallback: use a default duration
            traces_df["duration_ms"] = 10.0

        for chunk_idx, (start, end) in enumerate(intervals):
            chunk_traces = traces_df[
                (traces_df["timestamp"] >= start) & (traces_df["timestamp"] <= end)
            ]

            # Group by service and time
            for service, group in chunk_traces.groupby("service_name"):
                if service not in self.service2nid:
                    continue

                nid = self.service2nid[service]

                # Group by second
                group = group.copy()
                group["time_offset"] = (group["timestamp"] - start).astype(int)
                group = group[group["time_offset"] < self.chunk_length]

                for time_offset, time_group in group.groupby("time_offset"):
                    time_idx = int(time_offset)
                    if 0 <= time_idx < self.chunk_length:
                        result[chunk_idx, nid, time_idx, 0] = time_group[
                            "duration_ms"
                        ].mean()
                        result[chunk_idx, nid, time_idx, 1] = len(time_group)

        # Normalize latency channel
        for nid in range(node_num):
            result[:, nid, :, 0] = _z_score_normalize(result[:, nid, :, 0])

        return result

    def _create_metadata(self, sample_chunk: ChunkPayload) -> DatasetMetadata:
        """Create dataset metadata from a sample chunk."""

        src, dst = self.service_info.edges
        edge_index = (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))

        metadata = DatasetMetadata(
            node_count=self.service_info.node_num,
            event_dim=sample_chunk.logs.shape[-1],
            metric_dim=sample_chunk.metrics.shape[-1],
            trace_channels=sample_chunk.traces.shape[-1],
            chunk_length=self.chunk_length,
            edges=edge_index,
        )
        metadata.validate()

        return metadata

    def _print_statistics(self, dataset: EadroDataset) -> None:
        """Print dataset statistics."""
        from collections import defaultdict

        train_chunks = dataset.train.chunks
        test_chunks = dataset.test.chunks

        train_faulty = sum(1 for c in train_chunks if c.culprit != -1)
        test_faulty = sum(1 for c in test_chunks if c.culprit != -1)

        logging.info(f"\n{'=' * 80}")
        logging.info("Dataset Statistics")
        logging.info(f"{'=' * 80}")
        logging.info(
            f"Train: {len(train_chunks)} chunks ({train_faulty} faulty, "
            f"{100 * train_faulty / len(train_chunks):.1f}%)"
        )
        logging.info(
            f"Test: {len(test_chunks)} chunks ({test_faulty} faulty, "
            f"{100 * test_faulty / len(test_chunks):.1f}%)"
        )

        # Per-node fault statistics
        node_faults_train = defaultdict(int)
        node_faults_test = defaultdict(int)

        for chunk in train_chunks:
            if chunk.culprit > -1:
                node_faults_train[chunk.culprit] += 1

        for chunk in test_chunks:
            if chunk.culprit > -1:
                node_faults_test[chunk.culprit] += 1

        if node_faults_train:
            logging.info("\nTrain set fault distribution:")
            for node_id in sorted(node_faults_train.keys()):
                logging.info(f"  Node {node_id}: {node_faults_train[node_id]} chunks")

        if node_faults_test:
            logging.info("\nTest set fault distribution:")
            for node_id in sorted(node_faults_test.keys()):
                logging.info(f"  Node {node_id}: {node_faults_test[node_id]} chunks")

        logging.info(f"{'=' * 80}\n")
