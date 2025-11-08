"""Align multi-source data into temporal chunks with fault labels."""

from __future__ import annotations

import json
import logging
import pickle
import random
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..data.schema import ChunkPayload, DatasetMetadata, DatasetSplit, EadroDataset
from .processors import fit_hawkes_intensity, ServiceInfo

__all__ = ["ChunkAligner"]


def _z_score_normalize(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Z-score normalization."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


class ChunkGenerator:
    """Generate unique chunk IDs."""

    def __init__(self):
        self._used_ids: Set[str] = set()
        self._chars = string.ascii_letters + string.digits

    def generate(self) -> str:
        """Generate a unique 8-character ID."""
        while True:
            chunk_id = "".join(random.sample(self._chars, 8))
            if chunk_id not in self._used_ids:
                self._used_ids.add(chunk_id)
                return chunk_id


class ChunkAligner:
    """Align logs, metrics, and traces into labeled chunks."""

    def __init__(
        self,
        service_info: ServiceInfo,
        chunk_length: int = 60,
        overlap_threshold: int = 1,
    ):
        """Initialize chunk aligner.

        Args:
            service_info: Service topology metadata
            chunk_length: Duration of each chunk in seconds
            overlap_threshold: Minimum overlap to mark chunk as faulty
        """
        self.service_info = service_info
        self.chunk_length = chunk_length
        self.overlap_threshold = overlap_threshold
        self.chunk_gen = ChunkGenerator()

    def _get_intervals_and_labels(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Generate time intervals and fault labels.

        Args:
            record: Processed fault record

        Returns:
            Tuple of (intervals, labels)
        """
        faults = [
            (f["s"], f["e"], self.service_info.service2nid[f["service"]])
            for f in record["faults"]
        ]
        start, end = record["start"], record["end"]

        # Generate sliding windows
        intervals = [
            (s, s + self.chunk_length - 1)
            for s in range(start, end - self.chunk_length + 1)
        ]
        labels = [-1] * len(intervals)

        # Assign fault labels
        for chunk_idx, (s, e) in enumerate(intervals):
            for fs, fe, culprit in faults:
                overlap = 0
                if s >= fs and s <= fe:
                    overlap = fe - s + 1
                elif e >= fs and e <= fe:
                    overlap = e - fs + 1

                if overlap >= self.overlap_threshold:
                    labels[chunk_idx] = culprit
                if overlap > 0:
                    break

        logging.info(f"Generated {len(intervals)} intervals from {start} to {end}")
        return intervals, labels

    def _process_logs(
        self,
        intervals: List[Tuple[int, int]],
        log_csv: Path,
        templates: List[str],
    ) -> npt.NDArray[np.float32]:
        """Process logs using Hawkes process.

        Args:
            intervals: List of (start, end) tuples
            log_csv: Path to logs CSV
            templates: List of log templates

        Returns:
            Array of shape (num_chunks, num_nodes, event_dim)

        Note:
            Output matches ChunkPayload.logs format:
            (node_count, event_dim) - Hawkes baseline intensity for each event type
        """
        logging.info("Processing logs with Hawkes...")
        df = pd.read_csv(log_csv)

        event_num = len(templates) + 1  # +1 for "Unseen"
        event2id = {temp: idx + 1 for idx, temp in enumerate(templates)}
        event2id["Unseen"] = 0

        result = np.zeros(
            (len(intervals), self.service_info.node_num, event_num), dtype=np.float32
        )

        for chunk_idx, (s, e) in enumerate(intervals):
            if (chunk_idx + 1) % 100 == 0:
                logging.info(f"Processing chunk {chunk_idx + 1}/{len(intervals)}")

            try:
                rows = df.loc[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
            except Exception:
                continue

            service_groups = rows.groupby("service")
            for service, sgroup in service_groups:
                event_groups = sgroup.groupby("events")
                knots = [np.array([0.0]) for _ in range(event_num)]

                for event, egroup in event_groups:
                    # Convert pandas Scalar to str
                    event_str = str(event)
                    eid = event2id.get(event_str, 0)
                    timestamps = np.array(sorted(egroup["timestamp"].values)) - s
                    # Add small jitter to avoid identical timestamps
                    jitter = np.array([idx * 1e-5 for idx in range(len(timestamps))])
                    knots[eid] = timestamps + jitter

                # Fit Hawkes process to get baseline intensities
                paras = fit_hawkes_intensity(knots, end_time=e + 1)
                # Convert pandas Scalar to str
                service_str = str(service)
                result[chunk_idx, self.service_info.service2nid[service_str], :] = paras

        return result

    def _process_metrics(
        self,
        intervals: List[Tuple[int, int]],
        metrics_dir: Path,
    ) -> npt.NDArray[np.float32]:
        """Process system metrics.

        Args:
            intervals: List of (start, end) tuples
            metrics_dir: Directory containing metric CSVs

        Returns:
            Array of shape (num_chunks, num_nodes, chunk_length, metric_dim)

        Note:
            Output matches ChunkPayload.metrics format:
            (node_count, time_steps, metric_dim)
        """
        logging.info("Processing metrics...")
        metric_num = len(self.service_info.metric_names)
        result = np.zeros(
            (len(intervals), self.service_info.node_num, self.chunk_length, metric_num),
            dtype=np.float32,
        )

        for nid, service in enumerate(self.service_info.service_names):
            csv_path = metrics_dir / f"{service}.csv"
            if not csv_path.exists():
                logging.warning(f"Metrics file not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            # Z-score normalize each metric column
            df[self.service_info.metric_names] = df[
                self.service_info.metric_names
            ].apply(_z_score_normalize)
            df.set_index("timestamp", inplace=True)

            for chunk_idx, (s, e) in enumerate(intervals):
                values = df.loc[s:e, :].to_numpy()
                if values.shape[0] != self.chunk_length:
                    logging.warning(
                        f"Expected {self.chunk_length} rows, got {values.shape[0]} for {s}:{e}"
                    )
                    continue
                result[chunk_idx, nid, :, :] = values

        return result

    def _process_traces(
        self,
        intervals: List[Tuple[int, int]],
        traces_json: Path,
    ) -> npt.NDArray[np.float32]:
        """Process distributed traces.

        Args:
            intervals: List of (start, end) tuples
            traces_json: Path to processed traces JSON

        Returns:
            Array of shape (num_chunks, num_nodes, chunk_length, 2)
            Channel 0: average latency (normalized), Channel 1: invocation count

        Note:
            Output matches ChunkPayload.traces format:
            (node_count, time_steps, trace_channels) where trace_channels=2
        """
        logging.info("Processing traces...")
        with traces_json.open("r") as f:
            traces = json.load(f)

        result = np.zeros(
            (len(intervals), self.service_info.node_num, self.chunk_length, 2),
            dtype=np.float32,
        )

        for chunk_idx, (s, e) in enumerate(intervals):
            slots = list(range(s, e + 1))
            for i, ts in enumerate(slots):
                if str(ts) not in traces:
                    continue

                spans = traces[str(ts)]
                tmp_node_lat = [[] for _ in range(self.service_info.node_num)]

                for service_key, lat_list in spans.items():
                    node_id = int(service_key.split("-")[-1])
                    tmp_node_lat[node_id].extend(lat_list)

                for node_id in range(self.service_info.node_num):
                    if tmp_node_lat[node_id]:
                        result[chunk_idx][node_id][i][0] = np.mean(
                            tmp_node_lat[node_id]
                        )
                        result[chunk_idx][node_id][i][1] = len(tmp_node_lat[node_id])

        # Normalize latency channel (channel 0) only
        for i in range(self.service_info.node_num):
            result[:, i, :, 0] = _z_score_normalize(result[:, i, :, 0])

        return result

    def align_chunks(
        self,
        record_idx: int,
        records_dir: Path,
        logs_dir: Path,
        metrics_dir: Path,
        traces_dir: Path,
        templates: List[str],
        output_dir: Optional[Path] = None,
    ) -> List[ChunkPayload]:
        """Align all data sources into chunks.

        Args:
            record_idx: Index of the fault record
            records_dir: Directory containing processed records
            logs_dir: Directory containing processed logs
            metrics_dir: Directory containing processed metrics
            traces_dir: Directory containing processed traces
            templates: List of log templates
            output_dir: Optional directory to save intermediate files

        Returns:
            List of ChunkPayload objects
        """
        # Load record
        record_path = records_dir / f"records{record_idx}.json"
        with record_path.open("r") as f:
            record = json.load(f)

        intervals, labels = self._get_intervals_and_labels(record)

        # Process each data source
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        traces = self._process_traces(
            intervals, traces_dir / f"traces{record_idx}.json"
        )
        metrics = self._process_metrics(intervals, metrics_dir / f"metrics{record_idx}")
        logs = self._process_logs(
            intervals, logs_dir / f"logs{record_idx}.csv", templates
        )

        # Save intermediate files if requested
        if output_dir:
            with (output_dir / "traces.pkl").open("wb") as f:
                pickle.dump({"latency": traces}, f)
            with (output_dir / "metrics.pkl").open("wb") as f:
                pickle.dump(metrics, f)
            with (output_dir / "logs.pkl").open("wb") as f:
                pickle.dump(logs, f)

        # Combine into chunks
        logging.info("Combining data sources into chunks...")
        payloads = []
        for idx in range(len(intervals)):
            chunk_id = self.chunk_gen.generate()
            payload = ChunkPayload(
                chunk_id=chunk_id,
                logs=logs[idx],
                metrics=metrics[idx],
                traces=traces[idx],
                culprit=labels[idx],
            )
            payload.validate()
            payloads.append(payload)

        return payloads

    def process_all_records(
        self,
        records_dir: Path,
        logs_dir: Path,
        metrics_dir: Path,
        traces_dir: Path,
        templates: List[str],
        output_dir: Path,
        test_ratio: float = 0.2,
    ) -> EadroDataset:
        """Process all fault records into chunks.

        Args:
            records_dir: Directory with processed records
            logs_dir: Directory with processed logs
            metrics_dir: Directory with processed metrics
            traces_dir: Directory with processed traces
            templates: Log templates
            output_dir: Output directory for chunks
            test_ratio: Proportion for test set (default: 0.2)

        Returns:
            EadroDataset with train and test splits
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_payloads: List[ChunkPayload] = []

        record_idx = 0
        while (records_dir / f"records{record_idx}.json").exists():
            logging.info(f"\n{'=' * 20} Processing record {record_idx} {'=' * 20}")

            chunk_output = output_dir / str(record_idx)
            chunks = self.align_chunks(
                record_idx,
                records_dir,
                logs_dir,
                metrics_dir,
                traces_dir,
                templates,
                chunk_output,
            )
            all_payloads.extend(chunks)
            record_idx += 1

        logging.info(
            f"Processed {record_idx} records, generated {len(all_payloads)} chunks"
        )

        # Create metadata
        if all_payloads:
            first_chunk = all_payloads[0]
            # Convert edge lists to numpy arrays
            src, dst = self.service_info.edges
            edge_index = (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))

            metadata = DatasetMetadata(
                node_count=self.service_info.node_num,
                event_dim=first_chunk.logs.shape[-1],
                metric_dim=first_chunk.metrics.shape[-1],
                trace_channels=first_chunk.traces.shape[-1],
                chunk_length=self.chunk_length,
                edges=edge_index,
            )
        else:
            raise ValueError("No chunks generated!")

        # Split train/test
        logging.info("\nSplitting chunks into train/test sets...")
        np.random.seed(42)
        indices = np.arange(len(all_payloads))
        np.random.shuffle(indices)

        train_num = int((1 - test_ratio) * len(all_payloads))
        train_indices = indices[:train_num]
        test_indices = indices[train_num:]

        train_chunks = [all_payloads[i] for i in train_indices]
        test_chunks = [all_payloads[i] for i in test_indices]

        # Print statistics
        train_faulty = sum(1 for c in train_chunks if c.culprit != -1)
        test_faulty = sum(1 for c in test_chunks if c.culprit != -1)

        logging.info(
            f"Train: {train_faulty}/{len(train_chunks)} ({100 * train_faulty / len(train_chunks):.2f}%) faulty"
        )
        logging.info(
            f"Test: {test_faulty}/{len(test_chunks)} ({100 * test_faulty / len(test_chunks):.2f}%) faulty"
        )
        logging.info(f"Total: {train_faulty + test_faulty}/{len(all_payloads)} faulty")

        # Print per-node statistics
        node_faults = defaultdict(int)
        for chunk in all_payloads:
            if chunk.culprit > -1:
                node_faults[chunk.culprit] += 1

        for node_id in sorted(node_faults.keys()):
            logging.info(f"Node {node_id}: {node_faults[node_id]} faulty chunks")

        return EadroDataset(
            metadata=metadata,
            train=DatasetSplit(chunks=train_chunks),
            test=DatasetSplit(chunks=test_chunks),
        )

    @staticmethod
    def split_train_test(
        chunks: Dict[str, Dict[str, Any]],
        output_dir: Path,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """Split chunks into train and test sets.

        Args:
            chunks: Dictionary of all chunks
            output_dir: Output directory
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
        """
        logging.info("\nSplitting chunks into train/test sets...")
        np.random.seed(random_seed)

        chunk_ids = np.array(list(chunks.keys()))
        chunk_num = len(chunk_ids)
        indices = np.arange(chunk_num)
        np.random.shuffle(indices)

        train_num = int((1 - test_ratio) * chunk_num)
        test_num = int(test_ratio * chunk_num)

        train_indices = indices[:train_num]
        test_indices = indices[train_num : train_num + test_num]

        train_chunks = {chunk_ids[i]: chunks[chunk_ids[i]] for i in train_indices}
        test_chunks = {chunk_ids[i]: chunks[chunk_ids[i]] for i in test_indices}

        with (output_dir / "chunk_train.pkl").open("wb") as f:
            pickle.dump(train_chunks, f)
        with (output_dir / "chunk_test.pkl").open("wb") as f:
            pickle.dump(test_chunks, f)

        # Print statistics
        train_faulty = sum(1 for c in train_chunks.values() if c["culprit"] != -1)
        test_faulty = sum(1 for c in test_chunks.values() if c["culprit"] != -1)

        logging.info(
            f"Train: {train_faulty}/{train_num} ({100 * train_faulty / train_num:.2f}%) faulty"
        )
        logging.info(
            f"Test: {test_faulty}/{test_num} ({100 * test_faulty / test_num:.2f}%) faulty"
        )
        logging.info(f"Total: {train_faulty + test_faulty}/{chunk_num} faulty")

        # Print per-node statistics
        node_faults = defaultdict(int)
        for chunk in chunks.values():
            if chunk["culprit"] > -1:
                node_faults[chunk["culprit"]] += 1

        for node_id in sorted(node_faults.keys()):
            logging.info(f"Node {node_id}: {node_faults[node_id]} faulty chunks")

    @staticmethod
    def save_legacy_format(dataset: EadroDataset, output_dir: Path) -> None:
        """Save dataset in legacy dictionary format for backward compatibility.

        Args:
            dataset: EadroDataset to convert
            output_dir: Output directory

        Note:
            This creates chunks.pkl in the old format where chunks are stored as
            a dictionary with chunk IDs as keys.
        """
        logging.info("\nSaving legacy format (chunks.pkl)...")

        # Combine all chunks into a single dictionary
        all_chunks = {}
        for chunk in list(dataset.train.chunks) + list(dataset.test.chunks):
            all_chunks[chunk.chunk_id] = {
                "traces": chunk.traces,
                "metrics": chunk.metrics,
                "logs": chunk.logs,
                "culprit": chunk.culprit,
            }

        # Save combined chunks
        chunks_path = output_dir / "chunks.pkl"
        with chunks_path.open("wb") as f:
            pickle.dump(all_chunks, f)

        logging.info(f"Saved {len(all_chunks)} chunks to {chunks_path}")
