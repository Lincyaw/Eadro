"""Data processors for logs, metrics, traces, and fault injection records.

This module consolidates all data processing components for microservice
monitoring data preprocessing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tick.hawkes import HawkesADM4

__all__ = [
    "ServiceInfo",
    "LogTemplateExtractor",
    "MetricsProcessor",
    "TracesProcessor",
    "RecordsProcessor",
    "fit_hawkes_intensity",
]


# ============================================================================
# Service Information
# ============================================================================


class ServiceInfo:
    """Service topology and configuration for a microservice benchmark.

    Args:
        benchmark: Benchmark name ('TrainTicket' or 'SocialNetwork')
    """

    def __init__(self, benchmark: str = "TrainTicket"):
        benchmark = benchmark.lower()

        if benchmark == "trainticket":
            self._init_trainticket()
        elif benchmark == "socialnetwork":
            self._init_socialnetwork()
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.metric_names = [
            "cpu_usage_system",
            "cpu_usage_total",
            "cpu_usage_user",
            "memory_usage",
            "memory_working_set",
            "rx_bytes",
            "tx_bytes",
        ]

        self.service2nid = {s: idx for idx, s in enumerate(self.service_names)}
        self.node_num = len(self.service_names)
        self._compute_edges()

        self.metadata: Dict[str, Any] = {
            "node_num": self.node_num,
            "metric_num": len(self.metric_names),
        }

    def _init_trainticket(self) -> None:
        """Initialize Train Ticket service topology."""
        tmp_apis = [
            "assurance",
            "auth",
            "basic",
            "cancel",
            "config",
            "contacts",
            "food-map",
            "food",
            "inside-payment",
            "notification",
            "order-other",
            "order",
            "payment",
            "preserve",
            "price",
            "route-plan",
            "route",
            "seat",
            "security",
            "station",
            "ticketinfo",
            "train",
            "travel-plan",
            "travel",
            "travel2",
            "user",
            "verification-code",
        ]
        self.service_names = [f"ts-{api}-service" for api in tmp_apis]

        edge_info = {
            "preserve": [
                "preserve",
                "seat",
                "security",
                "food",
                "order",
                "ticketinfo",
                "travel",
                "contacts",
                "notification",
                "user",
                "station",
            ],
            "seat": ["seat", "order", "config", "travel"],
            "cancel": ["inside-payment", "order-other", "order"],
            "security": ["security", "order-other", "order"],
            "food": ["travel", "food-map", "station"],
            "travel": ["travel", "order", "ticketinfo", "train", "route"],
            "inside-payment": ["payment", "order"],
            "ticketinfo": ["ticketinfo", "basic"],
            "basic": ["basic", "route", "price", "train", "station"],
            "order-other": ["station"],
            "order": ["order", "station", "assurance"],
            "auth": ["auth", "verification-code"],
        }
        self.edge_info = {
            f"ts-{k}-service": [f"ts-{vi}-service" for vi in v]
            for k, v in edge_info.items()
        }

    def _init_socialnetwork(self) -> None:
        """Initialize Social Network service topology."""
        self.service_names = [
            "social-graph-service",
            "compose-post-service",
            "post-storage-service",
            "user-timeline-service",
            "url-shorten-service",
            "user-service",
            "media-service",
            "text-service",
            "unique-id-service",
            "user-mention-service",
            "home-timeline-service",
            "nginx-web-server",
        ]
        self.edge_info = {
            "compose-post-service": [
                "compose-post-service",
                "home-timeline-service",
                "media-service",
                "post-storage-service",
                "text-service",
                "unique-id-service",
                "user-service",
                "user-timeline-service",
            ],
            "home-timeline-service": [
                "home-timeline-service",
                "post-storage-service",
                "social-graph-service",
            ],
            "post-storage-service": ["post-storage-service"],
            "social-graph-service": ["social-graph-service", "user-service"],
            "text-service": [
                "text-service",
                "url-shorten-service",
                "user-mention-service",
            ],
            "user-service": ["user-service"],
            "user-timeline-service": ["user-timeline-service"],
            "nginx-web-server": [
                "compose-post-service",
                "home-timeline-service",
                "nginx-web-server",
                "social-graph-service",
                "user-service",
            ],
        }

    def _compute_edges(self) -> None:
        """Compute edge list from service dependencies."""
        src, dst = [], []
        for s, targets in self.edge_info.items():
            sid = self.service2nid[s]
            for t in targets:
                src.append(sid)
                dst.append(self.service2nid[t])
        self.edges: Tuple[List[int], List[int]] = (src, dst)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add additional metadata field."""
        self.metadata[key] = value


# ============================================================================
# Log Processing
# ============================================================================


class LogTemplateExtractor:
    """Extract log templates using Drain3 algorithm."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize template extractor.

        Args:
            config_path: Path to drain3.ini config file
        """
        self.config_path = config_path

    def _extract_timestamp(self, log: str) -> Optional[float]:
        """Extract Unix timestamp from log string."""
        time_pattern = r"\d{4}-(?:[A-Za-z]{3}|\d{2})-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
        match = re.search(time_pattern, log)
        if not match:
            return None

        time_str = match.group(0)
        try:
            unix_time = pd.to_datetime(
                time_str, format="%Y-%b-%d %H:%M:%S.%f"
            ).timestamp()
        except ValueError:
            unix_time = pd.to_datetime(
                time_str, format="%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()

        return unix_time

    def extract_templates(
        self,
        fault_free_path: Path,
        fault_data_path: Path,
        output_dir: Path,
    ) -> List[str]:
        """Extract templates from fault-free data and parse fault data.

        Args:
            fault_free_path: Path to fault-free dataset (for template learning)
            fault_data_path: Path to fault injection data
            output_dir: Directory to save processed logs

        Returns:
            List of extracted templates
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all fault-free logs for template learning
        logging.info("Collecting fault-free logs for template extraction...")
        subfolders = [f for f in fault_free_path.iterdir() if f.is_dir()]
        all_log_strs = []

        for folder in subfolders:
            log_file = folder / "logs.json"
            if not log_file.exists():
                continue
            with log_file.open("r") as f:
                logs = json.load(f)
            for service_logs in logs.values():
                all_log_strs.extend(service_logs)

        # Train Drain3 on fault-free logs
        logging.info(f"Training Drain3 on {len(all_log_strs)} log lines...")
        config = TemplateMinerConfig()
        if self.config_path and self.config_path.exists():
            config.load(str(self.config_path))
        config.profiling_enabled = True

        miner = TemplateMiner(config=config)
        for log_str in all_log_strs:
            miner.add_log_message(log_str)

        templates = []
        for cluster in miner.drain.clusters:
            templates.append(cluster.get_template())

        logging.info(f"Extracted {len(templates)} templates")

        # Save templates
        template_path = output_dir / "templates.json"
        with template_path.open("w") as f:
            json.dump(templates, f, indent=2)

        # Parse fault injection logs
        logging.info("Parsing fault injection logs...")
        fault_folders = sorted([f for f in fault_data_path.iterdir() if f.is_dir()])

        for idx, folder in enumerate(fault_folders):
            log_file = folder / "logs.json"
            if not log_file.exists():
                logging.warning(f"No logs.json in {folder}")
                continue

            with log_file.open("r") as f:
                log_dict = json.load(f)

            df_data = {"timestamp": [], "service": [], "events": []}

            for service, log_list in log_dict.items():
                for log_line in log_list:
                    match = miner.match(log_line)
                    if match:
                        log_temp = match.get_template()
                    else:
                        log_temp = "Unseen"

                    log_time = self._extract_timestamp(log_line)
                    if log_time is None:
                        logging.warning(
                            f"Could not extract timestamp: {log_line[:100]}"
                        )
                        continue

                    df_data["timestamp"].append(log_time)
                    df_data["service"].append(service)
                    df_data["events"].append(log_temp)

            df = pd.DataFrame(df_data)
            output_csv = output_dir / f"logs{idx}.csv"
            df.to_csv(output_csv, index=False)
            logging.info(f"Saved {len(df)} log entries to logs{idx}.csv")

        return templates


# ============================================================================
# Metrics Processing
# ============================================================================


class MetricsProcessor:
    """Process system metrics from CSV files."""

    def __init__(self, service_info: ServiceInfo):
        """Initialize metrics processor.

        Args:
            service_info: Service topology information
        """
        self.service_info = service_info

    def process(self, data_folders: List[Path], output_dir: Path) -> None:
        """Process metrics from multiple data folders.

        Args:
            data_folders: List of folders containing metrics/ subdirectories
            output_dir: Base directory for output
        """
        sorted_folders = sorted(data_folders)

        for idx, folder in enumerate(sorted_folders):
            metrics_path = folder / "metrics"
            if not metrics_path.exists():
                logging.warning(f"No metrics folder in {folder}")
                continue

            metric_files = list(metrics_path.glob("*.csv"))
            if not metric_files:
                logging.warning(f"No CSV files in {metrics_path}")
                continue

            output_subdir = output_dir / f"metrics{idx}"
            output_subdir.mkdir(parents=True, exist_ok=True)

            for metric_file in metric_files:
                df = pd.read_csv(metric_file)

                # Select relevant columns
                required_cols = ["timestamp"] + self.service_info.metric_names
                df = df[required_cols]

                # Adjust timestamp (add 16 hours)
                df["timestamp"] = df["timestamp"] + 16 * 60 * 60

                # Save processed metrics
                output_path = output_subdir / metric_file.name
                df.to_csv(output_path, index=False)

            logging.info(f"Processed {len(metric_files)} metric files to metrics{idx}/")


# ============================================================================
# Traces Processing
# ============================================================================


class TracesProcessor:
    """Process Jaeger trace spans into aggregated statistics."""

    def __init__(self, service_info: ServiceInfo):
        """Initialize traces processor.

        Args:
            service_info: Service topology information
        """
        self.service_info = service_info

    def process(self, span_files: List[Path], output_dir: Path) -> None:
        """Process trace span files.

        Args:
            span_files: List of spans.json files
            output_dir: Directory for output
        """
        sorted_files = sorted(span_files)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, span_file in enumerate(sorted_files):
            if not span_file.exists():
                logging.warning(f"Span file not found: {span_file}")
                continue

            with span_file.open("r") as f:
                data = json.load(f)

            # Parse spans and aggregate by timestamp
            t_s_lat = []  # (timestamp, service, latency) tuples

            for trace in data:
                processes = trace["processes"]
                for span in trace["spans"]:
                    service_name = processes[span["processID"]]["serviceName"]
                    service_key = (
                        f"{service_name}-{self.service_info.service2nid[service_name]}"
                    )

                    # Convert to seconds and adjust timezone
                    start_time = int(span["startTime"]) // 1_000_000
                    start_time += 8 * 3600  # Add 8 hours offset

                    latency = int(span["duration"]) / 1_000_000  # microseconds to ms
                    t_s_lat.append((start_time, service_key, latency))

            # Aggregate by timestamp
            unique_timestamps = sorted(set(t for t, _, _ in t_s_lat))
            processed_data: Dict[str, Dict[str, List[float]]] = {
                str(t): {} for t in unique_timestamps
            }

            for t, s, lat in t_s_lat:
                ts_key = str(t)
                if s not in processed_data[ts_key]:
                    processed_data[ts_key][s] = []
                processed_data[ts_key][s].append(lat)

            # Save processed traces
            output_path = output_dir / f"traces{idx}.json"
            with output_path.open("w") as f:
                json.dump(processed_data, f, indent=2)

            logging.info(
                f"Processed traces{idx}.json with {len(unique_timestamps)} timestamps"
            )


# ============================================================================
# Records Processing
# ============================================================================


class RecordsProcessor:
    """Process raw fault injection records.

    Parses fault metadata including timestamps and affected services.
    """

    def __init__(self, dataset_name: str):
        """Initialize processor.

        Args:
            dataset_name: 'TT' or 'SN'
        """
        self.dataset_name = dataset_name

    def _parse_service_name(self, raw_name: str) -> str:
        """Convert container name to service name."""
        if self.dataset_name == "SN":
            mapping = {
                "socialnetwork-text-service-1": "text-service",
                "socialnetwork-home-timeline-service-1": "home-timeline-service",
                "socialnetwork-media-service-1": "media-service",
                "socialnetwork-post-storage-service-1": "post-storage-service",
                "socialnetwork-social-graph-service-1": "social-graph-service",
                "socialnetwork-url-shorten-service-1": "url-shorten-service",
                "socialnetwork-nginx-thrift-1": "nginx-web-server",
                "socialnetwork-unique-id-service-1": "unique-id-service",
                "socialnetwork-user-service-1": "user-service",
                "socialnetwork-compose-post-service-1": "compose-post-service",
                "socialnetwork-user-timeline-service-1": "user-timeline-service",
                "socialnetwork-user-mention-service-1": "user-mention-service",
            }
            if raw_name not in mapping:
                raise KeyError(f"Unknown service: {raw_name}")
            return mapping[raw_name]
        else:  # TT
            return raw_name.split("_")[1]

    def process(self, data_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process all fault record files.

        Args:
            data_path: Directory containing fault JSON files
            output_dir: Directory to save processed records

        Returns:
            List of processed record metadata
        """
        record_files = sorted(data_path.glob("*.json"))
        if not record_files:
            raise ValueError(f"No JSON files found in {data_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        processed = []

        for idx, file_path in enumerate(record_files):
            with file_path.open("r") as f:
                records = json.load(f)

            # Adjust timestamps (add 16 hours offset)
            # See: https://github.com/BEbillionaireUSD/Eadro/issues/11
            processed_record = {
                "start": int(records["start"]) + 16 * 3600,
                "end": int(records["end"]) + 16 * 3600,
                "faults": [],
            }

            for fault in records["faults"]:
                service = self._parse_service_name(fault["name"])
                fault_type = fault["fault"]
                start = int(fault["start"]) + 16 * 3600
                end = start + int(fault["duration"])

                processed_record["faults"].append(
                    {
                        "service": service,
                        "fault_type": fault_type,
                        "s": start,
                        "e": end,
                    }
                )

            output_path = output_dir / f"records{idx}.json"
            with output_path.open("w") as f:
                json.dump(processed_record, f, indent=2)

            processed.append(processed_record)
            logging.info(f"Processed record {idx}: {file_path.name}")

        return processed


# ============================================================================
# Hawkes Process Modeling
# ============================================================================


def fit_hawkes_intensity(
    event_sequences: list[npt.NDArray[np.float64]],
    end_time: float,
    decay: float = 3.0,
    initial_intensity: float = 0.2,
) -> npt.NDArray[np.float64]:
    """Fit Hawkes process to estimate base event intensities.

    Args:
        event_sequences: List of event timestamp arrays for each event type
        end_time: End of observation window
        decay: Exponential decay parameter
        initial_intensity: Initial baseline intensity guess

    Returns:
        Estimated baseline intensities for each event type
    """
    event_num = len(event_sequences)
    model = HawkesADM4(decay)
    baseline_start = np.ones(event_num) * initial_intensity
    model.fit(event_sequences, end_time, baseline_start=baseline_start)
    return np.array(model.baseline)
