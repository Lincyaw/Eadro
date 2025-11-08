from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "EdgeIndex",
    "ChunkPayload",
    "DatasetMetadata",
    "DatasetSplit",
    "EadroDataset",
]

# Type alias for graph edges: (source_nodes, target_nodes)
EdgeIndex = Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]


@dataclass(frozen=True)
class ChunkPayload:
    """Intermediate representation for a single temporal graph window.

    This is the standardized data format that all custom datasets must be converted to.
    It represents a time window of microservice monitoring data across multiple sources.

    Attributes:
        chunk_id: Unique identifier for this time window chunk.
        logs: Log event features aggregated per node.
              Shape: (node_count, event_dim)
              - node_count: number of microservices/nodes in the system
              - event_dim: dimension of log event embeddings (e.g., one-hot encoded events)
        metrics: Time-series metric data per node.
                 Shape: (node_count, time_steps, metric_dim)
                 - time_steps: length of the temporal window
                 - metric_dim: number of different metrics (CPU, memory, etc.)
        traces: Distributed tracing statistics per node.
                Shape: (node_count, time_steps, trace_channels)
                - trace_channels: typically 2 (call count and response time statistics)
        culprit: Index of the root cause node (faulty microservice).
                 Use -1 for healthy windows with no fault.

    Example:
        >>> import numpy as np
        >>> chunk = ChunkPayload(
        ...     chunk_id="chunk_001",
        ...     logs=np.zeros((10, 50), dtype=np.float32),      # 10 nodes, 50 event types
        ...     metrics=np.zeros((10, 60, 8), dtype=np.float32), # 10 nodes, 60 timesteps, 8 metrics
        ...     traces=np.zeros((10, 60, 2), dtype=np.float32),  # 10 nodes, 60 timesteps, 2 channels
        ...     culprit=3  # node 3 is the root cause
        ... )
        >>> chunk.validate()  # Ensures data consistency
    """

    chunk_id: str
    logs: npt.NDArray[np.float32]
    metrics: npt.NDArray[np.float32]
    traces: npt.NDArray[np.float32]
    culprit: int

    def validate(self) -> None:
        """Ensure consistent array ranks and matching node counts.

        Raises:
            ValueError: If array dimensions don't match the expected shapes.
        """

        if self.logs.ndim != 2:
            raise ValueError("logs must be 2D: (node_count, event_dim)")
        if self.metrics.ndim != 3:
            raise ValueError("metrics must be 3D: (node_count, time_steps, metric_dim)")
        if self.traces.ndim != 3:
            raise ValueError(
                "traces must be 3D: (node_count, time_steps, trace_channels)"
            )

        node_count = self.logs.shape[0]
        if self.metrics.shape[0] != node_count or self.traces.shape[0] != node_count:
            raise ValueError("All feature tensors must share the same node dimension")


@dataclass(frozen=True)
class DatasetMetadata:
    """Structural information shared by all chunks of a dataset.

    This metadata describes the graph topology and feature dimensions that are
    consistent across all time windows in a dataset.

    Attributes:
        node_count: Total number of microservice nodes in the system graph.
        event_dim: Dimension of log event feature vectors (vocabulary size).
        metric_dim: Number of distinct metrics tracked per node.
        trace_channels: Number of trace statistic channels (typically 2).
        chunk_length: Number of time steps in each temporal window.
        edges: Graph connectivity as (source_indices, target_indices).
               Both arrays have shape (num_edges,) and contain node indices.

    Example:
        >>> import numpy as np
        >>> metadata = DatasetMetadata(
        ...     node_count=10,
        ...     event_dim=50,
        ...     metric_dim=8,
        ...     trace_channels=2,
        ...     chunk_length=60,
        ...     edges=(np.array([0, 1, 2]), np.array([1, 2, 3]))  # edges: 0->1, 1->2, 2->3
        ... )
        >>> metadata.validate()
    """

    node_count: int
    event_dim: int
    metric_dim: int
    trace_channels: int
    chunk_length: int
    edges: EdgeIndex

    def validate(self) -> None:
        """Basic sanity checks on graph structure and feature dimensions.

        Raises:
            ValueError: If any structural constraint is violated.
        """

        src, dst = self.edges
        if src.shape != dst.shape:
            raise ValueError("Edge index arrays must share the same shape")
        if src.ndim != 1:
            raise ValueError("Edge indices must be 1D")
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        if self.event_dim <= 0 or self.metric_dim <= 0 or self.trace_channels <= 0:
            raise ValueError("Feature dimensions must be positive")


@dataclass(frozen=True)
class DatasetSplit:
    """Collection of chunks for either the training or evaluation split.

    Attributes:
        chunks: Sequence of chunk payloads in this split.
    """

    chunks: Sequence[ChunkPayload]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.chunks)

    def __iter__(self) -> Iterable[ChunkPayload]:  # pragma: no cover - trivial
        return iter(self.chunks)


@dataclass(frozen=True)
class EadroDataset:
    """Complete dataset bundle with metadata and train/test partitions.

    This is the top-level data structure that combines all components needed
    for training and evaluation.

    Attributes:
        metadata: Shared structural information (topology, dimensions).
        train: Training split with labeled fault injection windows.
        test: Testing split for evaluation.

    Example:
        >>> from eadro.data.io import load_preprocessed_dataset
        >>> from pathlib import Path
        >>> dataset = load_preprocessed_dataset(Path("./chunks/TT"))
        >>> print(f"Train: {len(dataset.train)} chunks")
        >>> print(f"Test: {len(dataset.test)} chunks")
        >>> print(f"Nodes: {dataset.metadata.node_count}")
    """

    metadata: DatasetMetadata
    train: DatasetSplit
    test: DatasetSplit

    def validate(self) -> None:
        """Validate the entire dataset for consistency.

        Checks:
            - Metadata structural constraints
            - All chunks have valid shapes
            - Chunk dimensions match metadata

        Raises:
            ValueError: If any validation fails.
        """
        self.metadata.validate()
        for chunk in (*self.train.chunks, *self.test.chunks):
            chunk.validate()
            if chunk.logs.shape[0] != self.metadata.node_count:
                raise ValueError("Chunk node dimension mismatch")
            if chunk.metrics.shape[1] != self.metadata.chunk_length:
                raise ValueError("Chunk length mismatch for metrics")
            if chunk.traces.shape[1] != self.metadata.chunk_length:
                raise ValueError("Chunk length mismatch for traces")
            if chunk.logs.shape[1] != self.metadata.event_dim:
                raise ValueError("Event dimension mismatch")
            if chunk.metrics.shape[2] != self.metadata.metric_dim:
                raise ValueError("Metric dimension mismatch")
            if chunk.traces.shape[2] != self.metadata.trace_channels:
                raise ValueError("Trace channel mismatch")
