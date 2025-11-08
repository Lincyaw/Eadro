from __future__ import annotations

from pathlib import Path
import json
import logging
import pickle
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import numpy.typing as npt

from .schema import ChunkPayload, DatasetMetadata, DatasetSplit, EadroDataset, EdgeIndex

__all__ = ["load_preprocessed_dataset", "save_preprocessed_dataset", "load_chunk_file"]


def _as_edge_index(raw_edges: Iterable[Iterable[int]]) -> EdgeIndex:
    sources, destinations = raw_edges
    src = np.asarray(list(sources), dtype=np.int64)
    dst = np.asarray(list(destinations), dtype=np.int64)
    return src, dst


def _load_metadata(metadata_path: Path) -> Dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_chunk_file(
    chunk_path: Path,
) -> Mapping[str, Mapping[str, npt.NDArray[np.float32]]]:
    with chunk_path.open("rb") as fh:
        return pickle.load(fh)


def _build_chunks(
    raw_chunks: Mapping[str, Mapping[str, npt.NDArray[np.float32]]],
) -> List[ChunkPayload]:
    processed: List[ChunkPayload] = []
    for chunk_id, payload in raw_chunks.items():
        try:
            logs = np.asarray(payload["logs"], dtype=np.float32)
            metrics = np.asarray(payload["metrics"], dtype=np.float32)
            traces = np.asarray(payload["traces"], dtype=np.float32)
            culprit = int(payload["culprit"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Missing key {exc} for chunk '{chunk_id}'") from exc

        chunk = ChunkPayload(
            chunk_id=chunk_id,
            logs=logs,
            metrics=metrics,
            traces=traces,
            culprit=culprit,
        )
        chunk.validate()
        processed.append(chunk)
    return processed


def load_preprocessed_dataset(data_dir: Path) -> EadroDataset:
    """Load a dataset produced by the original preprocessing utilities."""

    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found under {data_dir}")

    metadata = _load_metadata(metadata_path)
    edges = _as_edge_index(metadata["edges"])

    train_chunks = load_chunk_file(data_dir / "chunk_train.pkl")
    test_chunks = load_chunk_file(data_dir / "chunk_test.pkl")

    processed_train = _build_chunks(train_chunks)
    processed_test = _build_chunks(test_chunks)

    if not processed_train and not processed_test:
        raise ValueError("The dataset does not contain any chunks")

    reference = processed_train[0] if processed_train else processed_test[0]
    trace_channels = reference.traces.shape[-1]
    metadata_obj = DatasetMetadata(
        node_count=int(metadata["node_num"]),
        event_dim=int(metadata["event_num"]),
        metric_dim=int(metadata["metric_num"]),
        trace_channels=int(trace_channels),
        chunk_length=int(metadata["chunk_lenth"]),
        edges=edges,
    )

    dataset = EadroDataset(
        metadata=metadata_obj,
        train=DatasetSplit(tuple(processed_train)),
        test=DatasetSplit(tuple(processed_test)),
    )
    dataset.validate()

    logging.info(
        "Loaded %d train / %d test chunks from %s",
        len(processed_train),
        len(processed_test),
        data_dir,
    )
    return dataset


def save_preprocessed_dataset(dataset: EadroDataset, output_dir: Path) -> None:
    """Save a preprocessed dataset to disk.

    Args:
        dataset: The dataset to save
        output_dir: Directory to save the dataset files

    Creates:
        - metadata.json: Dataset metadata
        - chunk_train.pkl: Training chunks
        - chunk_test.pkl: Testing chunks
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate before saving
    dataset.validate()

    # Convert chunks back to dictionary format for compatibility
    def chunks_to_dict(chunks: Iterable[ChunkPayload]) -> Dict[str, Dict[str, Any]]:
        result = {}
        for chunk in chunks:
            result[chunk.chunk_id] = {
                "logs": chunk.logs,
                "metrics": chunk.metrics,
                "traces": chunk.traces,
                "culprit": chunk.culprit,
            }
        return result

    # Save train and test chunks
    train_dict = chunks_to_dict(dataset.train.chunks)
    test_dict = chunks_to_dict(dataset.test.chunks)

    with (output_dir / "chunk_train.pkl").open("wb") as f:
        pickle.dump(train_dict, f)

    with (output_dir / "chunk_test.pkl").open("wb") as f:
        pickle.dump(test_dict, f)

    # Save metadata
    metadata_dict = {
        "node_num": dataset.metadata.node_count,
        "event_num": dataset.metadata.event_dim,
        "metric_num": dataset.metadata.metric_dim,
        "chunk_lenth": dataset.metadata.chunk_length,
        "edges": [
            dataset.metadata.edges[0].tolist(),
            dataset.metadata.edges[1].tolist(),
        ],
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata_dict, f, indent=2)

    logging.info(
        "Saved %d train / %d test chunks to %s",
        len(dataset.train),
        len(dataset.test),
        output_dir,
    )
