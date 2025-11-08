from __future__ import annotations

from typing import List, Sequence, Tuple

import dgl
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .schema import ChunkPayload, DatasetMetadata

__all__ = ["ChunkGraphDataset", "collate_graph_batches"]


class ChunkGraphDataset(Dataset):
    """Torch-compatible dataset that yields DGL graphs and culprit labels."""

    def __init__(
        self, chunks: Sequence[ChunkPayload], metadata: DatasetMetadata
    ) -> None:
        self._chunks: List[ChunkPayload] = list(chunks)
        self._metadata = metadata
        self._edge_index = (
            torch.from_numpy(metadata.edges[0].copy()),
            torch.from_numpy(metadata.edges[1].copy()),
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> Tuple[dgl.DGLGraph, Tensor]:
        chunk = self._chunks[index]
        graph = dgl.graph(self._edge_index, num_nodes=self._metadata.node_count)

        graph.ndata["logs"] = torch.from_numpy(chunk.logs.copy())
        graph.ndata["metrics"] = torch.from_numpy(chunk.metrics.copy())
        graph.ndata["traces"] = torch.from_numpy(chunk.traces.copy())

        label = torch.tensor(chunk.culprit, dtype=torch.long)
        return graph, label

    def chunk_id(self, index: int) -> str:
        return self._chunks[index].chunk_id


def collate_graph_batches(
    batch: Sequence[Tuple[dgl.DGLGraph, Tensor]],
) -> Tuple[dgl.DGLGraph, Tensor]:
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(labels)
