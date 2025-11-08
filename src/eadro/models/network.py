"""Neural network components for multi-source microservice data encoding.

This module contains the model architecture components:
- GraphModel: Graph neural network for service topology
- ConvNet: Temporal convolutional network for time series
- TraceModel, MetricModel, LogModel: Modality-specific encoders
- MultiSourceEncoder: Fuses all data sources
- MainModel: Complete end-to-end model for detection and localization
"""

from __future__ import annotations

import math
from typing import List, Optional

import dgl
import torch
from dgl.nn import GlobalAttentionPooling
from dgl.nn.pytorch import GATv2Conv
from torch import Tensor, nn

__all__ = [
    "GraphModel",
    "ConvNet",
    "TraceModel",
    "MetricModel",
    "LogModel",
    "MultiSourceEncoder",
    "MainModel",
]


class Chomp1d(nn.Module):
    """Remove extra padding from causal convolution output."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class ConvNet(nn.Module):
    """Temporal Convolutional Network (TCN) for sequence modeling.

    Args:
        num_inputs: Input feature dimension
        num_channels: List of hidden channel sizes for each layer
        kernel_sizes: List of kernel sizes for each layer
        dilation: Dilation growth factor (default: 2)
        device: Computation device
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_sizes: List[int],
        dilation: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        for i in range(len(kernel_sizes)):
            dilation_size = dilation**i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    Chomp1d(padding),
                ]
            )

        self.network = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]
        self.network.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, out_dim)
        """
        x = x.permute(0, 2, 1).float()  # (B, in_dim, T)
        out = self.network(x)  # (B, out_dim, T)
        return out.permute(0, 2, 1)  # (B, T, out_dim)


class SelfAttention(nn.Module):
    """Self-attention mechanism for temporal aggregation.

    Args:
        input_size: Hidden feature dimension
        seq_len: Sequence length (time window size)
    """

    def __init__(self, input_size: int, seq_len: int) -> None:
        super().__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self._glorot_init(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Aggregated tensor of shape (batch_size, input_size)
        """
        input_tensor = x.transpose(1, 0)  # (seq_len, batch, input_size)
        input_tensor = torch.bmm(input_tensor, self.atten_w) + self.atten_bias
        input_tensor = input_tensor.transpose(1, 0)  # (batch, seq_len, 1)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    @staticmethod
    def _glorot_init(tensor: Tensor) -> None:
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)


class GraphModel(nn.Module):
    """Graph Attention Network for service topology encoding.

    Args:
        in_dim: Node feature dimension
        graph_hiddens: List of hidden dimensions for GAT layers
        device: Computation device
        attn_head: Number of attention heads
        activation: Negative slope for LeakyReLU
        **kwargs: Additional arguments (e.g., attn_drop)
    """

    def __init__(
        self,
        in_dim: int,
        graph_hiddens: List[int] = [64, 128],
        device: str = "cpu",
        attn_head: int = 4,
        activation: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        layers: List[GATv2Conv] = []
        dropout = kwargs.get("attn_drop", 0.0)

        for i, hidden in enumerate(graph_hiddens):
            in_feats = graph_hiddens[i - 1] if i > 0 else in_dim
            layers.append(
                GATv2Conv(
                    in_feats,
                    out_feats=hidden,
                    num_heads=attn_head,
                    attn_drop=dropout,
                    negative_slope=activation,
                    allow_zero_in_degree=True,
                )
            )

        self.net = nn.ModuleList(layers).to(device)
        self.maxpool = nn.MaxPool1d(attn_head)
        self.out_dim = graph_hiddens[-1]
        self.pooling = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        """
        Args:
            graph: DGL batched graph
            x: Node features of shape (batch_size * node_num, in_dim)

        Returns:
            Graph-level embeddings of shape (batch_size, out_dim)
        """
        out = x
        for layer in self.net:
            out = layer(graph, out)
            out = self.maxpool(out.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        return self.pooling(graph, out)


class TraceModel(nn.Module):
    """Encoder for distributed tracing data.

    Args:
        device: Computation device
        trace_hiddens: List of hidden dimensions for TCN
        trace_kernel_sizes: List of kernel sizes for TCN
        self_attn: Whether to use self-attention pooling
        chunk_lenth: Time window length (required if self_attn=True)
    """

    def __init__(
        self,
        device: str = "cpu",
        trace_hiddens: List[int] = [20, 50],
        trace_kernel_sizes: List[int] = [3, 3],
        self_attn: bool = False,
        chunk_lenth: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        assert len(trace_hiddens) == len(trace_kernel_sizes)
        self.out_dim = trace_hiddens[-1]
        self.self_attn = self_attn

        self.net = ConvNet(
            2,
            num_channels=trace_hiddens,
            kernel_sizes=trace_kernel_sizes,
            device=device,
        )

        if self_attn:
            if chunk_lenth is None:
                raise ValueError("chunk_lenth required when self_attn=True")
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Trace features of shape (batch_size, time_steps, 2)

        Returns:
            Encoded features of shape (batch_size, out_dim)
        """
        hidden_states = self.net(x)
        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]  # Take last timestep


class MetricModel(nn.Module):
    """Encoder for system metrics (CPU, memory, etc.).

    Args:
        metric_num: Number of metric channels
        device: Computation device
        metric_hiddens: List of hidden dimensions for TCN
        metric_kernel_sizes: List of kernel sizes for TCN
        self_attn: Whether to use self-attention pooling
        chunk_lenth: Time window length (required if self_attn=True)
    """

    def __init__(
        self,
        metric_num: int,
        device: str = "cpu",
        metric_hiddens: List[int] = [64, 128],
        metric_kernel_sizes: List[int] = [3, 3],
        self_attn: bool = False,
        chunk_lenth: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        assert len(metric_hiddens) == len(metric_kernel_sizes)
        self.metric_num = metric_num
        self.out_dim = metric_hiddens[-1]
        self.self_attn = self_attn

        self.net = ConvNet(
            num_inputs=metric_num,
            num_channels=metric_hiddens,
            kernel_sizes=metric_kernel_sizes,
            device=device,
        )

        if self_attn:
            if chunk_lenth is None:
                raise ValueError("chunk_lenth required when self_attn=True")
            self.attn_layer = SelfAttention(self.out_dim, chunk_lenth)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Metric features of shape (batch_size, time_steps, metric_num)

        Returns:
            Encoded features of shape (batch_size, out_dim)
        """
        assert x.shape[-1] == self.metric_num
        hidden_states = self.net(x)
        if self.self_attn:
            return self.attn_layer(hidden_states)
        return hidden_states[:, -1, :]


class LogModel(nn.Module):
    """Encoder for log event features.

    Args:
        event_num: Log vocabulary size (number of unique event types)
        out_dim: Output embedding dimension
    """

    def __init__(self, event_num: int, out_dim: int) -> None:
        super().__init__()
        self.embedder = nn.Linear(event_num, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Log event features of shape (batch_size, event_num)

        Returns:
            Embedded features of shape (batch_size, out_dim)
        """
        return self.embedder(x)


class MultiSourceEncoder(nn.Module):
    """Multi-source data fusion encoder.

    Combines logs, metrics, traces, and service topology into a unified
    graph-level representation.

    Args:
        event_num: Log vocabulary size
        metric_num: Number of metric channels
        node_num: Number of nodes in the service graph
        device: Computation device
        log_dim: Log embedding dimension
        fuse_dim: Fusion layer output dimension
        alpha: Loss weighting factor (not used in encoder)
        **kwargs: Additional arguments passed to sub-models
    """

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        log_dim: int = 64,
        fuse_dim: int = 64,
        alpha: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()

        self.node_num = node_num
        self.alpha = alpha

        self.trace_model = TraceModel(device=device, **kwargs)
        trace_dim = self.trace_model.out_dim

        self.log_model = LogModel(event_num, log_dim)

        self.metric_model = MetricModel(metric_num, device=device, **kwargs)
        metric_dim = self.metric_model.out_dim

        fuse_in = trace_dim + log_dim + metric_dim

        if fuse_dim % 2 != 0:
            fuse_dim += 1
        self.fuse = nn.Linear(fuse_in, fuse_dim)
        self.activate = nn.GLU()
        self.feat_in_dim = fuse_dim // 2

        self.status_model = GraphModel(in_dim=self.feat_in_dim, device=device, **kwargs)
        self.feat_out_dim = self.status_model.out_dim

    def forward(self, graph: dgl.DGLGraph) -> Tensor:
        """
        Args:
            graph: Batched DGL graph with node features:
                   - "logs": (batch*node_num, event_dim)
                   - "metrics": (batch*node_num, time_steps, metric_dim)
                   - "traces": (batch*node_num, time_steps, trace_channels)

        Returns:
            Graph-level embeddings of shape (batch_size, feat_out_dim)
        """
        trace_embedding = self.trace_model(graph.ndata["traces"])
        log_embedding = self.log_model(graph.ndata["logs"])
        metric_embedding = self.metric_model(graph.ndata["metrics"])

        feature = self.activate(
            self.fuse(
                torch.cat((trace_embedding, log_embedding, metric_embedding), dim=-1)
            )
        )
        embeddings = self.status_model(graph, feature)
        return embeddings


class FullyConnected(nn.Module):
    """Multi-layer perceptron with ReLU activations.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        linear_sizes: List of hidden layer sizes
    """

    def __init__(self, in_dim: int, out_dim: int, linear_sizes: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers.extend([nn.Linear(input_size, hidden), nn.ReLU()])

        layers.append(nn.Linear(linear_sizes[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MainModel(nn.Module):
    """Complete Eadro model for anomaly detection and root cause localization.

    Args:
        event_num: Log vocabulary size
        metric_num: Number of metric channels
        node_num: Number of nodes in the service graph
        device: Computation device
        alpha: Loss weight balancing detection and localization (default: 0.5)
        debug: Enable debug mode (unused)
        **kwargs: Configuration dict with keys:
                  - detect_hiddens: Hidden sizes for detection head
                  - locate_hiddens: Hidden sizes for localization head
                  - Additional encoder arguments
    """

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: str,
        alpha: float = 0.5,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = device
        self.node_num = node_num
        self.alpha = alpha

        self.encoder = MultiSourceEncoder(
            event_num, metric_num, node_num, device, alpha=alpha, **kwargs
        )

        self.detector = FullyConnected(
            self.encoder.feat_out_dim, 2, kwargs["detect_hiddens"]
        ).to(device)
        self.decoder_criterion = nn.CrossEntropyLoss()

        self.locator = FullyConnected(
            self.encoder.feat_out_dim, node_num, kwargs["locate_hiddens"]
        ).to(device)
        self.locator_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.get_prob = nn.Softmax(dim=-1)

    def forward(
        self, graph: dgl.DGLGraph, fault_indexs: Tensor
    ) -> dict[str, Tensor | list]:
        """
        Args:
            graph: Batched DGL graph with multi-source features
            fault_indexs: Ground truth fault node indices, shape (batch_size,)
                          Use -1 for healthy windows

        Returns:
            Dictionary containing:
                - loss: Combined detection + localization loss
                - y_pred: List of predicted node rankings per sample
                - y_prob: Ground truth one-hot labels (batch_size, node_num)
                - pred_prob: Predicted probabilities (batch_size, node_num)
        """
        batch_size = graph.batch_size
        embeddings = self.encoder(graph)

        # Prepare ground truth labels
        y_prob = torch.zeros((batch_size, self.node_num)).to(self.device)
        for i in range(batch_size):
            if fault_indexs[i] > -1:
                y_prob[i, fault_indexs[i]] = 1

        y_anomaly = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        for i in range(batch_size):
            y_anomaly[i] = int(fault_indexs[i] > -1)

        # Localization head
        locate_logits = self.locator(embeddings)
        locate_loss = self.locator_criterion(
            locate_logits, fault_indexs.to(self.device)
        )

        # Detection head
        detect_logits = self.detector(embeddings)
        detect_loss = self.decoder_criterion(detect_logits, y_anomaly)

        # Combined loss
        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss

        # Inference
        node_probs = self.get_prob(locate_logits.detach()).cpu().numpy()
        y_pred = self._inference(batch_size, node_probs, detect_logits)

        return {
            "loss": loss,
            "y_pred": y_pred,
            "y_prob": y_prob.detach().cpu().numpy(),
            "pred_prob": node_probs,
        }

    def _inference(
        self, batch_size: int, node_probs, detect_logits: Tensor
    ) -> List[List[int]]:
        """Generate ranked node predictions.

        Args:
            batch_size: Number of samples
            node_probs: Localization probabilities (batch_size, node_num)
            detect_logits: Detection logits (batch_size, 2)

        Returns:
            List of ranked node indices per sample. [-1] if predicted healthy.
        """
        node_list = node_probs.argsort(axis=1)[:, ::-1]  # Descending order
        detect_pred = detect_logits.detach().cpu().numpy().argmax(axis=1)

        y_pred: List[List[int]] = []
        for i in range(batch_size):
            if detect_pred[i] < 1:
                y_pred.append([-1])  # Predicted healthy
            else:
                y_pred.append(node_list[i].tolist())

        return y_pred
