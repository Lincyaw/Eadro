from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    random_seed: int = 42
    gpu: bool = True
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 10
    self_attention: bool = True
    fuse_dim: int = 128
    alpha: float = 0.5
    locate_hiddens: Tuple[int, ...] = field(default_factory=lambda: (64,))
    detect_hiddens: Tuple[int, ...] = field(default_factory=lambda: (64,))
    log_dim: int = 16
    trace_kernel_sizes: Tuple[int, ...] = field(default_factory=lambda: (2,))
    trace_hiddens: Tuple[int, ...] = field(default_factory=lambda: (64,))
    metric_kernel_sizes: Tuple[int, ...] = field(default_factory=lambda: (2,))
    metric_hiddens: Tuple[int, ...] = field(default_factory=lambda: (64,))
    graph_hiddens: Tuple[int, ...] = field(default_factory=lambda: (64,))
    attn_head: int = 4
    activation: float = 0.2
    attn_drop: float = 0.0
    data_root: Path = Path("./chunks")
    dataset_name: str = ""
    result_dir: Path = Path("result")
    evaluation_epoch: int = 10

    def data_dir(self) -> Path:
        if not self.dataset_name:
            raise ValueError("dataset_name must be provided")
        return (self.data_root / self.dataset_name).resolve()

    def to_logging_dict(self) -> dict:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["result_dir"] = str(self.result_dir)
        return payload

    @staticmethod
    def parse_sequence(raw: Iterable[int]) -> Tuple[int, ...]:
        return tuple(int(v) for v in raw)
