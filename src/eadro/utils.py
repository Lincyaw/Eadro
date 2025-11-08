from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch

__all__ = ["seed_everything", "configure_experiment", "record_scores"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_experiment(
    params: Mapping[str, Any], result_dir: Path
) -> Tuple[str, Path]:
    """Create a unique directory for the current run and configure logging."""

    sortable = sorted((str(k), str(v)) for k, v in params.items())
    hash_id = hashlib.md5(repr(sortable).encode("utf-8")).hexdigest()[:8]
    experiment_dir = result_dir / hash_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    params_path = experiment_dir / "params.json"
    with params_path.open("w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, ensure_ascii=False)

    log_file = experiment_dir / "running.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return hash_id, experiment_dir


def record_scores(
    result_dir: Path, hash_id: str, scores: Mapping[str, float], converge_epoch: int
) -> None:
    experiments_file = result_dir / "experiments.txt"
    timestamp = datetime.utcnow() + timedelta(hours=8)
    with experiments_file.open("a", encoding="utf-8") as fh:
        fh.write(f"{hash_id}: {timestamp:%Y/%m/%d-%H:%M:%S}\n")
        score_line = "\t".join(f"{key}:{value:.4f}" for key, value in scores.items())
        fh.write(f"* Test result -- {score_line}\n")
        fh.write(f"Best score got at epoch: {converge_epoch}\n")
        fh.write(f"{'=' * 40}\n")
