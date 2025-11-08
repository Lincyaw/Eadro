"""Training and evaluation logic for Eadro models."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader

from eadro.models import MainModel

__all__ = ["Trainer"]


class Trainer:
    """Training and evaluation manager for Eadro models.

    Args:
        event_num: Log vocabulary size
        metric_num: Number of metric channels
        node_num: Number of nodes in the service graph
        device: Computation device (cpu or cuda)
        learning_rate: Optimizer learning rate
        epochs: Number of training epochs
        patience: Early stopping patience (0 to disable)
        result_dir: Directory to save checkpoints
        hash_id: Unique experiment identifier
        **kwargs: Additional model configuration
    """

    def __init__(
        self,
        event_num: int,
        metric_num: int,
        node_num: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        result_dir: Path = Path("./result"),
        hash_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.device = device

        self.model_save_dir = result_dir / (hash_id or "default")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.model = MainModel(event_num, metric_num, node_num, str(device), **kwargs)
        self.model.to(device)

    def evaluate(
        self, test_loader: DataLoader, datatype: str = "Test"
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            test_loader: DataLoader for evaluation data
            datatype: Label for logging (e.g., "Test", "Val")

        Returns:
            Dictionary of evaluation metrics (F1, Recall, Precision, HR@k, NDCG@k)
        """
        self.model.eval()
        hrs = np.zeros(5)
        ndcgs = np.zeros(5)
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for graph, ground_truths in test_loader:
                res = self.model.forward(graph.to(self.device), ground_truths)

                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()

                    if culprit == -1:  # Healthy sample
                        if faulty_nodes[0] == -1:
                            tp += 1
                        else:
                            fp += 1
                    else:  # Faulty sample
                        if faulty_nodes[0] == -1:
                            fn += 1
                        else:
                            tp += 1
                            rank = list(faulty_nodes).index(culprit)
                            for j in range(5):
                                hrs[j] += int(rank <= j)
                                ndcgs[j] += ndcg_score(
                                    [res["y_prob"][idx]],
                                    [res["pred_prob"][idx]],
                                    k=j + 1,
                                )

        pos = tp + fn
        eval_results = {
            "F1": tp * 2.0 / (tp + fp + pos) if (tp + fp + pos) > 0 else 0.0,
            "Rec": tp * 1.0 / pos if pos > 0 else 0.0,
            "Pre": tp * 1.0 / (tp + fp) if (tp + fp) > 0 else 0.0,
        }

        for j in [1, 3, 5]:
            eval_results[f"HR@{j}"] = hrs[j - 1] * 1.0 / pos
            eval_results[f"ndcg@{j}"] = ndcgs[j - 1] * 1.0 / pos

        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()])
        logging.info(f"{datatype} -- {metrics_str}")

        return eval_results

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        evaluation_epoch: int = 10,
    ) -> Tuple[Dict[str, float], int]:
        """Train the model with optional periodic evaluation.

        Args:
            train_loader: DataLoader for training data
            test_loader: Optional DataLoader for evaluation
            evaluation_epoch: Evaluate every N epochs

        Returns:
            Tuple of (best_eval_results, converge_epoch)
        """
        best_hr1 = -1.0
        best_state: Optional[Dict[str, Any]] = None
        best_eval_results: Dict[str, float] = {}
        converge_epoch = 0

        pre_loss = float("inf")
        worse_count = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            batch_cnt = 0

            epoch_time_start = time.time()
            for graph, label in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(graph.to(self.device), label)["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt

            logging.info(
                f"Epoch {epoch}/{self.epochs}, training loss: {epoch_loss:.5f} "
                f"[{epoch_time_elapsed:.2f}s]"
            )

            # Early stopping
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info(f"Early stop at epoch: {epoch}")
                    break
            else:
                worse_count = 0
            pre_loss = epoch_loss

            # Periodic evaluation
            if test_loader is not None and epoch % evaluation_epoch == 0:
                eval_results = self.evaluate(test_loader, datatype="Test")
                current_hr1 = eval_results.get("HR@1", 0.0)

                if current_hr1 > best_hr1:
                    best_hr1 = current_hr1
                    converge_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_eval_results = eval_results

        # Final evaluation with best model
        if test_loader is not None:
            if best_state is not None:
                self.model.load_state_dict(best_state)
            best_eval_results = self.evaluate(test_loader, datatype="Test")
            logging.info(
                f"* Best result got at epoch {converge_epoch} with HR@1: {best_hr1:.4f}"
            )

        # Save checkpoint
        if best_state is not None:
            checkpoint_path = self.model_save_dir / "model.ckpt"
            torch.save(best_state, checkpoint_path)
            logging.info(f"Saved best model to {checkpoint_path}")

        return best_eval_results, converge_epoch
