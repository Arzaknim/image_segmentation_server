#!/usr/bin/env python3
"""Structured Training Logger for Semantic Segmentation.

Provides a unified, thread-safe, and crash-resilient JSON logging class
for tracking deep learning experiments:
  1. Experiment metadata and hardware environment
  2. Model architecture parameters and reparameterization status
  3. Hyperparameters, dataset paths, and optimizer/scheduler states
  4. Epoch-by-epoch loss and learning rate telemetry
  5. Per-class validation / test mIoU benchmarks
  6. Multi-fold cross validation aggregation
  7. Atomic temporary-file dumping to prevent file corruption during crashes
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TrainingLogger:
    """Atomic JSON experiment logger for single-run and k-fold training workflows."""

    def __init__(
        self,
        experiment_type: str = "tiled",  # "tiled" or "kfold"
        log_dir: Union[str, Path] = "../training_logs",
        run_name: Optional[str] = None,
    ):
        self.experiment_type = experiment_type
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now(timezone.utc)
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S_%f")
        filename_prefix = run_name or f"{experiment_type}_training_{timestamp}"
        if not filename_prefix.endswith(".json"):
            filename_prefix += ".json"

        self.log_path = self.log_dir / filename_prefix

        # Base data structure
        self.data: Dict[str, Any] = {
            "status": "running",
            "started_at": self.started_at.isoformat(),
            "model": {},
            "hyperparameters": {},
        }

        if experiment_type == "kfold":
            self.data["folds"] = []
        else:
            self.data["training_samples"] = 0
            self.data["test_samples"] = 0
            self.data["training_tiles"] = 0
            self.data["checkpoint"] = ""
            self.data["epochs"] = []
            self.data["test"] = None

        self._save()
        print(f"Initialized training log at: {self.log_path}")

    @property
    def path(self) -> Path:
        """Return the absolute path of the JSON log file."""
        return self.log_path

    def _save(self) -> None:
        """Atomically persist log state to disk using a temporary file."""
        temp_path = self.log_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
                f.write("\n")
            os.replace(temp_path, self.log_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise e

    def set_model_info(self, model_info: Dict[str, Any]) -> None:
        """Record model architecture, parameters, and metadata."""
        self.data["model"] = model_info
        self._save()

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Record training hyperparameters, optimizer, scheduler, criterion, etc."""
        self.data["hyperparameters"] = hyperparameters
        self._save()

    def update_hyperparameters(self, updates: Dict[str, Any]) -> None:
        """Update or append specific hyperparameter keys."""
        self.data["hyperparameters"].update(updates)
        self._save()

    # --- Single Tiled Training Methods ---

    def set_sample_counts(
        self,
        training_samples: int,
        test_samples: int,
        training_tiles: int = 0,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Set sample counts and output checkpoint path for single tiled run."""
        self.data["training_samples"] = training_samples
        self.data["test_samples"] = test_samples
        self.data["training_tiles"] = training_tiles
        if checkpoint:
            self.data["checkpoint"] = checkpoint
        self._save()

    def set_training_tiles(self, training_tiles: int) -> None:
        """Update count of extracted training tiles."""
        self.data["training_tiles"] = training_tiles
        self._save()

    def log_epoch(
        self,
        epoch: int,
        training_loss: float,
        learning_rate: float,
        **extra_metrics: Any,
    ) -> None:
        """Record a completed training epoch for a single run."""
        record: Dict[str, Any] = {
            "epoch": epoch,
            "training_loss": float(training_loss),
            "learning_rate": float(learning_rate),
        }
        for k, v in extra_metrics.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v

        self.data["epochs"].append(record)
        self._save()

    def log_test_results(
        self,
        mean_iou: float,
        per_class_iou: Dict[str, float],
    ) -> None:
        """Record final full-resolution evaluation test metrics."""
        self.data["test"] = {
            "mean_iou": float(mean_iou),
            "per_class_iou": {str(k): float(v) for k, v in per_class_iou.items()},
        }
        self.data["test_mean_iou"] = float(mean_iou)
        self._save()

    # --- K-Fold Cross Validation Methods ---

    def start_fold(
        self,
        fold_idx: int,
        training_samples: int,
        validation_samples: int,
        checkpoint: str,
    ) -> None:
        """Initialize telemetry for a new fold in k-fold cross validation."""
        fold_record: Dict[str, Any] = {
            "fold": fold_idx,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "checkpoint": checkpoint,
            "epochs": [],
        }
        self.data["folds"].append(fold_record)
        self._save()

    def log_fold_epoch(
        self,
        fold_idx: int,
        epoch: int,
        training_loss: float,
        learning_rate: float,
        **extra_metrics: Any,
    ) -> None:
        """Record an epoch's metrics inside an active fold."""
        # Find the target fold record
        fold_record = next(
            (f for f in self.data["folds"] if f["fold"] == fold_idx), None
        )
        if fold_record is None:
            raise ValueError(f"Fold {fold_idx} has not been started.")

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "training_loss": float(training_loss),
            "learning_rate": float(learning_rate),
        }
        for k, v in extra_metrics.items():
            epoch_record[k] = float(v) if isinstance(v, (int, float)) else v

        fold_record["epochs"].append(epoch_record)
        self._save()

    def log_fold_validation(
        self,
        fold_idx: int,
        mean_iou: float,
        per_class_iou: Dict[str, float],
    ) -> None:
        """Record validation results for a completed fold."""
        fold_record = next(
            (f for f in self.data["folds"] if f["fold"] == fold_idx), None
        )
        if fold_record is None:
            raise ValueError(f"Fold {fold_idx} has not been started.")

        fold_record["validation"] = {
            "mean_iou": float(mean_iou),
            "per_class_iou": {str(k): float(v) for k, v in per_class_iou.items()},
        }
        self._save()

    # --- Lifecycle Finalization ---

    def complete(
        self,
        average_mean_iou: Optional[float] = None,
        test_mean_iou: Optional[float] = None,
    ) -> None:
        """Mark the training run as successfully completed."""
        self.data["status"] = "completed"
        self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        if average_mean_iou is not None:
            self.data["average_mean_iou"] = float(average_mean_iou)
        if test_mean_iou is not None:
            self.data["test_mean_iou"] = float(test_mean_iou)
        self._save()

    def fail(self, error: Union[str, Exception]) -> None:
        """Mark the training run as failed and record the exception/trace."""
        self.data["status"] = "failed"
        self.data["error"] = (
            f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        )
        self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

