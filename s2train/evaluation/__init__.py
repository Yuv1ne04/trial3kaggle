"""Evaluation workflow: score a trained checkpoint over a dataset split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..builder import build_dataloader
from ..config import ExperimentConfig
from ..inference import Predictor
from ..logging_setup import get_logger
from ..metrics import compute_metrics

logger = get_logger()


class Evaluator:
    """Computes overall / cloud-region / clear-region metrics over a split."""

    def __init__(self, predictor: Predictor, config: ExperimentConfig) -> None:
        """Initialise the evaluator.

        Args:
            predictor: A loaded predictor.
            config: The experiment configuration.
        """
        self.predictor = predictor
        self.config = config

    @classmethod
    def from_checkpoint(cls, checkpoint: Path | str, *, device: str = "auto") -> "Evaluator":
        """Build an evaluator from a checkpoint.

        Args:
            checkpoint: Path to the ``.pt`` checkpoint.
            device: Device string.

        Returns:
            A ready :class:`Evaluator`.
        """
        predictor = Predictor.from_checkpoint(checkpoint, device=device)
        return cls(predictor, predictor.config)

    def evaluate_split(self, split: str | None = None,
                       *, root: str | None = None) -> dict[str, Any]:
        """Evaluate the model over a dataset split.

        Args:
            split: Split name (defaults to the configured test split).
            root: Optional dataset root override.

        Returns:
            A report dict with mean metrics and the sample count.
        """
        config = self.config
        if root is not None:
            config.data.root = root
        split = split or config.data.test_split
        loader: DataLoader = build_dataloader(config, split, shuffle=False, augment=False)

        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            pred = self.predictor.predict_batch(batch)
            metrics = compute_metrics(pred, batch["ground_truth"], batch["mask"],
                                      config.metrics)
            for key, value in metrics.items():
                if not np.isnan(value):
                    totals[key] = totals.get(key, 0.0) + value
            count += 1
        means = {k: round(v / max(1, count), 6) for k, v in totals.items()}
        report = {"split": split, "num_batches": count, "metrics": means,
                  "model": config.model.name}
        logger.info("Evaluation on '%s': %s", split, means)
        return report

    def write_report(self, report: dict[str, Any], path: Path | str) -> Path:
        """Write an evaluation report to JSON.

        Args:
            report: The report from :meth:`evaluate_split`.
            path: Destination path.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path
