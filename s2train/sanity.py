"""Sanity / overfit mode: prove the whole pipeline learns on a tiny subset.

Runs a short training on a small subset (validating on the *same* subset) and
checks that the training loss falls by at least the configured fraction — a fast,
decisive smoke test that data loading, the model, the loss, the optimizer and
the trainer all function before committing to a full run.
"""

from __future__ import annotations

import csv
from typing import Any

from .config import ExperimentConfig
from .logging_setup import get_logger


def _apply_sanity_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Mutate a config into sanity/overfit mode (in place).

    Args:
        config: The experiment configuration.

    Returns:
        The same configuration, adjusted for a tiny overfit run.
    """
    sanity = config.sanity
    config.name = f"{config.name}_sanity"
    config.data.max_samples = sanity.num_samples
    config.data.augment = False
    config.data.num_workers = 0
    config.data.val_split = config.data.train_split  # validate on the same subset
    config.data.batch_size = min(config.data.batch_size, max(2, sanity.num_samples // 4))
    config.data.drop_last = False
    config.trainer.epochs = sanity.epochs
    config.trainer.val_interval = 1
    config.early_stopping.enabled = False
    config.visualization.every_n_epochs = 1
    config.resume = None
    return config


def _loss_curve(csv_path) -> list[float]:
    """Read the ``train/loss`` column from a metrics CSV.

    Args:
        csv_path: Path to ``metrics.csv``.

    Returns:
        The per-epoch training-loss values.
    """
    values: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("train/loss", "")
            if raw not in ("", None):
                values.append(float(raw))
    return values


def run_sanity(config: ExperimentConfig) -> dict[str, Any]:
    """Run sanity mode and verify the training loss decreases.

    Args:
        config: The experiment configuration (sanity overrides are applied).

    Returns:
        A result dict with ``passed`` and the loss curve summary.
    """
    from .trainers import Trainer

    logger = get_logger()
    config = _apply_sanity_overrides(config)
    trainer = Trainer(config)
    trainer.logger.info("SANITY MODE: overfitting %d sample(s) for %d epoch(s)",
                        config.data.max_samples, config.trainer.epochs)
    trainer.fit()

    curve = _loss_curve(trainer.output_dir / "metrics.csv")
    result: dict[str, Any] = {"loss_curve": curve, "passed": False}
    if len(curve) >= 2 and curve[0] > 0:
        drop = (curve[0] - curve[-1]) / curve[0]
        result.update({
            "first_loss": round(curve[0], 6), "last_loss": round(curve[-1], 6),
            "loss_drop_fraction": round(drop, 4),
            "required_drop_fraction": config.sanity.require_loss_drop,
            "passed": drop >= config.sanity.require_loss_drop and curve[-1] < curve[0],
        })
    status = "PASSED" if result["passed"] else "FAILED"
    trainer.logger.info("SANITY %s: loss %s -> %s (drop %.1f%%)", status,
                        result.get("first_loss"), result.get("last_loss"),
                        100 * result.get("loss_drop_fraction", 0.0))
    logger.info("Sanity check %s", status)
    return result
