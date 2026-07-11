"""Learning-rate scheduler factories (registered for config selection).

Schedulers here step **per epoch**. ``plateau`` additionally consumes a
validation metric (the trainer passes it).
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau

from ..registry import SCHEDULERS


@SCHEDULERS.register("cosine")
def build_cosine(optimizer: Optimizer, epochs: int = 100, warmup_epochs: int = 5,
                 min_lr_ratio: float = 0.01) -> LRScheduler:
    """Build a warmup + cosine-decay scheduler (per-epoch).

    Args:
        optimizer: The optimizer.
        epochs: Total epochs.
        warmup_epochs: Linear warmup epochs.
        min_lr_ratio: Final LR as a fraction of the peak LR.

    Returns:
        A :class:`LambdaLR` implementing warmup + cosine decay.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@SCHEDULERS.register("step")
def build_step(optimizer: Optimizer, step_size: int = 30,
               gamma: float = 0.5) -> LRScheduler:
    """Build a step-decay scheduler.

    Args:
        optimizer: The optimizer.
        step_size: Epochs between decays.
        gamma: Multiplicative decay factor.

    Returns:
        A :class:`torch.optim.lr_scheduler.StepLR`.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


@SCHEDULERS.register("plateau")
def build_plateau(optimizer: Optimizer, mode: str = "max", factor: float = 0.5,
                  patience: int = 5, min_lr: float = 1e-6) -> ReduceLROnPlateau:
    """Build a ReduceLROnPlateau scheduler (steps on a validation metric).

    Args:
        optimizer: The optimizer.
        mode: ``"min"`` or ``"max"`` for the monitored metric.
        factor: LR reduction factor.
        patience: Epochs without improvement before reducing.
        min_lr: Lower LR bound.

    Returns:
        A :class:`ReduceLROnPlateau`.
    """
    return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                             min_lr=min_lr)


@SCHEDULERS.register("none")
def build_none(optimizer: Optimizer) -> LRScheduler:
    """Build a no-op (constant-LR) scheduler.

    Args:
        optimizer: The optimizer.

    Returns:
        A :class:`LambdaLR` that never changes the LR.
    """
    return LambdaLR(optimizer, lambda _epoch: 1.0)
