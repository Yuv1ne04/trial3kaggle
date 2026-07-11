"""Optimizer factories (registered for config selection)."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..registry import OPTIMIZERS


def _params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    """Return the trainable parameters of a model."""
    return (p for p in model.parameters() if p.requires_grad)


@OPTIMIZERS.register("adamw")
def build_adamw(model: nn.Module, lr: float = 2e-4, weight_decay: float = 1e-4,
                betas: tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
    """Build an AdamW optimizer.

    Args:
        model: The model whose parameters are optimised.
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.

    Returns:
        A configured :class:`torch.optim.AdamW`.
    """
    return torch.optim.AdamW(_params(model), lr=lr, weight_decay=weight_decay, betas=tuple(betas))


@OPTIMIZERS.register("adam")
def build_adam(model: nn.Module, lr: float = 2e-4,
               weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Build an Adam optimizer.

    Args:
        model: The model.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        A configured :class:`torch.optim.Adam`.
    """
    return torch.optim.Adam(_params(model), lr=lr, weight_decay=weight_decay)


@OPTIMIZERS.register("sgd")
def build_sgd(model: nn.Module, lr: float = 1e-2, momentum: float = 0.9,
              weight_decay: float = 1e-4, nesterov: bool = True) -> torch.optim.Optimizer:
    """Build an SGD optimizer.

    Args:
        model: The model.
        lr: Learning rate.
        momentum: Momentum.
        weight_decay: Weight decay.
        nesterov: Use Nesterov momentum.

    Returns:
        A configured :class:`torch.optim.SGD`.
    """
    return torch.optim.SGD(_params(model), lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=nesterov)
