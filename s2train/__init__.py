"""s2train — a configuration-driven experiment framework for Sentinel-2
cloud-reconstruction models.

Everything (model, dataset, optimizer, scheduler, loss, metrics, callbacks) is
selected and parameterised from a single YAML file and instantiated through
registries, so new architectures plug in without touching the training code.

This package provides the *framework*, not the research model itself: real
baseline models (a UNet and a reference-conditioned UNet) are included to make
the framework runnable end-to-end; heavier architectures (TCR-Net, Restormer,
SwinIR, ...) register the same way.
"""

from __future__ import annotations

from .config import ExperimentConfig, load_config
from .registry import (
    CALLBACKS,
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)

__all__ = [
    "ExperimentConfig",
    "load_config",
    "MODELS",
    "LOSSES",
    "METRICS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "DATASETS",
    "CALLBACKS",
]

__version__ = "1.0.0"
