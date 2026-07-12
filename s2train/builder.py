"""Dependency-injection builders: construct all components from an ExperimentConfig.

This is the single place where configuration strings become live objects via the
registries. The trainer/evaluator/predictor depend only on the built objects,
never on the registries directly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

# Importing these packages registers their components.
from . import callbacks as _callbacks  # noqa: F401
from . import datasets as _datasets  # noqa: F401
from . import losses as _losses  # noqa: F401
from . import metrics as _metrics  # noqa: F401
from . import models as _models  # noqa: F401
from . import optimizers as _optimizers  # noqa: F401
from . import schedulers as _schedulers  # noqa: F401
from .config import ExperimentConfig
from .datasets import collate_batch
from .registry import (
    CALLBACKS,
    DATASETS,
    LOSSES,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)


def resolve_device(name: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        name: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved device (``auto`` prefers CUDA when available).
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(config: ExperimentConfig) -> nn.Module:
    """Build the model from the configuration.

    Args:
        config: The experiment configuration.

    Returns:
        The constructed model.
    """
    return MODELS.build(config.model.name, **config.model.params)


def build_loss(config: ExperimentConfig) -> nn.Module:
    """Build the loss from the configuration.

    Args:
        config: The experiment configuration.

    Returns:
        The constructed loss module.
    """
    return LOSSES.build(config.loss.name, **config.loss.params)


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Build the optimizer from the configuration.

    Args:
        config: The experiment configuration.
        model: The model to optimise.

    Returns:
        The constructed optimizer.
    """
    return OPTIMIZERS.build(config.optimizer.name, model=model, **config.optimizer.params)


def build_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
    """Build the LR scheduler from the configuration.

    Args:
        config: The experiment configuration.
        optimizer: The optimizer.

    Returns:
        The constructed scheduler.
    """
    params = dict(config.scheduler.params)
    if config.scheduler.name.lower() == "cosine" and "epochs" not in params:
        params["epochs"] = config.trainer.epochs
    return SCHEDULERS.build(config.scheduler.name, optimizer=optimizer, **params)


def build_dataloader(config: ExperimentConfig, split: str, *, shuffle: bool,
                     augment: bool) -> DataLoader:
    """Build a dataloader for a split from the configuration.

    Applies Kaggle-friendly defaults: worker count is clamped to the available
    CPUs; ``pin_memory`` is only enabled with CUDA; ``persistent_workers`` and
    ``prefetch_factor`` are used only when workers are present.

    Args:
        config: The experiment configuration.
        split: Dataset split name.
        shuffle: Whether to shuffle.
        augment: Whether to apply augmentation.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    import os

    data = config.data
    dataset = DATASETS.build(
        data.name, root=data.root, split=split, max_references=data.max_references,
        reflectance_scale=data.reflectance_scale, augment=augment,
        difficulty=data.difficulty, seed=config.seed, **data.params)

    # Overfit / sanity mode: restrict to the first ``max_samples`` samples.
    if data.max_samples and len(dataset) > data.max_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(data.max_samples)))

    workers = max(0, min(data.num_workers, os.cpu_count() or 1))
    kwargs: dict = {
        "batch_size": data.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": data.pin_memory and torch.cuda.is_available(),
        "collate_fn": collate_batch,
        "drop_last": data.drop_last and shuffle,
    }
    if workers > 0:
        kwargs["persistent_workers"] = data.persistent_workers
        kwargs["prefetch_factor"] = data.prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_callbacks(config: ExperimentConfig) -> list:
    """Build the callback list (implicit checkpoint/early-stop/vis/csv + extras).

    Args:
        config: The experiment configuration.

    Returns:
        The list of constructed callbacks.
    """
    built = [
        CALLBACKS.build("csv_logger"),
        CALLBACKS.build("checkpoint", monitor=config.checkpoint.monitor,
                        mode=config.checkpoint.mode, save_best=config.checkpoint.save_best,
                        save_latest=config.checkpoint.save_latest,
                        every_n_epochs=config.checkpoint.every_n_epochs),
    ]
    if config.early_stopping.enabled:
        built.append(CALLBACKS.build(
            "early_stopping", monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode, patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta))
    if config.visualization.enabled:
        built.append(CALLBACKS.build(
            "visualizer", every_n_epochs=config.visualization.every_n_epochs,
            num_samples=config.visualization.num_samples,
            rgb_bands=config.visualization.rgb_bands))
    for spec in config.callbacks:
        built.append(CALLBACKS.build(spec.name, **spec.params))
    return built
