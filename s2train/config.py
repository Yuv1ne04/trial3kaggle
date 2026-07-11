"""Typed experiment configuration and YAML loading.

An :class:`ExperimentConfig` fully specifies a run: which model/dataset/optimizer/
scheduler/loss/metrics/callbacks to build (by registry name) and all their
hyper-parameters. Loading is pure data — nothing is instantiated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ComponentSpec:
    """A registry-driven component: a ``name`` plus its constructor ``params``.

    Attributes:
        name: The registry key selecting the implementation.
        params: Keyword arguments passed to the implementation's factory.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: Any, default_name: str = "") -> "ComponentSpec":
        """Build a spec from a string, mapping or existing spec.

        Args:
            value: ``"name"``, ``{name, params}`` or a :class:`ComponentSpec`.
            default_name: Name to use when ``value`` is ``None``.

        Returns:
            A :class:`ComponentSpec`.
        """
        if isinstance(value, ComponentSpec):
            return value
        if value is None:
            return cls(name=default_name)
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            return cls(name=value.get("name", default_name),
                       params=dict(value.get("params", {})))
        raise ValueError(f"Cannot parse component spec from {value!r}")


@dataclass(slots=True)
class DataConfig:
    """Dataset and dataloader configuration.

    Attributes:
        name: Dataset registry key (e.g. ``"synthetic"``).
        root: Dataset root directory.
        batch_size: Batch size per step.
        num_workers: DataLoader worker processes.
        patch_size: Patch side length (informational / for tiling).
        max_references: Fixed reference-slot count.
        reflectance_scale: DN -> reflectance divisor.
        train_split / val_split / test_split: Split folder names.
        augment: Whether to apply D4 augmentation on the training split.
        difficulty: Optional curriculum band filter (e.g. ``"easy"``).
        pin_memory: DataLoader ``pin_memory`` (auto-disabled without CUDA).
        persistent_workers: Keep workers alive between epochs (big speedup on
            Kaggle when ``num_workers > 0``; only applied when it is).
        prefetch_factor: Batches prefetched per worker (only when workers > 0).
        drop_last: Drop the last incomplete training batch.
        params: Extra dataset-specific keyword arguments.
    """

    name: str = "synthetic"
    root: str = "synthetic_dataset"
    batch_size: int = 8
    num_workers: int = 4
    patch_size: int = 256
    max_references: int = 4
    reflectance_scale: float = 10000.0
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    augment: bool = True
    difficulty: str | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True
    max_samples: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainerConfig:
    """Training-loop configuration.

    Attributes:
        epochs: Number of epochs.
        precision: ``"fp32"``, ``"bf16"`` or ``"fp16"`` (AMP on CUDA).
        grad_accum_steps: Gradient-accumulation steps (effective batch multiplier).
        grad_clip_norm: Max gradient norm (0 disables clipping).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        val_interval: Validate every N epochs.
        log_interval: Log every N optimizer steps.
        ema_decay: Weight-EMA decay (0 disables EMA).
        deterministic: Enable deterministic algorithms where possible.
        limit_batches: If > 0, cap the number of train/val batches per epoch —
            a fast-dev switch to exercise the whole pipeline in seconds before
            committing GPU hours.
        data_parallel: Use ``nn.DataParallel`` across all visible GPUs when more
            than one is present (e.g. Kaggle's T4 x2). Set ``False`` to force a
            single GPU.
    """

    epochs: int = 100
    precision: str = "bf16"
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    device: str = "auto"
    val_interval: int = 1
    log_interval: int = 50
    ema_decay: float = 0.0
    deterministic: bool = False
    limit_batches: int = 0
    data_parallel: bool = True


@dataclass(slots=True)
class CheckpointConfig:
    """Checkpointing configuration.

    Attributes:
        monitor: Validation metric used to select the best checkpoint.
        mode: ``"min"`` or ``"max"`` for ``monitor``.
        save_best: Whether to keep the best checkpoint.
        save_latest: Whether to keep the latest checkpoint (for resume).
        every_n_epochs: Also snapshot every N epochs (0 disables).
    """

    monitor: str = "val/psnr_cloud"
    mode: str = "max"
    save_best: bool = True
    save_latest: bool = True
    every_n_epochs: int = 0


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Early-stopping configuration.

    Attributes:
        enabled: Whether early stopping is active.
        monitor: Metric to watch.
        mode: ``"min"`` or ``"max"``.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change counting as improvement.
    """

    enabled: bool = True
    monitor: str = "val/psnr_cloud"
    mode: str = "max"
    patience: int = 15
    min_delta: float = 0.0


@dataclass(slots=True)
class VisualizationConfig:
    """Prediction-visualisation configuration.

    Attributes:
        enabled: Whether to save visual panels.
        every_n_epochs: Save panels every N epochs.
        num_samples: Number of validation samples to visualise.
        rgb_bands: 1-based band indices used for the RGB preview (B4,B3,B2).
    """

    enabled: bool = True
    every_n_epochs: int = 5
    num_samples: int = 4
    rgb_bands: tuple[int, int, int] = (4, 3, 2)


@dataclass(slots=True)
class SanityConfig:
    """Sanity / overfit-a-tiny-subset configuration.

    Attributes:
        enabled: Run in sanity mode (overrides epochs/samples/augment).
        num_samples: Size of the tiny subset to overfit (32-64 recommended).
        epochs: Number of epochs to run in sanity mode.
        require_loss_drop: Fraction the train loss must drop by to pass
            (e.g. ``0.2`` = at least 20% lower than the first epoch).
    """

    enabled: bool = False
    num_samples: int = 64
    epochs: int = 2
    require_loss_drop: float = 0.1


@dataclass(slots=True)
class ExperimentConfig:
    """A complete, self-contained experiment specification.

    Attributes:
        name: Experiment name (used for the output directory).
        seed: Master random seed.
        output_root: Root directory for experiment outputs.
        tags: Free-form tags recorded in the summary.
        model: Model component spec.
        data: Dataset/dataloader configuration.
        optimizer: Optimizer component spec.
        scheduler: Scheduler component spec.
        loss: Loss component spec (typically the composite loss).
        metrics: Metric registry names to compute.
        callbacks: Extra callback specs (checkpoint/early-stop/vis are implicit).
        trainer: Training-loop configuration.
        checkpoint: Checkpointing configuration.
        early_stopping: Early-stopping configuration.
        visualization: Visualisation configuration.
        resume: ``"auto"``, a checkpoint path, or ``None``.
        resume_search_dirs: Extra directories searched by ``resume: auto`` for a
            checkpoint from a previous (e.g. interrupted) session. ``/kaggle/input``
            is added automatically on Kaggle.
    """

    name: str = "experiment"
    seed: int = 1234
    output_root: str = "experiments"
    tags: list[str] = field(default_factory=list)
    model: ComponentSpec = field(default_factory=lambda: ComponentSpec("unet"))
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: ComponentSpec = field(default_factory=lambda: ComponentSpec("adamw"))
    scheduler: ComponentSpec = field(default_factory=lambda: ComponentSpec("cosine"))
    loss: ComponentSpec = field(default_factory=lambda: ComponentSpec("composite"))
    metrics: list[str] = field(default_factory=lambda: [
        "psnr", "ssim", "sam", "rmse", "mae", "ergas"])
    callbacks: list[ComponentSpec] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    sanity: SanityConfig = field(default_factory=SanityConfig)
    resume: str | None = None
    resume_search_dirs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation."""
        return asdict(self)


def _build_dataclass(cls: type, value: Any):
    """Construct a (nested) dataclass from a mapping, ignoring unknown keys.

    Args:
        cls: The dataclass type.
        value: A mapping of field values (or an instance/None).

    Returns:
        An instance of ``cls``.
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in value.items() if k in valid})


def load_config(path: Path | str, **overrides: Any) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Nested blocks are parsed into their dataclasses; component blocks
    (``model``/``optimizer``/``scheduler``/``loss``/``callbacks``) accept either a
    bare name or a ``{name, params}`` mapping. Top-level keyword ``overrides``
    replace loaded values.

    Args:
        path: Path to the YAML config file.
        **overrides: Top-level field overrides (``None`` ignored).

    Returns:
        A populated :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML or not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    raw.update({k: v for k, v in overrides.items() if v is not None})

    return ExperimentConfig(
        name=raw.get("name", "experiment"),
        seed=int(raw.get("seed", 1234)),
        output_root=raw.get("output_root", "experiments"),
        tags=list(raw.get("tags", [])),
        model=ComponentSpec.parse(raw.get("model"), "unet"),
        data=_build_dataclass(DataConfig, raw.get("data")),
        optimizer=ComponentSpec.parse(raw.get("optimizer"), "adamw"),
        scheduler=ComponentSpec.parse(raw.get("scheduler"), "cosine"),
        loss=ComponentSpec.parse(raw.get("loss"), "composite"),
        metrics=list(raw.get("metrics", ["psnr", "ssim", "sam", "rmse", "mae", "ergas"])),
        callbacks=[ComponentSpec.parse(c) for c in raw.get("callbacks", [])],
        trainer=_build_dataclass(TrainerConfig, raw.get("trainer")),
        checkpoint=_build_dataclass(CheckpointConfig, raw.get("checkpoint")),
        early_stopping=_build_dataclass(EarlyStoppingConfig, raw.get("early_stopping")),
        visualization=_build_dataclass(VisualizationConfig, raw.get("visualization")),
        sanity=_build_dataclass(SanityConfig, raw.get("sanity")),
        resume=raw.get("resume"),
        resume_search_dirs=list(raw.get("resume_search_dirs", [])),
    )
