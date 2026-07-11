"""Training callbacks (registered for config selection).

Callbacks observe the training loop through a small, stable interface and never
mutate the model directly. Checkpointing, early stopping, CSV logging, image
visualisation and TensorBoard logging are all callbacks, so they compose freely.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..registry import CALLBACKS

if TYPE_CHECKING:  # pragma: no cover
    from ..trainers import Trainer


class Callback:
    """Base callback with no-op hooks."""

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before training begins."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Called after each epoch's validation.

        Args:
            trainer: The trainer.
            epoch: The completed epoch index.
            metrics: All scalar metrics for the epoch (train + val).
        """

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after training finishes."""


def _improved(value: float, best: float, mode: str, min_delta: float) -> bool:
    """Return whether ``value`` improves on ``best`` under ``mode``."""
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


@CALLBACKS.register("checkpoint")
class CheckpointCallback(Callback):
    """Saves the best and latest checkpoints (and optional periodic snapshots)."""

    def __init__(self, monitor: str = "val/psnr_cloud", mode: str = "max",
                 save_best: bool = True, save_latest: bool = True,
                 every_n_epochs: int = 0) -> None:
        """Initialise the checkpoint callback.

        Args:
            monitor: Metric selecting the best checkpoint.
            mode: ``"min"`` or ``"max"``.
            save_best: Keep the best checkpoint.
            save_latest: Keep the latest checkpoint (for resume).
            every_n_epochs: Also snapshot every N epochs (0 disables).
        """
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_latest = save_latest
        self.every_n_epochs = every_n_epochs
        self.best = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Save latest/best/periodic checkpoints based on the monitored metric."""
        ckpt_dir = trainer.checkpoint_dir
        if self.save_latest:
            trainer.save_checkpoint(ckpt_dir / "latest.pt", epoch, metrics)
        value = metrics.get(self.monitor)
        if self.save_best and value is not None and not np.isnan(value):
            if _improved(value, self.best, self.mode, 0.0):
                self.best = value
                trainer.best_value = value
                trainer.save_checkpoint(ckpt_dir / "best.pt", epoch, metrics)
                trainer.logger.info("New best %s=%.4f (epoch %d)", self.monitor, value, epoch)
        if self.every_n_epochs and (epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:04d}.pt", epoch, metrics)


@CALLBACKS.register("early_stopping")
class EarlyStoppingCallback(Callback):
    """Stops training when the monitored metric stops improving."""

    def __init__(self, monitor: str = "val/psnr_cloud", mode: str = "max",
                 patience: int = 15, min_delta: float = 0.0) -> None:
        """Initialise the early-stopping callback.

        Args:
            monitor: Metric to watch.
            mode: ``"min"`` or ``"max"``.
            patience: Epochs without improvement before stopping.
            min_delta: Minimum change counting as improvement.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf") if mode == "max" else float("inf")
        self.waited = 0

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Update patience and request a stop if exhausted."""
        value = metrics.get(self.monitor)
        if value is None or np.isnan(value):
            return
        if _improved(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.waited = 0
        else:
            self.waited += 1
            if self.waited >= self.patience:
                trainer.logger.info("Early stopping at epoch %d (no %s improvement in %d)",
                                    epoch, self.monitor, self.patience)
                trainer.request_stop()


@CALLBACKS.register("csv_logger")
class CSVLoggerCallback(Callback):
    """Appends per-epoch metrics to ``metrics.csv`` and ``training_curves.csv``."""

    def __init__(self, curve_keys: list[str] | None = None) -> None:
        """Initialise the CSV logger.

        Args:
            curve_keys: Metric keys written to ``training_curves.csv`` (a compact
                subset); ``None`` uses a sensible default.
        """
        self.curve_keys = curve_keys or [
            "epoch", "lr", "train/loss", "val/loss", "val/psnr_cloud", "val/sam_cloud"]

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        """Append a row to a CSV, writing the header if the file is new."""
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Write the full metrics row and the compact curves row."""
        full = {"epoch": epoch, **{k: round(v, 6) for k, v in metrics.items()}}
        self._append(trainer.output_dir / "metrics.csv", full)
        curve = {k: full.get(k, metrics.get(k, "")) for k in self.curve_keys}
        curve["epoch"] = epoch
        self._append(trainer.output_dir / "training_curves.csv", curve)


@CALLBACKS.register("visualizer")
class VisualizerCallback(Callback):
    """Saves prediction / GT / difference / mask / RGB panels every N epochs."""

    def __init__(self, every_n_epochs: int = 5, num_samples: int = 4,
                 rgb_bands: tuple[int, int, int] = (4, 3, 2), gain: float = 3.0) -> None:
        """Initialise the visualizer.

        Args:
            every_n_epochs: Save panels every N epochs.
            num_samples: Number of validation samples to visualise.
            rgb_bands: 1-based band indices for the RGB preview.
            gain: Display gain applied to reflectance before clipping to [0,1].
        """
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.rgb_idx = [b - 1 for b in rgb_bands]
        self.gain = gain

    def _rgb(self, img: torch.Tensor) -> np.ndarray:
        """Convert a 13-band tensor to a display RGB array ``(H, W, 3)``."""
        rgb = img[self.rgb_idx].detach().cpu().numpy()
        rgb = np.clip(rgb * self.gain, 0.0, 1.0)
        return np.transpose(rgb, (1, 2, 0))

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Render and save comparison panels for a few validation samples."""
        if (epoch + 1) % self.every_n_epochs != 0 or trainer.val_loader is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = trainer.output_dir / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = next(iter(trainer.val_loader))
        batch = trainer.to_device(batch)
        trainer.model.eval()
        with torch.no_grad():
            pred = trainer.forward(batch)
        n = min(self.num_samples, batch["cloudy"].shape[0])
        titles = ["Cloudy input", "Cloud mask", "Prediction", "Ground truth", "|Diff|"]
        fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n), squeeze=False)
        for i in range(n):
            cloudy, gt, mask = batch["cloudy"][i], batch["ground_truth"][i], batch["mask"][i]
            pr = pred[i]
            diff = np.clip(np.abs(self._rgb(pr) - self._rgb(gt)) * 2, 0, 1)
            panels = [self._rgb(cloudy), mask[0].cpu().numpy(), self._rgb(pr),
                      self._rgb(gt), diff]
            for j, (img, title) in enumerate(zip(panels, titles)):
                ax = axes[i][j]
                ax.imshow(img, cmap="gray" if j == 1 else None)
                if i == 0:
                    ax.set_title(title)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"epoch_{epoch + 1:04d}.png", dpi=90)
        plt.close(fig)


@CALLBACKS.register("baseline_visualizer")
class BaselineVisualizerCallback(Callback):
    """Saves a 7-panel comparison per validation epoch for the baseline model.

    Panels: synthetic input RGB, ground-truth RGB, prediction RGB, cloud mask,
    difference map (signed, RGB), absolute-error map, cloud-only error map.
    """

    def __init__(self, every_n_epochs: int = 1, num_samples: int = 4,
                 rgb_bands: tuple[int, int, int] = (4, 3, 2), gain: float = 3.0) -> None:
        """Initialise the baseline visualizer.

        Args:
            every_n_epochs: Save panels every N epochs.
            num_samples: Number of validation samples to visualise.
            rgb_bands: 1-based band indices for the RGB preview.
            gain: Display gain applied to reflectance before clipping.
        """
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.rgb_idx = [b - 1 for b in rgb_bands]
        self.gain = gain

    def _rgb(self, img: torch.Tensor) -> np.ndarray:
        """Convert a 13-band tensor to a display RGB array ``(H, W, 3)``."""
        rgb = img[self.rgb_idx].detach().cpu().numpy()
        return np.transpose(np.clip(rgb * self.gain, 0.0, 1.0), (1, 2, 0))

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Render and save the 7-panel comparison for a few samples."""
        if (epoch + 1) % self.every_n_epochs != 0 or trainer.val_loader is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = trainer.output_dir / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = trainer.to_device(next(iter(trainer.val_loader)))
        trainer.model.eval()
        with torch.no_grad():
            pred = trainer.forward(batch)
        n = min(self.num_samples, batch["cloudy"].shape[0])
        titles = ["Synthetic input", "Ground truth", "Prediction", "Cloud mask",
                  "Difference", "Absolute error", "Cloud-only error"]
        fig, axes = plt.subplots(n, 7, figsize=(21, 3 * n), squeeze=False)
        for i in range(n):
            gt_rgb = self._rgb(batch["ground_truth"][i])
            pr_rgb = self._rgb(pred[i])
            mask = batch["mask"][i, 0].detach().cpu().numpy()
            # Reduce on-GPU with torch, then move to host (np.abs on a CUDA
            # tensor would try to convert before the .cpu() and fail).
            abs_err = (pred[i] - batch["ground_truth"][i]).abs().mean(0).detach().cpu().numpy()
            cloud_err = abs_err * mask
            panels = [self._rgb(batch["cloudy"][i]), gt_rgb, pr_rgb, mask,
                      np.clip((pr_rgb - gt_rgb) * 2 + 0.5, 0, 1), abs_err, cloud_err]
            cmaps = [None, None, None, "gray", None, "magma", "magma"]
            for j, (img, title, cmap) in enumerate(zip(panels, titles, cmaps)):
                ax = axes[i][j]
                ax.imshow(img, cmap=cmap)
                if i == 0:
                    ax.set_title(title, fontsize=10)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"epoch_{epoch + 1:04d}.png", dpi=90)
        plt.close(fig)


@CALLBACKS.register("tensorboard")
class TensorBoardCallback(Callback):
    """Logs scalar metrics (and RGB previews) to TensorBoard."""

    def __init__(self, rgb_bands: tuple[int, int, int] = (4, 3, 2),
                 log_images: bool = True, gain: float = 3.0) -> None:
        """Initialise the TensorBoard callback.

        Args:
            rgb_bands: 1-based RGB band indices for image logging.
            log_images: Whether to log RGB previews.
            gain: Display gain for previews.
        """
        self.rgb_idx = [b - 1 for b in rgb_bands]
        self.log_images = log_images
        self.gain = gain
        self.writer = None

    def on_train_start(self, trainer: "Trainer") -> None:
        """Open the TensorBoard writer."""
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(trainer.output_dir / "tensorboard"))

    def on_epoch_end(self, trainer: "Trainer", epoch: int,
                     metrics: dict[str, float]) -> None:
        """Log scalar metrics for the epoch."""
        if self.writer is None:
            return
        for key, value in metrics.items():
            if not np.isnan(value):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Flush and close the writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
