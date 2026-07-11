"""The training loop: a custom, dependency-injected trainer.

PyTorch Lightning is intentionally not used: a small custom loop keeps the
dependency surface minimal, makes AMP / gradient-accumulation / resume fully
transparent, and gives complete control over checkpoint and provenance formats —
all desirable for a long-lived production system. Everything the loop uses is
built from configuration in :mod:`s2train.builder`.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

import torch

from .. import kaggle
from ..builder import (
    build_callbacks,
    build_dataloader,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    resolve_device,
)
from ..config import ExperimentConfig
from ..logging_setup import configure_logging, get_logger
from ..metrics import compute_metrics
from ..reproducibility import count_parameters, provenance, set_seed


class Trainer:
    """Trains one model according to an :class:`ExperimentConfig`."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Build all components and prepare the run directory.

        Args:
            config: The experiment configuration.
        """
        self.config = config
        set_seed(config.seed, deterministic=config.trainer.deterministic)

        # Stable, non-timestamped run directory keyed by experiment name so that
        # `resume: auto` finds the previous session's checkpoints (essential on
        # Kaggle, where each session starts fresh but /kaggle/working persists as
        # output). The default output root is redirected onto /kaggle/working.
        output_root = kaggle.default_output_root(config.output_root)
        self.output_dir = Path(output_root) / config.name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(self.output_dir / "training.log")

        self.device = resolve_device(config.trainer.device)
        if self.device.type == "cuda" and not config.trainer.deterministic:
            torch.backends.cudnn.benchmark = True  # fixed patch size -> faster
        self.model = build_model(config).to(self.device)
        self.loss_fn = build_loss(config)
        self.optimizer = build_optimizer(config, self.model)
        self.scheduler = build_scheduler(config, self.optimizer)
        self.callbacks = build_callbacks(config)

        self.train_loader = build_dataloader(
            config, config.data.train_split, shuffle=True, augment=config.data.augment)
        try:
            self.val_loader = build_dataloader(
                config, config.data.val_split, shuffle=False, augment=False)
        except (FileNotFoundError, ValueError) as exc:
            self.logger.warning("No validation split (%s); running train-only", exc)
            self.val_loader = None

        self._setup_amp()
        self.start_epoch = 0
        self.best_value = -float("inf") if config.checkpoint.mode == "max" else float("inf")
        self._stop = False
        self._maybe_resume()

    # ----- AMP -----------------------------------------------------------------

    def _setup_amp(self) -> None:
        """Resolve precision to the device and configure autocast + scaler.

        ``auto``/``amp`` picks bf16 on GPUs that support it (Ampere+) and fp16
        otherwise (e.g. Kaggle's T4/P100, where bf16 is not tensor-core
        accelerated); CPU always runs fp32. The fp16 path enables the gradient
        scaler; bf16 does not need it.
        """
        precision = self.config.trainer.precision.lower()
        if self.device.type != "cuda":
            precision = "fp32"
        elif precision in ("auto", "amp"):
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        self.precision = precision
        self.amp_enabled = precision in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        use_scaler = precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.logger.info("Precision resolved to '%s' on %s", precision, self.device)

    # ----- public API used by callbacks ---------------------------------------

    def to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch's tensors to the training device.

        Args:
            batch: A collated batch.

        Returns:
            The batch with tensors on ``self.device``.
        """
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run the model on a (device-resident) batch.

        Args:
            batch: A batch already on the device.

        Returns:
            The prediction ``(B, 13, H, W)``.
        """
        return self.model(batch)

    def request_stop(self) -> None:
        """Signal the loop to stop after the current epoch."""
        self._stop = True

    def _install_signal_handlers(self):
        """Install SIGINT/SIGTERM handlers that request a graceful stop.

        On Kaggle a time-limit or manual interruption sends a signal; catching it
        lets the current epoch finish and its ``latest.pt`` be written, so the
        next session can resume. Registration is best-effort (only in the main
        thread; some signals are unavailable on some platforms).

        Returns:
            A callable that restores the previous handlers.
        """
        previous: dict[int, Any] = {}

        def handler(signum, _frame) -> None:
            self.logger.warning("Received signal %s; will stop after this epoch "
                                "and save a resumable checkpoint", signum)
            self._stop = True

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):  # not main thread / unsupported
                pass

        def restore() -> None:
            for sig, prev in previous.items():
                try:
                    signal.signal(sig, prev)
                except (ValueError, OSError):
                    pass

        return restore

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> None:
        """Save a full checkpoint (model/optim/sched/scaler/epoch/best/config).

        Args:
            path: Destination checkpoint path.
            epoch: The epoch just completed.
            metrics: The epoch's metrics.
        """
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_value": self.best_value,
            "metrics": metrics,
            "config": self.config.to_dict(),
        }, tmp)
        tmp.replace(path)

    # ----- resume --------------------------------------------------------------

    def _maybe_resume(self) -> None:
        """Resume from a checkpoint if configured (``auto`` or explicit path).

        For ``auto`` the run's own ``checkpoints/latest.pt`` is preferred, then a
        checkpoint from a previous session found under ``resume_search_dirs``
        (``/kaggle/input`` is added automatically on Kaggle) — this is what makes
        training survive Kaggle session interruptions.
        """
        resume = self.config.resume
        if not resume:
            return
        if resume == "auto":
            search = kaggle.default_resume_search_dirs(self.config.resume_search_dirs)
            path = kaggle.find_resume_checkpoint(self.output_dir, self.config.name, search)
        else:
            path = Path(resume)
        if path is None or not Path(path).exists():
            self.logger.info("Resume requested but no checkpoint found (starting fresh)")
            return
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_value = state.get("best_value", self.best_value)
        self.start_epoch = int(state["epoch"]) + 1
        self.logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    # ----- loop ----------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            The experiment summary dict (also written to disk).
        """
        params = count_parameters(self.model)
        self.logger.info("Model '%s' | %s params | device=%s | precision=%s",
                         self.config.model.name, f"{params['total']:,}",
                         self.device, self.precision)
        self._write_model_summary()
        self._write_summary({"status": "running", "model_parameters": params,
                             "provenance": provenance(self.config.to_dict(),
                                                      self.config.data.root)})
        restore = self._install_signal_handlers()
        for cb in self.callbacks:
            cb.on_train_start(self)

        start = time.perf_counter()
        last_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate() if self._should_validate(epoch) else {}
            metrics = {"lr": self.optimizer.param_groups[0]["lr"],
                       **train_metrics, **val_metrics}
            self._step_scheduler(val_metrics)

            self.logger.info(
                "Epoch %d/%d | train/loss=%.4f%s", epoch + 1, self.config.trainer.epochs,
                metrics.get("train/loss", float("nan")),
                _fmt(val_metrics, self.config.checkpoint.monitor))
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)
            last_metrics = metrics
            if self._stop:
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        restore()
        duration = round(time.perf_counter() - start, 1)
        summary = self._final_summary(last_metrics, params, duration)
        self._write_summary(summary)
        self.logger.info("Training complete in %.1fs | best %s=%.4f", duration,
                         self.config.checkpoint.monitor, self.best_value)
        return summary

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch with AMP + gradient accumulation.

        Args:
            epoch: The epoch index.

        Returns:
            Averaged ``train/*`` metrics for the epoch.
        """
        self.model.train()
        accum = max(1, self.config.trainer.grad_accum_steps)
        clip = self.config.trainer.grad_clip_norm
        totals: dict[str, float] = {}
        count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            batch = self.to_device(batch)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp_enabled):
                pred = self.model(batch)
                loss, components = self.loss_fn(pred, batch["ground_truth"], batch["mask"])
            self.scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0:
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            if (step + 1) % self.config.trainer.log_interval == 0:
                self.logger.debug("  epoch %d step %d loss=%.4f", epoch + 1, step + 1,
                                  components.get("total", 0.0))
            if self._stop:  # interrupted (e.g. Kaggle time limit): end epoch early
                self.logger.info("Stop signalled; ending epoch %d early", epoch + 1)
                break
            limit = self.config.trainer.limit_batches
            if limit and (step + 1) >= limit:  # fast-dev: cap batches per epoch
                break
        return {f"train/{k}": v / max(1, count) for k, v in totals.items()} | \
               {"train/loss": totals.get("total", 0.0) / max(1, count)}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run validation, returning averaged ``val/*`` metrics.

        Returns:
            The averaged validation metrics.
        """
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals: dict[str, float] = {}
        loss_total = 0.0
        count = 0
        limit = self.config.trainer.limit_batches
        for batch in self.val_loader:
            batch = self.to_device(batch)
            pred = self.model(batch)
            loss, _ = self.loss_fn(pred, batch["ground_truth"], batch["mask"])
            loss_total += float(loss)
            metrics = compute_metrics(pred, batch["ground_truth"], batch["mask"],
                                      self.config.metrics)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            if limit and count >= limit:
                break
        out = {f"val/{k}": v / max(1, count) for k, v in totals.items()}
        out["val/loss"] = loss_total / max(1, count)
        return out

    def _should_validate(self, epoch: int) -> bool:
        """Return whether validation runs after ``epoch``."""
        return (self.val_loader is not None
                and (epoch + 1) % self.config.trainer.val_interval == 0)

    def _step_scheduler(self, val_metrics: dict[str, float]) -> None:
        """Advance the LR scheduler (plateau consumes the monitored metric).

        Args:
            val_metrics: The current validation metrics.
        """
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        if isinstance(self.scheduler, ReduceLROnPlateau):
            value = val_metrics.get(self.config.checkpoint.monitor)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()

    # ----- summaries -----------------------------------------------------------

    def _final_summary(self, metrics: dict[str, float], params: dict[str, int],
                       duration: float) -> dict[str, Any]:
        """Assemble the final experiment summary.

        Args:
            metrics: The last epoch's metrics.
            params: Model parameter counts.
            duration: Training duration in seconds.

        Returns:
            The summary dict.
        """
        return {
            "status": "completed",
            "experiment": self.config.name,
            "model": self.config.model.name,
            "model_parameters": params,
            "training_duration_seconds": duration,
            "best_metric": {"name": self.config.checkpoint.monitor, "value": self.best_value},
            "final_metrics": {k: round(v, 6) for k, v in metrics.items()},
            "output_dir": str(self.output_dir),
            "provenance": provenance(self.config.to_dict(), self.config.data.root),
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        """Write ``experiment_summary.json``.

        Args:
            summary: The summary dict.
        """
        (self.output_dir / "experiment_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    def _write_model_summary(self) -> None:
        """Measure and write ``model_summary.json`` (params/VRAM/throughput).

        Profiling failures never abort training; they are recorded in the file.
        """
        from ..summary import model_summary

        try:
            n_train = len(self.train_loader.dataset)
        except TypeError:  # pragma: no cover - dataset without __len__
            n_train = None
        info = model_summary(self.model, self.config, self.device,
                             num_train_samples=n_train)
        (self.output_dir / "model_summary.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")
        self.logger.info(
            "Summary: %.3fM params | %s | throughput %.1f samples/s | ~%.1f min/epoch",
            info.get("parameters_millions", 0.0),
            f"{info.get('peak_vram_mb') or info.get('estimated_vram_mb', 0)} MB VRAM",
            info.get("throughput_samples_per_sec", 0.0),
            info.get("estimated_epoch_minutes", 0.0) or 0.0)


def _fmt(val_metrics: dict[str, float], monitor: str) -> str:
    """Format the monitored validation metric for a log line."""
    if monitor in val_metrics:
        return f" | {monitor}={val_metrics[monitor]:.4f}"
    return ""


get_logger()  # ensure the logger exists at import time
