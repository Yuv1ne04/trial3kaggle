"""Parts 7-8 - micro-metric gate trainer for ReferenceResidualUNetV2.

A compact, self-contained trainer (decoupled from the v1 trainer so the original
baseline is untouched) that:

* trains the bounded v2 model with the repaired loss, AMP, gradient accumulation,
  gradient clipping, resume, and best/latest checkpoints;
* validates with **micro** (pixel-weighted) cloud-land metrics accumulated over
  the whole validation subset - never per-batch macro PSNR;
* selects the primary checkpoint on ``val/cloud_land_rmse_micro`` (min) and a
  secondary checkpoint on the best NDVI MAE;
* compares against the weighted-reference-mean baseline on the same batches.

It exposes ``run_gate`` which trains and then evaluates the acceptance criteria.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from s2audit.baselines import weighted_reference_mean
from s2audit.indices import ndvi
from s2audit.metrics import PairAccumulator, RegionAccumulator, build_regions, ssim_map
from s2train.builder import (build_loss, build_model, build_optimizer, build_scheduler,
                             resolve_device)
from s2train.config import ExperimentConfig


def build_curated_loader(root: str, sample_files: list[str], *, batch_size: int = 8,
                         num_workers: int = 0, augment: bool = False,
                         reflectance_scale: float = 10000.0, max_references: int = 4,
                         shuffle: bool = False, seed: int = 1234):
    """Build a DataLoader over an explicit list of manifest files."""
    from pathlib import Path as _P

    from torch.utils.data import DataLoader

    from s2train.datasets import SyntheticDataset, collate_batch

    ds = SyntheticDataset(root, split="test", max_references=max_references,
                          reflectance_scale=reflectance_scale, augment=augment, seed=seed)
    ds.inner.sample_files = [_P(f) for f in sample_files]      # curated subset
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_batch, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


@dataclass
class MicroValidation:
    """Micro cloud / cloud-land metrics for the model and the reference baseline."""

    model_cloud_land: RegionAccumulator = field(default_factory=RegionAccumulator)
    model_ndvi: PairAccumulator = field(default_factory=PairAccumulator)
    base_cloud_land: RegionAccumulator = field(default_factory=RegionAccumulator)
    base_ndvi: PairAccumulator = field(default_factory=PairAccumulator)
    neg: float = 0.0
    over: float = 0.0
    total: float = 0.0

    def update(self, pred, base, target, mask):
        regions = build_regions(target, mask)
        cl = regions["cloud_land"]
        self.model_cloud_land.update(pred, target, cl, ssim_map(pred, target))
        self.base_cloud_land.update(base, target, cl, ssim_map(base, target))
        for acc, img in ((self.model_ndvi, pred), (self.base_ndvi, base)):
            ip, it = ndvi(img), ndvi(target)
            v = cl & ip.valid & it.valid
            sel = v.expand_as(ip.value)
            acc.update(ip.value[sel], it.value[sel])
        cbp = regions["cloud"].expand_as(pred)
        self.total += cbp.sum().item()
        self.neg += ((pred < 0.0) & cbp).sum().item()
        self.over += ((pred > 1.0) & cbp).sum().item()

    def summary(self) -> dict[str, float]:
        m, b = self.model_cloud_land.result(), self.base_cloud_land.result()
        return {
            "cloud_land_rmse_micro": m["rmse_micro"], "cloud_land_psnr_micro": m["psnr_micro"],
            "cloud_land_mae_micro": m["mae_micro"], "cloud_land_sam_micro": m["sam_micro"],
            "cloud_land_ssim_micro": m["ssim_micro"], "ndvi_mae": self.model_ndvi.mae,
            "baseline_cloud_land_rmse_micro": b["rmse_micro"],
            "baseline_cloud_land_psnr_micro": b["psnr_micro"],
            "baseline_ndvi_mae": self.base_ndvi.mae,
            "negative_output_fraction": self.neg / self.total if self.total else 0.0,
            "over_one_output_fraction": self.over / self.total if self.total else 0.0,
        }


class GateTrainer:
    """Trains the bounded v2 model with micro-metric checkpoint selection."""

    def __init__(self, config: ExperimentConfig, output_dir: str | Path,
                 device: str = "auto") -> None:
        self.config = config
        self.device = resolve_device(device)
        self.output_dir = Path(output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.model = build_model(config).to(self.device)
        self.loss = build_loss(config)
        self.optimizer = build_optimizer(config, self.model)
        self.scheduler = build_scheduler(config, self.optimizer)
        # AMP precision: fp16 on <Ampere, bf16 on Ampere+, off on CPU.
        self.amp = self.device.type == "cuda"
        self.amp_dtype = torch.float16
        if self.amp and torch.cuda.get_device_capability(self.device)[0] >= 8:
            self.amp_dtype = torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.amp_dtype == torch.float16)
        self.best_rmse = math.inf
        self.best_ndvi = math.inf

    def _forward_loss(self, batch):
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = self.model(batch)
        total, comps = self.loss(pred, batch["ground_truth"], batch["mask"])
        return total, comps

    def train_epoch(self, loader, grad_accum: int, grad_clip: float) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running = 0.0
        count = 0
        for i, batch in enumerate(loader):
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                total, _ = self._forward_loss(batch)
            self.scaler.scale(total / grad_accum).backward()
            running += float(total.detach())
            count += 1
            if (i + 1) % grad_accum == 0:
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        if count % grad_accum != 0:               # flush trailing accumulation
            if grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return running / max(1, count)

    @torch.no_grad()
    def validate(self, loader) -> dict[str, float]:
        self.model.eval()
        val = MicroValidation()
        for batch in loader:
            dev = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = self.model(dev).float().cpu()
            base = weighted_reference_mean(batch).float()
            val.update(pred, base, batch["ground_truth"].float(), batch["mask"].float())
        return val.summary()

    def save(self, name: str, epoch: int, metrics: dict) -> None:
        torch.save({"epoch": epoch, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "metrics": metrics,
                    "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {}},
                   self.output_dir / "checkpoints" / name)

    def fit(self, train_loader, val_loader, *, epochs: int, grad_accum: int = 1,
            grad_clip: float = 1.0, log: bool = True) -> list[dict]:
        history = []
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, grad_accum, grad_clip)
            if self.scheduler is not None:
                self.scheduler.step()
            metrics = self.validate(val_loader)
            metrics["train_loss"] = train_loss
            metrics["epoch"] = epoch
            history.append(metrics)
            self.save("latest.pt", epoch, metrics)
            if metrics["cloud_land_rmse_micro"] < self.best_rmse:
                self.best_rmse = metrics["cloud_land_rmse_micro"]
                self.save("best_rmse.pt", epoch, metrics)
            if metrics["ndvi_mae"] < self.best_ndvi:
                self.best_ndvi = metrics["ndvi_mae"]
                self.save("best_ndvi.pt", epoch, metrics)
            if log:
                print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                      f"val cl_RMSE {metrics['cloud_land_rmse_micro']:.4f} "
                      f"(base {metrics['baseline_cloud_land_rmse_micro']:.4f}) | "
                      f"cl_PSNR {metrics['cloud_land_psnr_micro']:.2f} "
                      f"(base {metrics['baseline_cloud_land_psnr_micro']:.2f}) | "
                      f"NDVI_MAE {metrics['ndvi_mae']:.4f} "
                      f"(base {metrics['baseline_ndvi_mae']:.4f}) | "
                      f"neg {metrics['negative_output_fraction']:.3f}", flush=True)
        return history
