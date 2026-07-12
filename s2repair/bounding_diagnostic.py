"""Part 1 - post-hoc bounded-output diagnostic for the existing checkpoint.

Evaluates the *unchanged* ``best.pt`` under three output treatments to attribute
the failure to unbounded outputs, without retraining or modifying the checkpoint:

    A. raw prediction (as the model emits it),
    B. clamp to [0, 1],
    C. clip to empirically justified per-band train-set bounds.

Writes ``current_checkpoint_bounding_comparison.csv`` (all metrics per variant)
and ``prediction_range_by_band.csv`` (raw prediction quantiles per band).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from s2audit.datasets_compat import build_test_loader
from s2audit.indices import ndvi
from s2audit.metrics import PairAccumulator, RegionAccumulator, build_regions, ssim_map
from s2audit import BAND_ORDER

_QUANTILES = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]
_QNAMES = ["min", "p0.1", "p1", "p50", "p99", "p99.9", "max"]


def compute_train_band_bounds(root: str | Path, *, n_samples: int = 200,
                              low_q: float = 0.001, high_q: float = 0.999,
                              seed: int = 1234) -> np.ndarray | None:
    """Empirical per-band reflectance bounds from a sample of train ground truth.

    Args:
        root: Dataset root.
        n_samples: Number of train samples to pool.
        low_q / high_q: Lower/upper quantiles for the bounds.
        seed: Sampling seed.

    Returns:
        A ``(13, 2)`` array of ``[low, high]`` per band, or ``None`` if the
        train split is unavailable.
    """
    try:
        loader = build_test_loader(root, split="train", max_samples=n_samples,
                                   batch_size=8, seed=seed)
    except Exception:
        return None
    lows = [[] for _ in range(13)]
    highs = [[] for _ in range(13)]
    for batch in loader:
        gt = batch["ground_truth"]
        for b in range(13):
            v = gt[:, b].flatten()
            lows[b].append(torch.quantile(v, low_q).item())
            highs[b].append(torch.quantile(v, high_q).item())
    return np.array([[float(np.mean(lows[b])), float(np.mean(highs[b]))]
                     for b in range(13)], dtype=np.float32)


class _Variant:
    """Accumulators for one output-treatment variant."""

    def __init__(self) -> None:
        self.cloud = RegionAccumulator()
        self.cloud_land = RegionAccumulator()
        self.ndvi = PairAccumulator()
        self.neg = 0.0
        self.over = 0.0
        self.total = 0.0

    def update(self, pred, target, regions):
        smap = ssim_map(pred, target)
        self.cloud.update(pred, target, regions["cloud"], smap)
        self.cloud_land.update(pred, target, regions["cloud_land"], smap)
        ip, it = ndvi(pred), ndvi(target)
        valid = regions["cloud"] & ip.valid & it.valid
        sel = valid.expand_as(ip.value)
        self.ndvi.update(ip.value[sel], it.value[sel])
        cbp = regions["cloud"].expand_as(pred)
        self.total += cbp.sum().item()
        self.neg += ((pred < 0.0) & cbp).sum().item()
        self.over += ((pred > 1.0) & cbp).sum().item()

    def row(self, name: str) -> dict[str, Any]:
        c, cl = self.cloud.result(), self.cloud_land.result()
        return {
            "variant": name,
            "cloud_psnr_micro": c["psnr_micro"], "cloud_land_psnr_micro": cl["psnr_micro"],
            "cloud_rmse_micro": c["rmse_micro"], "cloud_mae_micro": c["mae_micro"],
            "cloud_ssim": c["ssim_micro"], "cloud_sam": c["sam_micro"],
            "ndvi_mae": self.ndvi.mae, "ndvi_rmse": self.ndvi.rmse, "ndvi_bias": self.ndvi.bias,
            "negative_output_fraction": self.neg / self.total if self.total else float("nan"),
            "over_one_output_fraction": self.over / self.total if self.total else float("nan"),
        }


def run_bounding_diagnostic(checkpoint: str | Path, root: str | Path, output_dir: str | Path,
                            *, max_samples: int = 240, batch_size: int = 8,
                            device: str = "auto", seed: int = 1234,
                            reflectance_scale: float = 10000.0) -> dict[str, Any]:
    """Run the three-variant bounding diagnostic and write the CSVs.

    Args:
        checkpoint: Path to ``best.pt`` (never modified).
        root: Dataset root.
        output_dir: Output directory.
        max_samples: Test samples to evaluate (0 = full).
        batch_size: Inference batch size.
        device: Device string.
        seed: Random seed.
        reflectance_scale: DN -> reflectance divisor.

    Returns:
        A summary dict.
    """
    from s2train.inference import Predictor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor.from_checkpoint(checkpoint, device=device)
    bounds = compute_train_band_bounds(root, seed=seed)
    bounds_t = torch.tensor(bounds) if bounds is not None else None

    variants = {"A_raw": _Variant(), "B_clamp01": _Variant()}
    if bounds_t is not None:
        variants["C_perband_bounds"] = _Variant()

    band_values: list[list[np.ndarray]] = [[] for _ in range(13)]
    per_band_cap = 400_000
    band_counts = [0] * 13
    loader = build_test_loader(root, split="test", max_samples=max_samples,
                               batch_size=batch_size, seed=seed,
                               reflectance_scale=reflectance_scale)
    n = 0
    for batch in loader:
        raw = predictor.predict_batch(batch).float()
        target = batch["ground_truth"].float()
        regions = build_regions(target, batch["mask"].float())
        n += raw.shape[0]

        variants["A_raw"].update(raw, target, regions)
        variants["B_clamp01"].update(raw.clamp(0.0, 1.0), target, regions)
        if bounds_t is not None:
            lo = bounds_t[:, 0].view(1, -1, 1, 1)
            hi = bounds_t[:, 1].view(1, -1, 1, 1)
            variants["C_perband_bounds"].update(torch.min(torch.max(raw, lo), hi), target, regions)

        # Raw prediction quantile collection over cloud pixels (capped per band).
        cloud = regions["cloud"][:, 0]
        for b in range(13):
            if band_counts[b] >= per_band_cap:
                continue
            vals = raw[:, b][cloud].detach().cpu().numpy()
            if vals.size:
                band_values[b].append(vals)
                band_counts[b] += vals.size

    # Write comparison CSV.
    rows = [variants[k].row(k) for k in variants]
    comp_path = output_dir / "current_checkpoint_bounding_comparison.csv"
    with comp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Write per-band raw-prediction quantiles.
    range_path = output_dir / "prediction_range_by_band.csv"
    with range_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["band"] + _QNAMES + ["train_bound_low", "train_bound_high"])
        for b, name in enumerate(BAND_ORDER):
            if band_values[b]:
                arr = np.concatenate(band_values[b])
                qs = np.quantile(arr, _QUANTILES).tolist()
            else:
                qs = [float("nan")] * len(_QUANTILES)
            lo = float(bounds[b, 0]) if bounds is not None else ""
            hi = float(bounds[b, 1]) if bounds is not None else ""
            w.writerow([name] + [round(q, 6) for q in qs] + [lo, hi])

    summary = {
        "n_samples": n,
        "variants": {k: variants[k].row(k) for k in variants},
        "train_bounds_available": bounds is not None,
        "artefacts": {"comparison": str(comp_path), "prediction_range": str(range_path)},
    }
    return summary
