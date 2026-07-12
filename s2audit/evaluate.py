"""Streaming test-split evaluation orchestrator (Parts 3-8).

Runs the saved best checkpoint once over the untouched test split and, in a
single pass, accumulates:

* region reconstruction metrics (micro + macro) - Part 3;
* per-band metrics - Part 5;
* operational vegetation-index metrics (cloud + land) - Part 4;
* non-learned baseline comparisons - Part 6;
* stratified metrics (difficulty / coverage / reference-count / surface) - Part 7.

Everything is accumulated as O(1) scalars (plus per-sample lists for macro), so
memory does not grow with the split size and the full 89k test set can be
streamed. Provenance (checkpoint hash, git commit, config, seed, timestamp) is
recorded for Part 8.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import BAND_ORDER
from .baselines import available_baselines
from .datasets_compat import build_test_loader
from .indices import VEGETATION_INDICES, ndvi, surface_category
from .metrics import (ErgasAccumulator, PairAccumulator, RegionAccumulator,
                      build_regions, ms_ssim_per_sample, ssim_map)
from .stratify import sample_strata

#: Regions reported for the model (primary = cloud-region variants).
_MODEL_REGIONS = ("cloud", "cloud_land", "cloud_ocean", "whole", "land", "ocean", "clear")


@dataclass
class ModelState:
    """All model accumulators for one evaluation pass."""

    regions: dict[str, RegionAccumulator] = field(default_factory=dict)
    ergas_cloud: ErgasAccumulator = field(default_factory=ErgasAccumulator)
    ergas_whole: ErgasAccumulator = field(default_factory=ErgasAccumulator)
    band_cloud: list = field(default_factory=lambda: [PairAccumulator() for _ in range(13)])
    band_whole: list = field(default_factory=lambda: [PairAccumulator() for _ in range(13)])
    veg_cloud: dict = field(default_factory=lambda: {k: PairAccumulator() for k in VEGETATION_INDICES})
    veg_land: dict = field(default_factory=lambda: {k: PairAccumulator() for k in VEGETATION_INDICES})
    msssim: list = field(default_factory=list)
    # Prediction-sanity: unphysical reflectance in the reconstructed region.
    neg_pixels: float = 0.0
    over_pixels: float = 0.0
    total_cloud_bandpix: float = 0.0

    def region(self, name: str) -> RegionAccumulator:
        return self.regions.setdefault(name, RegionAccumulator())


@dataclass
class BaselineState:
    """Cloud-region accumulators for one baseline method."""

    cloud: RegionAccumulator = field(default_factory=RegionAccumulator)
    ndvi_cloud: PairAccumulator = field(default_factory=PairAccumulator)


def _masked_pair(pred_field: torch.Tensor, tgt_field: torch.Tensor,
                 mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select matched values from ``pred``/``target`` where ``mask`` is True."""
    sel = mask.expand_as(pred_field)
    return pred_field[sel], tgt_field[sel]


def _update_veg(acc: dict, pred: torch.Tensor, target: torch.Tensor,
                region: torch.Tensor) -> None:
    """Accumulate vegetation-index error over a region (valid pixels only)."""
    for name, fn in VEGETATION_INDICES.items():
        ip, it = fn(pred), fn(target)
        valid = region & ip.valid & it.valid
        p, t = _masked_pair(ip.value, it.value, valid)
        acc[name].update(p, t)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def evaluate_test_split(checkpoint: Path | str, root: Path | str, output_dir: Path | str,
                        *, max_samples: int = 0, batch_size: int = 8, num_workers: int = 0,
                        device: str = "auto", seed: int = 1234, reflectance_scale: float = 10000.0,
                        visualize_n: int = 8, split: str = "test") -> dict[str, Any]:
    """Evaluate the checkpoint over the test split and write all artefacts.

    Args:
        checkpoint: Path to ``best.pt``.
        root: Dataset root.
        output_dir: Audit output directory.
        max_samples: Cap the split (0 = full). A seeded random subset when set.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        device: ``auto`` / ``cuda`` / ``cpu``.
        seed: Random seed (subset + provenance).
        reflectance_scale: DN -> reflectance divisor.
        visualize_n: Number of samples to render enhanced panels for.
        split: Split to evaluate (default ``test``).

    Returns:
        The evaluation report dict (also written to disk).
    """
    from s2train.inference import Predictor  # local import: heavy framework dep

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor.from_checkpoint(checkpoint, device=device)
    composite = bool(getattr(predictor.config.model, "params", {}).get("composite", True))

    loader = build_test_loader(root, split=split, max_samples=max_samples,
                               batch_size=batch_size, num_workers=num_workers,
                               reflectance_scale=reflectance_scale, seed=seed)

    model = ModelState()
    baselines_state: dict[str, BaselineState] = {}
    strata: dict[str, RegionAccumulator] = {}
    skipped_baselines: list[str] = []
    n_samples = 0
    vis_saved = []

    for batch in loader:
        pred = predictor.predict_batch(batch).float()
        target = batch["ground_truth"].float()
        cloud = batch["mask"].float()
        regions = build_regions(target, cloud)
        smap = ssim_map(pred, target)
        b = pred.shape[0]
        n_samples += b

        # ---- Part 3: region reconstruction metrics (model) ----
        for name in _MODEL_REGIONS:
            model.region(name).update(pred, target, regions[name], smap)
        model.ergas_cloud.update(pred, target, regions["cloud"])
        model.ergas_whole.update(pred, target, regions["whole"])
        model.msssim.extend(ms_ssim_per_sample(pred, target))

        # Prediction sanity: unphysical reflectance in the reconstructed region.
        cloud_bp = regions["cloud"].expand_as(pred)
        model.total_cloud_bandpix += cloud_bp.sum().item()
        model.neg_pixels += ((pred < 0.0) & cloud_bp).sum().item()
        model.over_pixels += ((pred > 1.0) & cloud_bp).sum().item()

        # ---- Part 5: per-band metrics (cloud + whole) ----
        for bi in range(13):
            pf, tf = pred[:, bi:bi + 1], target[:, bi:bi + 1]
            for region_name, store in (("cloud", model.band_cloud), ("whole", model.band_whole)):
                p, t = _masked_pair(pf, tf, regions[region_name])
                store[bi].update(p, t)

        # ---- Part 4: vegetation indices (cloud + land) ----
        _update_veg(model.veg_cloud, pred, target, regions["cloud"])
        _update_veg(model.veg_land, pred, target, regions["land"])

        # ---- Part 6: baseline comparisons (cloud region) ----
        baselines, skipped = available_baselines(batch)
        skipped_baselines = skipped
        ndvi_p_model = ndvi(pred)
        for name, fn in baselines.items():
            bpred = fn(batch).float()
            st = baselines_state.setdefault(name, BaselineState())
            st.cloud.update(bpred, target, regions["cloud"], ssim_map(bpred, target))
            ip, it = ndvi(bpred), ndvi(target)
            valid = regions["cloud"] & ip.valid & it.valid
            p, t = _masked_pair(ip.value, it.value, valid)
            st.ndvi_cloud.update(p, t)

        # ---- Part 7: stratified (cloud region) ----
        surfaces = surface_category(target)
        meta = batch.get("metadata") or [{}] * b
        for i in range(b):
            m = meta[i] if i < len(meta) else {}
            labels = sample_strata(difficulty=m.get("difficulty"),
                                   coverage=m.get("applied_cloud_coverage", m.get("cloud_percentage")),
                                   n_references=m.get("n_references"), surface=surfaces[i])
            one = slice(i, i + 1)
            for label in labels.values():
                strata.setdefault(label, RegionAccumulator()).update(
                    pred[one], target[one], regions["cloud"][one], smap[one])

        # ---- Part 2 hook: enhanced visual panels for the first few samples ----
        if len(vis_saved) < visualize_n:
            from .visualize import save_enhanced_panels
            vis_saved += save_enhanced_panels(
                batch, pred, output_dir / "visual_reconstruction",
                start_index=len(vis_saved), limit=visualize_n - len(vis_saved))

    report = _assemble(model, baselines_state, strata, skipped_baselines, composite,
                       n_samples, split)
    report["provenance"] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(Path(checkpoint)),
        "git_commit": _git_commit(),
        "dataset_root": str(root),
        "split": split,
        "n_samples": n_samples,
        "max_samples": max_samples or "all",
        "seed": seed,
        "device": device,
        "reflectance_scale": reflectance_scale,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": predictor.config.model.name,
        "composite_enabled": composite,
    }
    _write_csvs(report, output_dir)
    (output_dir / "test_evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    report["visual_panels"] = vis_saved
    return report


def _assemble(model: ModelState, baselines_state: dict, strata: dict,
              skipped_baselines: list, composite: bool, n_samples: int,
              split: str) -> dict[str, Any]:
    """Turn accumulators into the structured report."""
    region_metrics = {name: model.region(name).result() for name in _MODEL_REGIONS}
    # Flag non-informative clear metrics under hard compositing.
    clear_note = None
    if composite:
        clear_note = ("Clear-region pixel metrics are non-informative: hard "
                      "compositing copies observed clear pixels verbatim, so error "
                      "there is ~0 by construction.")
        for key in ("clear",):
            region_metrics[key]["non_informative"] = True

    per_band = []
    for bi, name in enumerate(BAND_ORDER):
        row = {"band": name}
        row.update({f"cloud_{k}": v for k, v in model.band_cloud[bi].summary().items()})
        row.update({f"whole_{k}": v for k, v in model.band_whole[bi].summary().items()})
        per_band.append(row)

    veg = {}
    for name in VEGETATION_INDICES:
        veg[name] = {"cloud": model.veg_cloud[name].summary(),
                     "land": model.veg_land[name].summary()}

    baseline_rows = []
    model_cloud = model.region("cloud").result()
    for name, st in baselines_state.items():
        r = st.cloud.result()
        baseline_rows.append({
            "method": name,
            "PSNR_cloud": r["psnr_micro"], "SSIM_cloud": r["ssim_micro"],
            "SAM_cloud": r["sam_micro"], "RMSE_cloud": r["rmse_micro"],
            "MAE_cloud": r["mae_micro"],
            "NDVI_MAE_cloud": st.ndvi_cloud.mae, "NDVI_RMSE_cloud": st.ndvi_cloud.rmse,
        })
    # Add the learned model as a row for direct comparison.
    baseline_rows.insert(0, {
        "method": "unet_baseline (learned)",
        "PSNR_cloud": model_cloud["psnr_micro"], "SSIM_cloud": model_cloud["ssim_micro"],
        "SAM_cloud": model_cloud["sam_micro"], "RMSE_cloud": model_cloud["rmse_micro"],
        "MAE_cloud": model_cloud["mae_micro"],
        "NDVI_MAE_cloud": model.veg_cloud["ndvi"].mae,
        "NDVI_RMSE_cloud": model.veg_cloud["ndvi"].rmse,
    })

    strat_rows = []
    for label, acc in sorted(strata.items()):
        r = acc.result()
        axis, _, value = label.partition("=")
        strat_rows.append({"axis": axis, "stratum": value, "n_pixels": r["n_pixels"],
                           "psnr_cloud_micro": r["psnr_micro"], "ssim_cloud_micro": r["ssim_micro"],
                           "sam_cloud_micro": r["sam_micro"], "rmse_cloud_micro": r["rmse_micro"]})

    return {
        "split": split,
        "n_samples": n_samples,
        "primary_metric_note": "Cloud-region metrics are primary; whole-image secondary.",
        "region_metrics": region_metrics,
        "clear_region_note": clear_note,
        "ms_ssim_whole_macro": float(np.nanmean(model.msssim)) if model.msssim else float("nan"),
        "prediction_sanity": {
            "cloud_region_negative_reflectance_fraction":
                model.neg_pixels / model.total_cloud_bandpix if model.total_cloud_bandpix else float("nan"),
            "cloud_region_over_one_reflectance_fraction":
                model.over_pixels / model.total_cloud_bandpix if model.total_cloud_bandpix else float("nan"),
            "note": "Physical reflectance is in [0, 1]; non-zero fractions indicate the "
                    "model emits unphysical values (no output activation / residual head).",
        },
        "ergas": {"cloud": model.ergas_cloud.result(), "whole": model.ergas_whole.result()},
        "per_band_metrics": per_band,
        "vegetation_metrics": veg,
        "baseline_comparison": baseline_rows,
        "skipped_baselines": skipped_baselines,
        "stratified_metrics": strat_rows,
    }


def _write_csvs(report: dict, output_dir: Path) -> None:
    """Write the flat CSV artefacts required by Part 8."""
    # test_metrics.csv - region-wise micro + macro.
    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "n_pixels", "psnr_micro", "psnr_macro", "rmse_micro",
                    "rmse_macro", "mae_micro", "mae_macro", "sam_micro", "sam_macro",
                    "ssim_micro", "ssim_macro", "non_informative"])
        for region, m in report["region_metrics"].items():
            w.writerow([region, m["n_pixels"], m["psnr_micro"], m["psnr_macro"],
                        m["rmse_micro"], m["rmse_macro"], m["mae_micro"], m["mae_macro"],
                        m["sam_micro"], m["sam_macro"], m["ssim_micro"], m["ssim_macro"],
                        m.get("non_informative", False)])

    # per_band_metrics.csv
    with (output_dir / "per_band_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = report["per_band_metrics"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # vegetation_metrics.csv
    with (output_dir / "vegetation_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "region", "n", "mae", "rmse", "bias", "pearson"])
        for name, regions in report["vegetation_metrics"].items():
            for region, m in regions.items():
                w.writerow([name, region, m["n"], m["mae"], m["rmse"], m["bias"], m["pearson"]])

    # baseline_comparison.csv
    with (output_dir / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = report["baseline_comparison"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # stratified_metrics.csv
    with (output_dir / "stratified_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = report["stratified_metrics"]
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
