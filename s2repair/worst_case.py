"""Part 2 - worst-case failure analysis of the existing checkpoint.

Streams the test split, ranks samples by cloud-region squared error, keeps the
50 worst, records their metadata + failure signatures, renders diagnostic panels
(raw vs bounded prediction vs weighted-reference composite), and correlates the
catastrophic errors with difficulty / coverage / references / surface / bands.
The checkpoint is never modified.
"""

from __future__ import annotations

import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from s2audit.baselines import weighted_reference_mean
from s2audit.datasets_compat import build_test_loader
from s2audit.indices import surface_category
from s2audit import BAND_ORDER

_NIR, _RED, _GREEN, _BLUE = 7, 3, 2, 1


def _ndvi(x: np.ndarray) -> np.ndarray:
    d = x[_NIR] + x[_RED]
    return np.where(np.abs(d) > 2e-3, (x[_NIR] - x[_RED]) / (d + 1e-6), 0.0)


def run_worst_case(checkpoint: str | Path, root: str | Path, output_dir: str | Path, *,
                   max_samples: int = 240, top_k: int = 50, batch_size: int = 8,
                   device: str = "auto", seed: int = 1234,
                   reflectance_scale: float = 10000.0, render: int = 20) -> dict[str, Any]:
    """Rank and analyse the worst-case samples; write CSV + panels.

    Args:
        checkpoint: Path to ``best.pt`` (unchanged).
        root: Dataset root.
        output_dir: Output directory.
        max_samples: Test samples to scan (0 = full).
        top_k: Number of worst samples to keep.
        batch_size: Inference batch size.
        device: Device string.
        seed: Random seed.
        reflectance_scale: DN -> reflectance divisor.
        render: Number of worst panels to render.

    Returns:
        A summary dict with correlation tables.
    """
    from s2train.inference import Predictor

    output_dir = Path(output_dir)
    (output_dir / "worst_case_panels").mkdir(parents=True, exist_ok=True)
    predictor = Predictor.from_checkpoint(checkpoint, device=device)
    loader = build_test_loader(root, split="test", max_samples=max_samples,
                               batch_size=batch_size, seed=seed,
                               reflectance_scale=reflectance_scale)

    heap: list = []          # min-heap of (sse, counter, record)
    counter = 0
    corr = defaultdict(lambda: {"count": 0, "sse": 0.0})

    for batch in loader:
        raw = predictor.predict_batch(batch).float()
        target = batch["ground_truth"].float()
        mask = batch["mask"].float()
        cloud = mask > 0.5
        base = weighted_reference_mean(batch).float()
        surfaces = surface_category(target)
        meta = batch.get("metadata") or [{}] * raw.shape[0]

        for i in range(raw.shape[0]):
            cm = cloud[i]
            n_cloud = float(cm.sum()) * raw.shape[1]
            if n_cloud == 0:
                continue
            err = ((raw[i] - target[i]) ** 2) * cm
            sse = float(err.sum())
            m = meta[i] if i < len(meta) else {}
            per_band_rmse = [float(torch.sqrt((((raw[i, b] - target[i, b]) ** 2) * cm[0]).sum()
                                              / cm[0].sum().clamp_min(1))) for b in range(13)]
            ndvi_err = float(np.abs(_ndvi(raw[i].numpy()) - _ndvi(target[i].numpy()))[cm[0].numpy()].mean())
            rec = {
                "sample_id": m.get("sample_id", ""), "target_date": m.get("target_date", ""),
                "cloud_coverage": m.get("applied_cloud_coverage", m.get("cloud_percentage")),
                "difficulty": m.get("difficulty", ""), "reference_count": m.get("n_references"),
                "reference_dates": ";".join(m.get("reference_dates", []) or []),
                "prediction_min": float(raw[i].min()), "prediction_max": float(raw[i].max()),
                "negative_fraction": float(((raw[i] < 0) & cm.expand_as(raw[i])).sum() / (n_cloud)),
                "per_band_rmse": per_band_rmse, "ndvi_error": ndvi_err,
                "surface_category": surfaces[i], "cloud_sse": sse,
            }
            # Correlation accumulation over all scanned samples.
            for axis, key in (("difficulty", rec["difficulty"]), ("surface", rec["surface_category"]),
                              ("date", rec["target_date"]), ("reference_count", str(rec["reference_count"]))):
                corr[f"{axis}={key}"]["count"] += 1
                corr[f"{axis}={key}"]["sse"] += sse
            counter += 1
            payload = None
            if len(heap) < render or (heap and sse > heap[0][0]):
                payload = {"cloudy": batch["cloudy"][i].numpy(), "gt": target[i].numpy(),
                           "raw": raw[i].numpy(), "base": base[i].numpy(), "mask": mask[i, 0].numpy()}
            item = (sse, counter, rec, payload)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif sse > heap[0][0]:
                heapq.heapreplace(heap, item)

    worst = sorted(heap, key=lambda x: -x[0])
    # Write CSV.
    csv_path = output_dir / "worst_case_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["rank", "sample_id", "target_date", "cloud_coverage", "difficulty",
                "reference_count", "reference_dates", "prediction_min", "prediction_max",
                "negative_fraction", "ndvi_error", "surface_category", "cloud_sse"] + \
               [f"rmse_{b}" for b in BAND_ORDER]
        w = csv.writer(fh)
        w.writerow(cols)
        for rank, (sse, _, rec, _) in enumerate(worst, 1):
            w.writerow([rank, rec["sample_id"], rec["target_date"], rec["cloud_coverage"],
                        rec["difficulty"], rec["reference_count"], rec["reference_dates"],
                        round(rec["prediction_min"], 4), round(rec["prediction_max"], 4),
                        round(rec["negative_fraction"], 4), round(rec["ndvi_error"], 4),
                        rec["surface_category"], round(rec["cloud_sse"], 2)]
                       + [round(v, 5) for v in rec["per_band_rmse"]])

    # Render panels for the worst `render`.
    rendered = _render_panels(worst[:render], output_dir / "worst_case_panels")

    # Correlation summary: mean SSE per stratum.
    corr_summary = {k: {"count": v["count"], "mean_sse": v["sse"] / max(1, v["count"])}
                    for k, v in corr.items()}
    band_rmse_mean = np.mean([r["per_band_rmse"] for _, _, r, _ in worst], axis=0).tolist()
    summary = {
        "n_scanned": counter, "top_k": len(worst),
        "worst_mean_negative_fraction": float(np.mean([r["negative_fraction"] for _, _, r, _ in worst])),
        "worst_by_difficulty": _tally(worst, "difficulty"),
        "worst_by_surface": _tally(worst, "surface_category"),
        "worst_by_reference_count": _tally(worst, "reference_count"),
        "worst_band_rmse_mean": dict(zip(BAND_ORDER, [round(v, 4) for v in band_rmse_mean])),
        "stratum_mean_sse": corr_summary,
        "panels_rendered": rendered,
        "csv": str(csv_path),
    }
    (output_dir / "worst_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _tally(worst: list, field: str) -> dict:
    out: dict = defaultdict(int)
    for _, _, rec, _ in worst:
        out[str(rec.get(field))] += 1
    return dict(out)


def _rgb(x: np.ndarray) -> np.ndarray:
    rgb = np.stack([x[_RED], x[_GREEN], x[_BLUE]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _render_panels(items: list, out_dir: Path) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return 0
    made = 0
    for rank, (sse, _, rec, payload) in enumerate(items, 1):
        if payload is None:
            continue
        raw, gt, base = payload["raw"], payload["gt"], payload["base"]
        mask = payload["mask"]
        abserr = np.abs(raw - gt).mean(0)
        ndvi_gt, ndvi_pr = _ndvi(gt), _ndvi(raw)
        fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
        for a in ax.ravel():
            a.axis("off")
        panels = [
            (_rgb(payload["cloudy"]), "Synthetic input"), (_rgb(gt), "Ground truth"),
            (_rgb(raw), "Raw prediction"), (_rgb(np.clip(raw, 0, 1)), "Bounded prediction"),
            (_rgb(base), "Weighted-ref composite"),
        ]
        for c, (img, title) in enumerate(panels):
            ax[0, c].imshow(img); ax[0, c].set_title(title, fontsize=8)
        ax[1, 0].imshow(mask, cmap="gray"); ax[1, 0].set_title("Cloud mask", fontsize=8)
        ax[1, 1].imshow(abserr, cmap="magma", vmin=0, vmax=0.3); ax[1, 1].set_title("Absolute error", fontsize=8)
        ax[1, 2].imshow(ndvi_gt, cmap="RdYlGn", vmin=-0.2, vmax=0.9); ax[1, 2].set_title("NDVI GT", fontsize=8)
        ax[1, 3].imshow(ndvi_pr, cmap="RdYlGn", vmin=-0.2, vmax=0.9); ax[1, 3].set_title("NDVI pred", fontsize=8)
        ax[1, 4].imshow(np.abs(ndvi_pr - ndvi_gt), cmap="magma", vmin=0, vmax=0.6); ax[1, 4].set_title("NDVI error", fontsize=8)
        fig.suptitle(f"Rank {rank} | {rec['sample_id']} | {rec['target_date']} | "
                     f"{rec['difficulty']} | pred[{rec['prediction_min']:.2f},{rec['prediction_max']:.2f}] | "
                     f"neg {rec['negative_fraction']:.2f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"worst_{rank:02d}_{rec['sample_id']}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        made += 1
    return made
