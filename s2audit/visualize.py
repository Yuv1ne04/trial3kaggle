"""Enhanced reconstruction visualisation (Part 2).

Every sample panel shows the eight required views plus NDVI (ground truth,
prediction, absolute error), and renders the RGB views under *both* a fixed
physical-reflectance stretch and a robust 2-98 percentile stretch, so dark
agricultural scenes are actually inspectable. Only the *display* is changed;
the physical reflectance tensors are never modified. Matplotlib is imported
lazily so importing this module never requires it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import BAND_INDEX

_RED, _GREEN, _BLUE = BAND_INDEX["B04"], BAND_INDEX["B03"], BAND_INDEX["B02"]
_NIR = BAND_INDEX["B08"]

#: Documented fixed true-colour display range (surface reflectance).
FIXED_REFLECTANCE_MAX = 0.30


def _rgb(img: np.ndarray, mode: str) -> np.ndarray:
    """Return an HxWx3 RGB image under a fixed or percentile stretch."""
    rgb = np.stack([img[_RED], img[_GREEN], img[_BLUE]], axis=-1)
    if mode == "fixed":
        return np.clip(rgb / FIXED_REFLECTANCE_MAX, 0, 1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _ndvi(img: np.ndarray) -> np.ndarray:
    denom = img[_NIR] + img[_RED]
    return np.where(np.abs(denom) > 2e-3, (img[_NIR] - img[_RED]) / (denom + 1e-6), 0.0)


def _weighted_reference_mean(refs: np.ndarray, validity: np.ndarray) -> np.ndarray:
    w = validity.reshape(-1, 1, 1, 1)
    denom = max(w.sum(), 1e-6)
    return (refs * w).sum(axis=0) / denom


def save_enhanced_panels(batch: dict, pred: torch.Tensor, out_dir: Path | str,
                         *, start_index: int = 0, limit: int = 8) -> list[str]:
    """Render enhanced panels for up to ``limit`` samples of a batch.

    Args:
        batch: The evaluation batch (tensors + metadata).
        pred: Model prediction ``(B, 13, H, W)`` (composited).
        out_dir: Output directory.
        start_index: Running index for filenames / cap accounting.
        limit: Maximum panels to render from this batch.

    Returns:
        A list of written file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cloudy = batch["cloudy"].detach().cpu().numpy()
    gt = batch["ground_truth"].detach().cpu().numpy()
    mask = batch["mask"].detach().cpu().numpy()
    refs = batch["references"].detach().cpu().numpy()
    validity = batch["reference_validity_mask"].detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()
    written = []

    n = min(limit, pr.shape[0])
    for i in range(n):
        wref = _weighted_reference_mean(refs[i], validity[i])
        cmask = mask[i, 0]
        signed = (pr[i] - gt[i]).mean(axis=0)
        abserr = np.abs(pr[i] - gt[i]).mean(axis=0)
        cloud_err = abserr * (cmask > 0.5)
        ndvi_gt, ndvi_pr = _ndvi(gt[i]), _ndvi(pr[i])
        ndvi_err = np.abs(ndvi_pr - ndvi_gt)

        fig, ax = plt.subplots(4, 5, figsize=(16, 13))
        for a in ax.ravel():
            a.axis("off")

        def show(r, c, data, title, **kw):
            ax[r, c].imshow(data, **kw)
            ax[r, c].set_title(title, fontsize=8)

        # Row 0 - fixed reflectance stretch.
        show(0, 0, _rgb(cloudy[i], "fixed"), "Synthetic input [fixed]")
        show(0, 1, _rgb(gt[i], "fixed"), "Ground truth [fixed]")
        show(0, 2, _rgb(pr[i], "fixed"), "Prediction [fixed]")
        show(0, 3, _rgb(wref, "fixed"), "Weighted ref mean [fixed]")
        show(0, 4, cmask, "Cloud mask", cmap="gray", vmin=0, vmax=1)
        # Row 1 - percentile stretch.
        show(1, 0, _rgb(cloudy[i], "pct"), "Synthetic input [2-98%]")
        show(1, 1, _rgb(gt[i], "pct"), "Ground truth [2-98%]")
        show(1, 2, _rgb(pr[i], "pct"), "Prediction [2-98%]")
        show(1, 3, _rgb(wref, "pct"), "Weighted ref mean [2-98%]")
        # Row 2 - error analysis.
        show(2, 0, signed, "Signed difference", cmap="RdBu", vmin=-0.1, vmax=0.1)
        show(2, 1, abserr, "Absolute error", cmap="magma", vmin=0, vmax=0.1)
        show(2, 2, cloud_err, "Cloud-only error", cmap="magma", vmin=0, vmax=0.1)
        # Row 3 - NDVI (operational).
        show(3, 0, ndvi_gt, "NDVI ground truth", cmap="RdYlGn", vmin=-0.2, vmax=0.9)
        show(3, 1, ndvi_pr, "NDVI prediction", cmap="RdYlGn", vmin=-0.2, vmax=0.9)
        show(3, 2, ndvi_err, "NDVI absolute error", cmap="magma", vmin=0, vmax=0.3)

        idx = start_index + i
        meta = (batch.get("metadata") or [{}])[i] if i < len(batch.get("metadata") or []) else {}
        fig.suptitle(f"Sample {idx} | date={meta.get('target_date','?')} "
                     f"| difficulty={meta.get('difficulty','?')} "
                     f"| coverage={meta.get('applied_cloud_coverage','?')}", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"sample_{idx:04d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written
