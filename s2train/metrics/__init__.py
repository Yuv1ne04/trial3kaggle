"""Reconstruction metrics with overall / cloud-region / clear-region variants.

The headline production metric is the *cloud-region* accuracy (the pixels the
model actually reconstructed), so every pixel-wise metric is reported over the
whole image, the cloudy region and the clear region.
"""

from __future__ import annotations

import math

import torch

from ..losses import sam as _sam_angle
from ..losses import ssim_index, ssim_map
from ..registry import METRICS


def _region_mse(pred: torch.Tensor, target: torch.Tensor,
                region: torch.Tensor) -> float:
    """Return mean squared error over a boolean region (bands averaged).

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        region: Boolean region mask ``(B, 1, H, W)``.

    Returns:
        The MSE over the region, or ``nan`` if the region is empty.
    """
    reg = region.expand_as(pred)
    count = reg.sum()
    if count.item() == 0:
        return float("nan")
    return float((((pred - target) ** 2) * reg).sum() / count)


def _region_mae(pred: torch.Tensor, target: torch.Tensor,
                region: torch.Tensor) -> float:
    """Return mean absolute error over a boolean region.

    Args:
        pred: Prediction.
        target: Target.
        region: Boolean region mask.

    Returns:
        The MAE over the region, or ``nan`` if empty.
    """
    reg = region.expand_as(pred)
    count = reg.sum()
    if count.item() == 0:
        return float("nan")
    return float(((pred - target).abs() * reg).sum() / count)


def _psnr_from_mse(mse: float, data_range: float) -> float:
    """Convert an MSE to PSNR (dB)."""
    if math.isnan(mse) or mse <= 0:
        return float("nan") if math.isnan(mse) else 100.0
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def _region_sam(pred: torch.Tensor, target: torch.Tensor,
                region: torch.Tensor, eps: float = 1e-8) -> float:
    """Return mean spectral angle (radians) over a boolean region.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        region: Boolean region mask ``(B, 1, H, W)``.
        eps: Numerical stabiliser.

    Returns:
        The mean SAM over the region, or ``nan`` if empty.
    """
    dot = (pred * target).sum(dim=1, keepdim=True)
    norm = pred.norm(dim=1, keepdim=True) * target.norm(dim=1, keepdim=True) + eps
    cos = torch.clamp(dot / norm, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos)
    count = region.sum()
    if count.item() == 0:
        return float("nan")
    return float((angle * region).sum() / count)


def _ergas(pred: torch.Tensor, target: torch.Tensor, ratio: float = 1.0) -> float:
    """Compute ERGAS (global relative dimensionless error).

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        ratio: High/low resolution ratio (1 for same-resolution reconstruction).

    Returns:
        The ERGAS value (lower is better).
    """
    bands = pred.shape[1]
    total = 0.0
    for b in range(bands):
        rmse = float(((pred[:, b] - target[:, b]) ** 2).mean()) ** 0.5
        mu = float(target[:, b].mean())
        if mu != 0:
            total += (rmse / mu) ** 2
    return 100.0 * ratio * math.sqrt(total / bands)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    names: list[str], *, data_range: float = 1.0) -> dict[str, float]:
    """Compute the requested metrics with region breakdowns.

    Args:
        pred: Prediction ``(B, 13, H, W)``.
        target: Ground truth ``(B, 13, H, W)``.
        mask: Cloud mask ``(B, 1, H, W)`` (1 = cloud).
        names: Metric names to compute (``psnr``/``ssim``/``sam``/``rmse``/
            ``mae``/``ergas``).
        data_range: Dynamic range of the data.

    Returns:
        A dict of metric name -> value (with ``_cloud`` / ``_clear`` suffixes for
        pixel-wise metrics).
    """
    pred = pred.detach().float()
    target = target.detach().float()
    cloud = mask.detach() > 0.5
    clear = ~cloud
    whole = torch.ones_like(cloud)
    regions = {"": whole, "_cloud": cloud, "_clear": clear}
    out: dict[str, float] = {}
    wanted = {n.lower() for n in names}

    for suffix, region in regions.items():
        if {"psnr", "rmse"} & wanted:
            mse = _region_mse(pred, target, region)
            if "rmse" in wanted:
                out[f"rmse{suffix}"] = math.sqrt(mse) if not math.isnan(mse) else float("nan")
            if "psnr" in wanted:
                out[f"psnr{suffix}"] = _psnr_from_mse(mse, data_range)
        if "mae" in wanted:
            out[f"mae{suffix}"] = _region_mae(pred, target, region)
        if "sam" in wanted:
            out[f"sam{suffix}"] = _region_sam(pred, target, region)

    if "ssim" in wanted:
        smap = ssim_map(pred, target, data_range=data_range)
        out["ssim"] = float(smap.mean())
        for suffix, region in (("_cloud", cloud), ("_clear", clear)):
            count = region.sum()
            out[f"ssim{suffix}"] = float((smap * region).sum() / count) \
                if count.item() > 0 else float("nan")
    if "ergas" in wanted:
        out["ergas"] = _ergas(pred, target)
    return out


# Register the individual metric names so they are discoverable/validated.
for _name in ("psnr", "ssim", "sam", "rmse", "mae", "ergas"):
    METRICS.register(_name)(lambda *a, _n=_name, **k: _n)
