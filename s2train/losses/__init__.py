"""Loss functions for cloud reconstruction (registered for config selection).

The default ``composite`` loss combines a cloud-weighted Charbonnier term with
structural (SSIM), spectral (SAM) and edge (gradient) terms — the combination
recommended for spectrally-faithful, structurally-sharp reconstruction. Every
loss returns ``(total, components)`` so the trainer can log each term.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import LOSSES

LossOutput = tuple[torch.Tensor, dict[str, float]]


def _cloud_weight(mask: torch.Tensor, cloud_weight: float) -> torch.Tensor:
    """Return a per-pixel weight emphasising the cloudy region.

    Args:
        mask: Cloud mask ``(B, 1, H, W)`` (1 = cloud).
        cloud_weight: Relative weight applied to cloudy pixels.

    Returns:
        A ``(B, 1, H, W)`` weight tensor.
    """
    return 1.0 + (cloud_weight - 1.0) * mask


def charbonnier(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                *, cloud_weight: float = 5.0, eps: float = 1e-3) -> torch.Tensor:
    """Cloud-weighted Charbonnier (robust L1/L2) reconstruction loss.

    Args:
        pred: Prediction ``(B, 13, H, W)``.
        target: Ground truth ``(B, 13, H, W)``.
        mask: Cloud mask ``(B, 1, H, W)``.
        cloud_weight: Relative weight on cloudy pixels.
        eps: Charbonnier epsilon.

    Returns:
        A scalar loss.
    """
    weight = _cloud_weight(mask, cloud_weight)
    diff = torch.sqrt((pred - target) ** 2 + eps ** 2)
    return (diff * weight).sum() / (weight.sum() * pred.shape[1] + 1e-8)


def _gaussian_window(channels: int, size: int = 11, sigma: float = 1.5,
                     device=None, dtype=None) -> torch.Tensor:
    """Build a depthwise Gaussian window for SSIM.

    Args:
        channels: Number of channels.
        size: Window size.
        sigma: Gaussian sigma.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A ``(channels, 1, size, size)`` kernel.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(device=device, dtype=dtype)
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim_map(pred: torch.Tensor, target: torch.Tensor, *, window_size: int = 11,
             data_range: float = 1.0) -> torch.Tensor:
    """Compute the channel-averaged per-pixel SSIM map.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        window_size: Gaussian window size.
        data_range: Dynamic range of the data.

    Returns:
        A ``(B, 1, H, W)`` SSIM map (mean over channels), enabling region-masked
        (e.g. cloud-only) SSIM.
    """
    channels = pred.shape[1]
    window = _gaussian_window(channels, window_size, device=pred.device, dtype=pred.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    smap = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return smap.mean(dim=1, keepdim=True)


def ssim_index(pred: torch.Tensor, target: torch.Tensor, *, window_size: int = 11,
               data_range: float = 1.0) -> torch.Tensor:
    """Compute the mean SSIM between two images.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        window_size: Gaussian window size.
        data_range: Dynamic range of the data.

    Returns:
        A scalar mean SSIM in ``[-1, 1]``.
    """
    return ssim_map(pred, target, window_size=window_size, data_range=data_range).mean()


def ms_ssim(pred: torch.Tensor, target: torch.Tensor, *,
            scale_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
            data_range: float = 1.0) -> torch.Tensor:
    """Compute a multi-scale SSIM (mean of SSIM across average-pooled scales).

    A pragmatic MS-SSIM: SSIM is evaluated at successively half-resolution
    versions of the images and combined by ``scale_weights``. This captures
    structure at multiple scales without the full Gaussian-pyramid machinery.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        scale_weights: Per-scale weights (finest first).
        data_range: Dynamic range of the data.

    Returns:
        A scalar MS-SSIM in ``[-1, 1]``.
    """
    cur_p, cur_t = pred, target
    total = 0.0
    for i, weight in enumerate(scale_weights):
        total = total + weight * ssim_index(cur_p, cur_t, data_range=data_range)
        if i < len(scale_weights) - 1 and min(cur_p.shape[-2:]) >= 22:
            cur_p = F.avg_pool2d(cur_p, 2)
            cur_t = F.avg_pool2d(cur_t, 2)
    return total / sum(scale_weights)


def sam(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean Spectral Angle Mapper (radians) between predicted/true spectra.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.
        eps: Numerical stabiliser.

    Returns:
        A scalar mean spectral angle in radians.
    """
    dot = (pred * target).sum(dim=1)
    norm = pred.norm(dim=1) * target.norm(dim=1) + eps
    cos = torch.clamp(dot / norm, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between the spatial gradients of prediction and target.

    Args:
        pred: Prediction ``(B, C, H, W)``.
        target: Target ``(B, C, H, W)``.

    Returns:
        A scalar edge/gradient loss.
    """
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


@LOSSES.register("composite")
class CompositeLoss(nn.Module):
    """Weighted combination of Charbonnier + (1-SSIM) + SAM + gradient losses.

    Attributes:
        weights: Per-term weights.
        cloud_weight: Relative weight on cloudy pixels for the Charbonnier term.
    """

    def __init__(self, charbonnier_weight: float = 1.0, ssim_weight: float = 0.5,
                 ms_ssim_weight: float = 0.0, sam_weight: float = 0.3,
                 gradient_weight: float = 0.2, cloud_weight: float = 5.0) -> None:
        """Initialise the composite loss.

        Args:
            charbonnier_weight: Weight of the Charbonnier term.
            ssim_weight: Weight of the single-scale ``1 - SSIM`` term.
            ms_ssim_weight: Weight of the ``1 - MS-SSIM`` term (0 disables it).
            sam_weight: Weight of the SAM term.
            gradient_weight: Weight of the gradient/edge term.
            cloud_weight: Relative weight on cloudy pixels.
        """
        super().__init__()
        self.weights = {"charbonnier": charbonnier_weight, "ssim": ssim_weight,
                        "ms_ssim": ms_ssim_weight, "sam": sam_weight,
                        "gradient": gradient_weight}
        self.cloud_weight = cloud_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> LossOutput:
        """Compute the total loss and its components.

        Args:
            pred: Prediction ``(B, 13, H, W)``.
            target: Ground truth ``(B, 13, H, W)``.
            mask: Cloud mask ``(B, 1, H, W)``.

        Returns:
            ``(total, components)`` where ``components`` maps active term -> value.
        """
        available = {
            "charbonnier": lambda: charbonnier(pred, target, mask,
                                               cloud_weight=self.cloud_weight),
            "ssim": lambda: 1.0 - ssim_index(pred, target),
            "ms_ssim": lambda: 1.0 - ms_ssim(pred, target),
            "sam": lambda: sam(pred, target),
            "gradient": lambda: gradient_l1(pred, target),
        }
        # Only compute terms with a non-zero weight (avoids wasted MS-SSIM cost).
        terms = {k: fn() for k, fn in available.items() if self.weights.get(k, 0.0) > 0}
        total = sum(self.weights[k] * v for k, v in terms.items())
        components = {k: float(v.detach()) for k, v in terms.items()}
        components["total"] = float(total.detach())
        return total, components


@LOSSES.register("charbonnier")
class CharbonnierLoss(nn.Module):
    """Cloud-weighted Charbonnier loss on its own (baseline)."""

    def __init__(self, cloud_weight: float = 5.0) -> None:
        """Initialise the loss.

        Args:
            cloud_weight: Relative weight on cloudy pixels.
        """
        super().__init__()
        self.cloud_weight = cloud_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> LossOutput:
        """Compute the Charbonnier loss.

        Args:
            pred: Prediction.
            target: Ground truth.
            mask: Cloud mask.

        Returns:
            ``(total, {"total": value})``.
        """
        total = charbonnier(pred, target, mask, cloud_weight=self.cloud_weight)
        return total, {"total": float(total.detach())}
