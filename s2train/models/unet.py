"""Baseline UNet architectures for cloud reconstruction.

Two real, lightweight models are provided so the framework runs end-to-end:
``unet`` (single-image) and ``ref_unet`` (reference-conditioned via early
fusion of the validity-weighted reference mean). Heavier architectures
(TCR-Net, Restormer, SwinIR) register the same way and reuse the same base
contract.
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import MODELS
from .base import BaseReconstructionModel


class _DoubleConv(nn.Module):
    """Two 3x3 conv + GroupNorm + GELU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialise the block.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
        """
        super().__init__()
        groups = max(1, min(8, out_ch // 4))
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double-conv block."""
        return self.block(x)


class _UNetBackbone(nn.Module):
    """A small, configurable UNet mapping ``in_ch`` -> 13 residual channels."""

    def __init__(self, in_ch: int, base: int = 48, depth: int = 4,
                 out_ch: int = 13) -> None:
        """Initialise the backbone.

        Args:
            in_ch: Number of input channels.
            base: Base channel width.
            depth: Number of down/up stages.
            out_ch: Number of output channels.
        """
        super().__init__()
        widths = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.downs.append(_DoubleConv(prev, w))
            self.pools.append(nn.MaxPool2d(2))
            prev = w
        self.bottleneck = _DoubleConv(prev, prev * 2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = prev * 2
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(prev, w, 2, stride=2))
            self.up_convs.append(_DoubleConv(w * 2, w))
            prev = w
        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the UNet, returning ``(B, out_ch, H, W)``."""
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = up_conv(torch.cat([x, skip], dim=1))
        return self.head(x)


@MODELS.register("unet")
class UNet(BaseReconstructionModel):
    """Single-image UNet baseline that predicts a residual to the cloudy input.

    The residual formulation (output = cloudy + UNet(cloudy[+mask])) makes the
    clear regions near-identity and focuses capacity on the cloudy pixels.
    """

    def __init__(self, base: int = 48, depth: int = 4, use_mask: bool = True,
                 in_bands: int = 13, composite: bool = True) -> None:
        """Initialise the UNet.

        Args:
            base: Base channel width.
            depth: UNet depth.
            use_mask: Concatenate the cloud mask to the input.
            in_bands: Number of spectral bands.
            composite: Preserve observed clear pixels in the output.
        """
        super().__init__(in_bands=in_bands, composite=composite)
        self.use_mask = use_mask
        in_ch = in_bands + (1 if use_mask else 0)
        self.backbone = _UNetBackbone(in_ch, base=base, depth=depth, out_ch=in_bands)

    def reconstruct(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict ``cloudy + residual``.

        Args:
            batch: The standard input batch.

        Returns:
            The raw reconstruction ``(B, 13, H, W)``.
        """
        cloudy = batch["cloudy"]
        parts = [cloudy]
        if self.use_mask:
            parts.append(batch["mask"])
        residual = self.backbone(torch.cat(parts, dim=1))
        return cloudy + residual


@MODELS.register("ref_unet")
class ReferenceUNet(BaseReconstructionModel):
    """Reference-conditioned UNet: early-fuses the mean reference image.

    Input channels = cloudy (13) + mask (1) + validity-weighted reference mean
    (13). This is a simple but real demonstration of exploiting the 2-4
    historical references through the same base contract.
    """

    def __init__(self, base: int = 48, depth: int = 4, in_bands: int = 13,
                 composite: bool = True) -> None:
        """Initialise the reference-conditioned UNet.

        Args:
            base: Base channel width.
            depth: UNet depth.
            in_bands: Number of spectral bands.
            composite: Preserve observed clear pixels in the output.
        """
        super().__init__(in_bands=in_bands, composite=composite)
        in_ch = in_bands + 1 + in_bands
        self.backbone = _UNetBackbone(in_ch, base=base, depth=depth, out_ch=in_bands)

    def reconstruct(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict ``ref_mean + residual`` (residual to the reference prior).

        Args:
            batch: The standard input batch.

        Returns:
            The raw reconstruction ``(B, 13, H, W)``.
        """
        cloudy = batch["cloudy"]
        ref_mean = self.reference_mean(batch)
        stacked = torch.cat([cloudy, batch["mask"], ref_mean], dim=1)
        residual = self.backbone(stacked)
        return ref_mean + residual
