"""Baseline reference-conditioned UNet (the scientific baseline before TCR-Net).

Design (deliberately simple — no transformers/attention/diffusion):

    target ─► shared encoder ─► target bottleneck ┐
    refs  ─► shared encoder ─► per-ref bottleneck  │
                              (mask invalid, average)│
                                     ▼               ▼
                              fuse (concat + conv, with mask) ─► decoder ─► residual
    output = cloudy + residual  →  composite with the cloud mask

The same CNN encoder weights encode the target and every reference (Siamese).
Invalid references are excluded via the validity mask before a (weighted)
average of their bottleneck features. Only the residual is predicted, and the
cloud mask preserves clear pixels in the final output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import MODELS
from .base import BaseReconstructionModel
from .unet import _DoubleConv


class _Encoder(nn.Module):
    """A UNet encoder returning multi-scale skips and the bottleneck feature."""

    def __init__(self, in_ch: int, base: int, depth: int) -> None:
        """Initialise the encoder.

        Args:
            in_ch: Number of input channels.
            base: Base channel width.
            depth: Number of down stages.
        """
        super().__init__()
        self.widths = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for width in self.widths:
            self.downs.append(_DoubleConv(prev, width))
            self.pools.append(nn.MaxPool2d(2))
            prev = width
        self.bottleneck = _DoubleConv(prev, prev * 2)
        self.bottleneck_channels = prev * 2

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode ``x``.

        Args:
            x: Input ``(B, in_ch, H, W)``.

        Returns:
            ``(skips, bottleneck)`` where ``skips`` is a list of per-stage
            features and ``bottleneck`` is ``(B, C, H/2**depth, W/2**depth)``.
        """
        skips: list[torch.Tensor] = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        return skips, self.bottleneck(x)


class _Decoder(nn.Module):
    """A UNet decoder consuming a bottleneck feature and target skips."""

    def __init__(self, widths: list[int], bottleneck_channels: int, out_ch: int) -> None:
        """Initialise the decoder.

        Args:
            widths: Encoder stage widths (ascending).
            bottleneck_channels: Channels of the (fused) bottleneck feature.
            out_ch: Number of output channels.
        """
        super().__init__()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = bottleneck_channels
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(prev, width, 2, stride=2))
            self.up_convs.append(_DoubleConv(width * 2, width))
            prev = width
        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, bottleneck: torch.Tensor,
                skips: list[torch.Tensor]) -> torch.Tensor:
        """Decode to the output residual.

        Args:
            bottleneck: The (fused) bottleneck feature.
            skips: The target encoder skips (ascending resolution).

        Returns:
            The output ``(B, out_ch, H, W)``.
        """
        x = bottleneck
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = up_conv(torch.cat([x, skip], dim=1))
        return self.head(x)


@MODELS.register("unet_baseline")
@MODELS.register("reference_unet")
class ReferenceUNetBaseline(BaseReconstructionModel):
    """Baseline UNet with shared-encoder, bottleneck reference fusion.

    Attributes:
        weighted_reference: When ``True`` references are averaged with their
            validity weights; padded/invalid references contribute nothing.
        use_mask_bottleneck: When ``True`` a downsampled cloud mask is fused at
            the bottleneck so the decoder knows where reconstruction is needed.
    """

    def __init__(self, base: int = 48, depth: int = 4, in_bands: int = 13,
                 composite: bool = True, weighted_reference: bool = True,
                 use_mask_bottleneck: bool = True) -> None:
        """Initialise the baseline model.

        Args:
            base: Base channel width.
            depth: UNet depth.
            in_bands: Number of spectral bands.
            composite: Preserve observed clear pixels in the output.
            weighted_reference: Validity-weighted reference averaging.
            use_mask_bottleneck: Fuse a downsampled mask at the bottleneck.
        """
        super().__init__(in_bands=in_bands, composite=composite)
        self.weighted_reference = weighted_reference
        self.use_mask_bottleneck = use_mask_bottleneck
        self.encoder = _Encoder(in_bands, base, depth)
        c = self.encoder.bottleneck_channels
        fuse_in = 2 * c + (1 if use_mask_bottleneck else 0)
        groups = max(1, min(8, c // 4))
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, c, 1), nn.GroupNorm(groups, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(groups, c), nn.GELU())
        self.decoder = _Decoder(self.encoder.widths, c, in_bands)

    def _encode_references(self, references: torch.Tensor,
                           validity: torch.Tensor) -> torch.Tensor:
        """Encode all references (shared weights) and average valid bottlenecks.

        Args:
            references: References ``(B, R, 13, H, W)``.
            validity: Validity mask ``(B, R)`` (1 = real reference).

        Returns:
            The aggregated reference bottleneck ``(B, C, h, w)``.
        """
        batch, refs = references.shape[0], references.shape[1]
        flat = references.reshape(batch * refs, references.shape[2],
                                  references.shape[3], references.shape[4])
        _, bottleneck = self.encoder(flat)                       # (B*R, C, h, w)
        bottleneck = bottleneck.view(batch, refs, *bottleneck.shape[1:])
        weight = validity.view(batch, refs, 1, 1, 1)
        if not self.weighted_reference:
            weight = (weight > 0).to(weight.dtype)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        return (bottleneck * weight).sum(dim=1) / denom

    def reconstruct(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict ``cloudy + residual`` using fused reference features.

        Args:
            batch: The standard input batch.

        Returns:
            The raw reconstruction ``(B, 13, H, W)``.
        """
        cloudy = batch["cloudy"]
        skips, target_bottleneck = self.encoder(cloudy)
        reference_bottleneck = self._encode_references(
            batch["references"], batch["reference_validity_mask"])

        features = [target_bottleneck, reference_bottleneck]
        if self.use_mask_bottleneck:
            mask_ds = F.interpolate(batch["mask"], size=target_bottleneck.shape[-2:],
                                    mode="bilinear", align_corners=False)
            features.append(mask_ds)
        fused = self.fuse(torch.cat(features, dim=1))
        residual = self.decoder(fused, skips)
        return cloudy + residual
