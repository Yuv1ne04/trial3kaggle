"""Physically-constrained reference-residual UNet (repaired baseline, v2).

The v1 baseline predicted full reflectance with an unbounded residual head; the
audit found 31% of its cloud-region outputs were negative, and it lost to the
weighted-reference mean. v2 fixes the *output formulation* (not the encoder):

    B      = weighted-reference composite (the physical prior)
    B_safe = clamp(B, eps, 1 - eps)
    delta  = residual_network(cloudy, references, mask)      # 13 channels
    pred   = sigmoid( logit(B_safe) + residual_scale * tanh(delta) )   # in (0, 1)

`sigmoid` guarantees every predicted reflectance is in ``(0, 1)`` -- negative and
over-one outputs are impossible by construction. `tanh` bounds the logit-space
correction to ``[-residual_scale, residual_scale]``, so the model can only nudge
the reference prior, never explode. The final residual head is zero-initialised,
so at step 0 ``delta == 0`` and the model reproduces the weighted-reference mean
exactly -- training starts from a baseline that already beats the old checkpoint.

Clear observed pixels are preserved by the base-class composite. No transformers,
no attention -- this is deliberately the simplest physically-valid repair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import MODELS
from .base import BaseReconstructionModel
from .reference_unet import _Decoder, _Encoder


@MODELS.register("reference_unet_v2")
@MODELS.register("reference_residual_unet_v2")
class ReferenceResidualUNetV2(BaseReconstructionModel):
    """Bounded reference-residual UNet over the weighted-reference prior.

    Attributes:
        residual_scale: Maximum logit-space correction magnitude (tanh-bounded).
        eps: Clamp/logit numerical floor keeping the prior strictly inside (0, 1).
    """

    def __init__(self, base: int = 48, depth: int = 4, in_bands: int = 13,
                 composite: bool = True, weighted_reference: bool = True,
                 use_mask_bottleneck: bool = True, residual_scale: float = 3.0,
                 eps: float = 1e-3) -> None:
        """Initialise the bounded model.

        Args:
            base: Base channel width.
            depth: UNet depth.
            in_bands: Number of spectral bands.
            composite: Preserve observed clear pixels in the output.
            weighted_reference: Validity-weighted reference averaging.
            use_mask_bottleneck: Fuse a downsampled mask at the bottleneck.
            residual_scale: Max logit-space correction (tanh-bounded).
            eps: Numerical floor for the clamp/logit of the prior.
        """
        super().__init__(in_bands=in_bands, composite=composite)
        self.weighted_reference = weighted_reference
        self.use_mask_bottleneck = use_mask_bottleneck
        self.residual_scale = float(residual_scale)
        self.eps = float(eps)

        self.encoder = _Encoder(in_bands, base, depth)
        c = self.encoder.bottleneck_channels
        fuse_in = 2 * c + (1 if use_mask_bottleneck else 0)
        groups = max(1, min(8, c // 4))
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, c, 1), nn.GroupNorm(groups, c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(groups, c), nn.GELU())
        self.decoder = _Decoder(self.encoder.widths, c, in_bands)
        # Zero-init the final residual head so delta == 0 at start => pred == prior.
        nn.init.zeros_(self.decoder.head.weight)
        if self.decoder.head.bias is not None:
            nn.init.zeros_(self.decoder.head.bias)

    def _encode_references(self, references: torch.Tensor,
                           validity: torch.Tensor) -> torch.Tensor:
        """Encode references with shared weights and average valid bottlenecks."""
        batch, refs = references.shape[0], references.shape[1]
        flat = references.reshape(batch * refs, references.shape[2],
                                  references.shape[3], references.shape[4])
        _, bottleneck = self.encoder(flat)
        bottleneck = bottleneck.view(batch, refs, *bottleneck.shape[1:])
        weight = validity.view(batch, refs, 1, 1, 1)
        if not self.weighted_reference:
            weight = (weight > 0).to(weight.dtype)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        return (bottleneck * weight).sum(dim=1) / denom

    def reference_prior(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the weighted-reference composite prior ``B`` (in [0, 1])."""
        return self.reference_mean(batch).clamp(0.0, 1.0)

    def reconstruct(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the bounded reconstruction ``sigmoid(logit(B) + scale*tanh(delta))``."""
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
        delta = self.decoder(fused, skips)

        prior = self.reference_prior(batch)
        b_safe = prior.clamp(self.eps, 1.0 - self.eps)
        b_logit = torch.log(b_safe) - torch.log1p(-b_safe)
        return torch.sigmoid(b_logit + self.residual_scale * torch.tanh(delta))

    def describe(self) -> dict:
        d = super().describe()
        d.update({"residual_scale": self.residual_scale, "eps": self.eps,
                  "output": "sigmoid(logit(B)+scale*tanh(delta)) in (0,1)"})
        return d
