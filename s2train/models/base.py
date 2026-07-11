"""Base model interface for cloud-reconstruction networks.

Every model consumes the standard batch produced by the datasets and returns a
reconstructed 13-band image. Keeping this contract fixed is what lets any
architecture be swapped purely from configuration.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class BaseReconstructionModel(nn.Module):
    """Abstract base class for 13-band cloud-reconstruction models.

    Subclasses implement :meth:`reconstruct`. The batch is a dict with:
        * ``cloudy``      -> ``(B, 13, H, W)`` synthetic/observed input
        * ``mask``        -> ``(B, 1, H, W)`` cloud mask (1 = cloud)
        * ``references``  -> ``(B, R, 13, H, W)`` zero-padded references
        * ``reference_validity_mask`` -> ``(B, R)`` (1 = real reference)
        * ``ground_truth``-> ``(B, 13, H, W)`` target (present during training)

    Attributes:
        in_bands: Number of spectral bands (13 for Sentinel-2).
        composite: When ``True`` the output preserves observed clear pixels via
            ``mask * prediction + (1 - mask) * cloudy``.
    """

    def __init__(self, in_bands: int = 13, composite: bool = True) -> None:
        """Initialise the base model.

        Args:
            in_bands: Number of spectral bands.
            composite: Whether to preserve clear pixels in the output.
        """
        super().__init__()
        self.in_bands = in_bands
        self.composite = composite

    def reconstruct(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Produce a raw reconstruction from a batch.

        Args:
            batch: The standard input batch.

        Returns:
            A ``(B, 13, H, W)`` raw reconstruction.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model and optionally composite with observed clear pixels.

        Args:
            batch: The standard input batch.

        Returns:
            The final ``(B, 13, H, W)`` reconstruction.
        """
        prediction = self.reconstruct(batch)
        if self.composite:
            mask = batch["mask"]
            prediction = mask * prediction + (1.0 - mask) * batch["cloudy"]
        return prediction

    @staticmethod
    def reference_mean(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the validity-weighted mean of the reference images.

        Args:
            batch: The standard input batch.

        Returns:
            A ``(B, 13, H, W)`` mean reference (zeros where no valid reference).
        """
        refs = batch["references"]                      # (B, R, 13, H, W)
        validity = batch["reference_validity_mask"]     # (B, R)
        weight = validity.view(validity.shape[0], validity.shape[1], 1, 1, 1)
        total = weight.sum(dim=1).clamp_min(1.0)
        return (refs * weight).sum(dim=1) / total

    def describe(self) -> dict[str, Any]:
        """Return a small description of the model (for the summary)."""
        return {"class": type(self).__name__, "in_bands": self.in_bands,
                "composite": self.composite}
