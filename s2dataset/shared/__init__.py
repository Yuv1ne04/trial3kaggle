"""Shared-reference dataset architecture for Sentinel-2 cloud reconstruction.

Because every acquisition is resampled onto the same 10 m grid, a patch at grid
cell ``(size, row, col)`` covers the same ground on every date and is therefore
*target-independent*. This package stores each such patch exactly once in a
shared ``patch_library`` and expresses training samples as lightweight JSON that
*reference* library patches by path — eliminating the reference duplication of
the original per-sample NPZ design.

Layout::

    dataset/
        target_library/{size}/{YYYYMMDD}/patch_{cell:06d}.npz     # 13-band image
        reference_library/{size}/{YYYYMMDD}/patch_{cell:06d}.npz   # 13-band image
        mask_library/{size}/{YYYYMMDD}/patch_{cell:06d}.npz        # cloud mask
        samples/{train,validation,test}/sample_{id:06d}.json       # references only

No machine learning is performed here.
"""

from __future__ import annotations

from .builder import SharedDatasetBuilder
from .dataset import S2SharedReconstructionDataset
from .models import PatchKey, SampleManifest

__all__ = [
    "SharedDatasetBuilder",
    "S2SharedReconstructionDataset",
    "PatchKey",
    "SampleManifest",
]
