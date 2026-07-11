"""Synthetic supervision pipeline for Sentinel-2 cloud reconstruction.

Under a real cloud there is no ground truth, so supervised training pairs are
manufactured by taking **clear** observed patches as ground truth and
transplanting **real cloud masks** (and, optionally, real cloud reflectance)
sampled from other Mauritius acquisitions onto them. Each clear patch can yield
several synthetic variants with configurable curriculum difficulty (easy /
medium / hard cloud coverage).

Storage-efficient layout (each patch/cloud stored once; samples are tiny JSON)::

    synthetic_dataset/
        patch_library/{size}/{YYYYMMDD}/patch_{cell}.npz       # 13-band image
        cloud_tile_library/{size}/{YYYYMMDD}/patch_{cell}.npz   # mask (+cloud)
        samples/{train,validation,test}/sample_000001.json      # references only

The PyTorch dataset composes the cloudy input at load time, so nothing corrupted
is materialised. No neural network is implemented here; this module only
produces data.
"""

from __future__ import annotations

from .builder import SyntheticSupervisionBuilder
from .config import SyntheticConfig
from .dataset import S2SyntheticDataset
from .models import GenOutcome, GroundTruthPatch, MaskEntry, SyntheticManifest

__all__ = [
    "SyntheticSupervisionBuilder",
    "SyntheticConfig",
    "S2SyntheticDataset",
    "GroundTruthPatch",
    "MaskEntry",
    "SyntheticManifest",
    "GenOutcome",
]
