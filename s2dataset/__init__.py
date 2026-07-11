"""s2dataset — AI training-dataset builder for Sentinel-2 cloud reconstruction.

Turns the stacked GeoTIFFs, cloud masks and the reference database into
patch-level training samples in two formats (georeferenced GeoTIFF folders and
PyTorch-ready ``.npz`` files), with a leakage-free temporal train/val/test
split, configurable patch extraction and filtering, parallel processing, and
automatic resume.

This module performs *no* machine learning; it only assembles datasets.

Typical usage
-------------
>>> from s2dataset import DatasetConfig, DatasetBuilder
>>> config = DatasetConfig.from_yaml("dataset_config.yaml")
>>> stats = DatasetBuilder(config).run()
>>> stats["total_samples"]
4821
"""

from __future__ import annotations

from .config import DatasetConfig, PatchScale
from .builder import DatasetBuilder
from .models import PatchSample, SampleSpec, AlignmentReport

__all__ = [
    "DatasetConfig",
    "PatchScale",
    "DatasetBuilder",
    "PatchSample",
    "SampleSpec",
    "AlignmentReport",
]

__version__ = "1.0.0"
