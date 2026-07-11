"""Datasets and data utilities (registered for config selection).

``synthetic`` wraps the manifest-based synthetic dataset (composes the cloudy
input on the fly); ``fake`` yields random tensors for fast framework smoke
tests. Both return the standard batch contract consumed by the models.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..registry import DATASETS


def apply_d4(sample: dict[str, torch.Tensor], rng: random.Random) -> dict[str, torch.Tensor]:
    """Apply a random D4 (flip/rotate) transform consistently to a sample.

    The same spatial transform is applied to the cloudy input, ground truth,
    mask and all references — physics-preserving for nadir imagery.

    Args:
        sample: A sample dict of tensors.
        rng: A seeded RNG.

    Returns:
        The transformed sample (new tensors; ``metadata`` untouched).
    """
    k = rng.randint(0, 3)
    flip = rng.random() < 0.5

    def tf(x: torch.Tensor) -> torch.Tensor:
        x = torch.rot90(x, k, dims=(-2, -1))
        if flip:
            x = torch.flip(x, dims=(-1,))
        return x.contiguous()

    out = dict(sample)
    for key in ("cloudy", "ground_truth", "mask", "references"):
        if key in out:
            out[key] = tf(out[key])
    return out


@DATASETS.register("synthetic")
class SyntheticDataset(Dataset):
    """The manifest-based synthetic dataset with optional D4 augmentation.

    Delegates loading/composition to
    :class:`s2dataset.synthetic.dataset.S2SyntheticDataset` and applies
    augmentation on the training split.
    """

    def __init__(self, root: str, split: str = "train", *, max_references: int = 4,
                 reflectance_scale: float = 10000.0, augment: bool = False,
                 difficulty: str | None = None, seed: int = 0, **_: Any) -> None:
        """Initialise the dataset.

        Args:
            root: Dataset root directory.
            split: Split name (``train`` / ``validation`` / ``test``).
            max_references: Fixed reference-slot count.
            reflectance_scale: DN -> reflectance divisor.
            augment: Apply D4 augmentation (typically only on train).
            difficulty: Optional curriculum band filter.
            seed: Augmentation RNG seed.
            **_: Ignored extra parameters (config passthrough).
        """
        from s2dataset.synthetic.dataset import S2SyntheticDataset

        self.inner = S2SyntheticDataset(
            root, split=split, maximum_references=max_references,
            reflectance_scale=reflectance_scale, difficulty=difficulty)
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one (optionally augmented) sample."""
        sample = self.inner[index]
        if self.augment:
            sample = apply_d4(sample, self.rng)
        return sample


@DATASETS.register("fake")
class FakeDataset(Dataset):
    """A random-tensor dataset for fast, disk-free framework smoke tests."""

    def __init__(self, root: str = "", split: str = "train", *, length: int = 32,
                 size: int = 64, bands: int = 13, max_references: int = 4,
                 augment: bool = False, seed: int = 0, **_: Any) -> None:
        """Initialise the fake dataset.

        Args:
            root: Ignored.
            split: Split name (affects only the RNG offset).
            length: Number of samples.
            size: Patch side length.
            bands: Number of spectral bands.
            max_references: Reference-slot count.
            augment: Apply D4 augmentation.
            seed: RNG seed.
            **_: Ignored extra parameters.
        """
        self.length = length
        self.size = size
        self.bands = bands
        self.max_references = max_references
        self.augment = augment
        self.rng = random.Random(seed + hash(split) % 1000)
        self.gen = torch.Generator().manual_seed(seed + hash(split) % 1000)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one random sample with a rectangular cloud mask."""
        s, b, r = self.size, self.bands, self.max_references
        gt = torch.rand(b, s, s, generator=self.gen)
        mask = torch.zeros(1, s, s)
        h0, w0 = s // 4, s // 4
        mask[:, h0:h0 + s // 2, w0:w0 + s // 2] = 1.0
        cloudy = gt.clone()
        cloudy[:, mask[0] > 0.5] = torch.rand(1, generator=self.gen).item()
        n_real = self.rng.randint(2, r)
        references = torch.rand(r, b, s, s, generator=self.gen)
        validity = torch.zeros(r)
        validity[:n_real] = 1.0
        references[n_real:] = 0.0
        sample = {"cloudy": cloudy, "mask": mask, "references": references,
                  "reference_validity_mask": validity, "ground_truth": gt,
                  "metadata": {"index": index}}
        if self.augment:
            sample = apply_d4(sample, self.rng)
        return sample


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples into a batch, stacking tensors and listing metadata.

    Args:
        items: A list of sample dicts.

    Returns:
        A batch dict with stacked tensors and a ``metadata`` list.
    """
    batch: dict[str, Any] = {}
    for key in ("cloudy", "mask", "references", "reference_validity_mask", "ground_truth"):
        batch[key] = torch.stack([item[key] for item in items], dim=0)
    batch["metadata"] = [item.get("metadata", {}) for item in items]
    return batch
