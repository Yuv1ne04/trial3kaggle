"""Thin dataloader builder for the audit (decoupled from the training config).

Wraps the existing :class:`s2dataset.synthetic.dataset.S2SyntheticDataset` (via
the ``s2train`` registered wrapper when available) with a seeded random subset
cap and the standard collate. Kept separate so the audit does not depend on an
``ExperimentConfig`` and can point at any dataset root.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def build_test_loader(root: Path | str, *, split: str = "test", max_samples: int = 0,
                      batch_size: int = 8, num_workers: int = 0, max_references: int = 4,
                      reflectance_scale: float = 10000.0, seed: int = 1234) -> DataLoader:
    """Build a non-shuffled evaluation dataloader over a split.

    Args:
        root: Dataset root.
        split: Split name.
        max_samples: Seeded random subset size (0 = full split).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        max_references: Reference slot count.
        reflectance_scale: DN -> reflectance divisor.
        seed: Subset seed.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    from s2train.datasets import SyntheticDataset, collate_batch

    dataset = SyntheticDataset(str(root), split=split, max_references=max_references,
                               reflectance_scale=reflectance_scale, augment=False, seed=seed)
    if max_samples and len(dataset) > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(dataset), generator=g)[:max_samples].tolist()
        dataset = Subset(dataset, idx)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_batch,
                      pin_memory=torch.cuda.is_available())
