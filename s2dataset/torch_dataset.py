"""Reference PyTorch ``Dataset`` for consuming the generated NPZ samples.

This module is import-safe without PyTorch installed: the ``torch`` import is
deferred to instantiation, so the rest of ``s2dataset`` (which must not depend
on torch) can be imported freely. Copy this file into the training repo, or
import it there, to load samples produced by the builder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class S2ReconstructionNPZDataset:
    """Loads NPZ samples (target, mask, references, metadata) for training.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> ds = S2ReconstructionNPZDataset("dataset/train_npz")
        >>> loader = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)

    Each item is a dict of tensors:
        * ``target``     -> float32 ``(13, H, W)`` reflectance in ``[0, 1]``
        * ``mask``       -> float32 ``(1, H, W)`` (1 = cloud)
        * ``references`` -> float32 ``(N, 13, H, W)`` reflectance in ``[0, 1]``
        * ``metadata``   -> the sample metadata dict (returned unbatched)
    """

    def __init__(
        self,
        npz_dir: Path | str,
        *,
        reflectance_scale: float = 10000.0,
    ) -> None:
        """Initialise the dataset over a directory of ``.npz`` samples.

        Args:
            npz_dir: Directory containing ``*.npz`` sample files.
            reflectance_scale: Divisor mapping stored DN to ``[0, 1]`` reflectance.

        Raises:
            ImportError: If PyTorch is not installed.
            FileNotFoundError: If the directory contains no ``.npz`` files.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("PyTorch is required to use this dataset") from exc

        self.npz_dir = Path(npz_dir)
        self.scale = reflectance_scale
        self.paths = sorted(self.npz_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz samples in {self.npz_dir}")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and tensorise one sample.

        Args:
            index: Sample index.

        Returns:
            A dict with ``target``, ``mask``, ``references`` tensors and
            ``metadata``.
        """
        import numpy as np
        import torch

        with np.load(self.paths[index]) as data:
            target = data["target"].astype(np.float32) / self.scale
            mask = data["mask"].astype(np.float32)
            references = data["references"].astype(np.float32) / self.scale
            metadata = json.loads(str(data["metadata"]))

        return {
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "references": torch.from_numpy(references),
            "metadata": metadata,
        }
