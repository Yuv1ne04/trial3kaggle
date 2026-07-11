"""PyTorch ``Dataset`` for the shared-reference architecture.

Reads a sample JSON (target/mask/reference paths), loads the referenced library
patches, pads the references to a fixed ``maximum_references`` count with zeros
and returns a ``reference_validity_mask`` so the model always receives a fixed
tensor shape regardless of how many references (2–4) were available. The
``torch`` import is deferred so this module is import-safe without PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class S2SharedReconstructionDataset:
    """Loads 2–4-reference samples with zero-padding and a validity mask.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> ds = S2SharedReconstructionDataset("dataset", split="train", maximum_references=4)
        >>> loader = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)

    Each item is a dict of tensors:
        * ``target``                 -> float32 ``(13, H, W)`` in ``[0, 1]``
        * ``mask``                   -> float32 ``(1, H, W)`` (1 = cloud)
        * ``references``             -> float32 ``(R, 13, H, W)``, zero-padded
        * ``reference_validity_mask``-> float32 ``(R,)`` (1 = real, 0 = padding)
        * ``metadata``               -> the sample metadata dict

    ``R`` is fixed (``maximum_references``); padded slots are all-zero and have
    validity 0, so the model never needs to know how many references existed.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        *,
        maximum_references: int = 4,
        reflectance_scale: float = 10000.0,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Dataset root directory (containing ``samples/``).
            split: ``"train"`` / ``"validation"`` / ``"test"`` (``"val"`` alias ok).
            maximum_references: Fixed number of reference slots ``R``; samples
                with fewer real references are zero-padded to this length.
            reflectance_scale: Divisor mapping stored DN to ``[0, 1]`` reflectance.

        Raises:
            ImportError: If PyTorch is not installed.
            FileNotFoundError: If no sample JSON files are found.
            ValueError: If ``maximum_references`` < 1.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("PyTorch is required to use this dataset") from exc
        if maximum_references < 1:
            raise ValueError("maximum_references must be >= 1")

        self.root = Path(root)
        folder = {"val": "validation"}.get(split, split)
        self.split_dir = self.root / "samples" / folder
        self.max_refs = maximum_references
        self.scale = reflectance_scale
        self.sample_files = sorted(self.split_dir.glob("*.json"))
        if not self.sample_files:
            raise FileNotFoundError(f"No sample JSON files in {self.split_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.sample_files)

    def _load_image(self, relpath: str):
        """Load a library image patch as a scaled float32 array.

        Args:
            relpath: Dataset-relative path to the image npz.

        Returns:
            A ``(13, H, W)`` float32 numpy array in ``[0, 1]``.
        """
        import numpy as np

        with np.load(self.root / relpath) as data:
            return data["image"].astype(np.float32) / self.scale

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load, pad and tensorise one sample.

        Args:
            index: Sample index within the split.

        Returns:
            A dict of tensors plus ``metadata``; references are zero-padded to
            ``maximum_references`` with an accompanying validity mask.
        """
        import numpy as np
        import torch

        spec = json.loads(self.sample_files[index].read_text(encoding="utf-8"))
        target = self._load_image(spec["target"])
        with np.load(self.root / spec["mask"]) as mdata:
            mask = mdata["mask"].astype(np.float32)

        ref_paths = spec["references"][: self.max_refs]
        bands, h, w = target.shape
        references = np.zeros((self.max_refs, bands, h, w), dtype=np.float32)
        validity = np.zeros((self.max_refs,), dtype=np.float32)
        for i, ref_path in enumerate(ref_paths):
            references[i] = self._load_image(ref_path)
            validity[i] = 1.0

        return {
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "references": torch.from_numpy(references),
            "reference_validity_mask": torch.from_numpy(validity),
            "metadata": spec.get("metadata", {}),
        }
