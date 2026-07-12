"""PyTorch ``Dataset`` for the manifest-based synthetic dataset.

Reads a sample manifest and *composes* the synthetic cloudy input at load time
from the clear ground truth + the transplanted cloud tile — so no corrupted
pixels are ever stored. References are zero-padded to a fixed slot count with a
``reference_validity_mask``. The ``torch`` import is deferred so this module is
import-safe without PyTorch.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .corruptor import compose_cloudy

_PATCH_KEY = re.compile(r"[/\\](\d{8})[/\\]patch_(\d+)\.npz$")


def _load_gt_exclusions(gt_filter: Any) -> set[str]:
    """Return the set of excluded ground-truth patch ids from a filter source.

    Args:
        gt_filter: A set/list of ``"<date>_<cell>"`` ids, or a path to an
            ``ground_truth_filter_manifest.json`` (uses ``exclude_patch_ids``).

    Returns:
        A set of excluded patch ids (empty when ``gt_filter`` is falsy).
    """
    if not gt_filter:
        return set()
    if isinstance(gt_filter, (set, list, tuple)):
        return set(gt_filter)
    data = json.loads(Path(gt_filter).read_text(encoding="utf-8"))
    return set(data.get("exclude_patch_ids", data if isinstance(data, list) else []))


class S2SyntheticDataset:
    """Loads synthetic (cloudy-in, clear-out) pairs, composing clouds on the fly.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> ds = S2SyntheticDataset("synthetic_dataset", split="train",
        ...                         maximum_references=4)
        >>> loader = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)

    Each item is a dict of tensors:
        * ``cloudy``                 -> float32 ``(13, H, W)`` synthetic input
        * ``mask``                   -> float32 ``(1, H, W)`` (1 = cloud)
        * ``references``             -> float32 ``(R, 13, H, W)`` zero-padded
        * ``reference_validity_mask``-> float32 ``(R,)``
        * ``ground_truth``           -> float32 ``(13, H, W)`` target
        * ``metadata``               -> the sample metadata dict
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "train",
        *,
        maximum_references: int = 4,
        reflectance_scale: float = 10000.0,
        cloud_fill: str | None = None,
        constant_fill_value: int = 8000,
        mask_cloud_value: int = 1,
        difficulty: str | None = None,
        gt_filter: Any = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            root: Dataset root (containing ``samples/`` and the libraries).
            split: ``train`` / ``validation`` / ``test`` (``val`` accepted).
            maximum_references: Fixed reference-slot count ``R`` (zero-padded).
            reflectance_scale: Divisor mapping DN to ``[0, 1]`` reflectance.
            cloud_fill: Override the manifest's fill mode (``overlay``/
                ``constant``/``zero``); ``None`` uses each manifest's value.
            constant_fill_value: Fill value for the ``constant`` mode.
            mask_cloud_value: Mask value denoting cloud.
            difficulty: If set, keep only samples of this curriculum band
                (enables easy->hard curriculum scheduling).
            gt_filter: Optional ground-truth exclusion source (a set of
                ``"<date>_<cell>"`` ids or a path to
                ``ground_truth_filter_manifest.json``). Samples whose ground
                truth is excluded are dropped. Default ``None`` = no filtering
                (fully backward compatible).

        Raises:
            ImportError: If PyTorch is not installed.
            FileNotFoundError: If no manifests are found.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("PyTorch is required to use this dataset") from exc

        self.root = Path(root)
        folder = {"val": "validation"}.get(split, split)
        self.split_dir = self.root / "samples" / folder
        self.max_refs = maximum_references
        self.scale = reflectance_scale
        self.cloud_fill_override = cloud_fill
        self.constant_fill_value = constant_fill_value
        self.mask_cloud_value = mask_cloud_value

        files = sorted(self.split_dir.glob("*.json"))
        if difficulty is not None:
            files = [f for f in files if self._difficulty(f) == difficulty]
        exclude = _load_gt_exclusions(gt_filter)
        if exclude:
            files = [f for f in files if self._gt_patch_id(f) not in exclude]
        self.sample_files = files
        if not self.sample_files:
            raise FileNotFoundError(f"No manifests in {self.split_dir} (difficulty={difficulty})")

    @staticmethod
    def _difficulty(path: Path) -> str | None:
        """Read a manifest's difficulty band without loading arrays."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))["metadata"].get("difficulty")
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    @staticmethod
    def _gt_patch_id(path: Path) -> str | None:
        """Return the ``"<date>_<cell>"`` id of a manifest's ground truth."""
        try:
            gt = json.loads(path.read_text(encoding="utf-8")).get("ground_truth", "")
        except (json.JSONDecodeError, OSError):
            return None
        m = _PATCH_KEY.search(gt or "")
        return f"{m.group(1)}_{int(m.group(2))}" if m else None

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.sample_files)

    def _load_image(self, relpath: str):
        """Load a library image patch as a scaled float32 array.

        Args:
            relpath: Dataset-relative image-patch path.

        Returns:
            A ``(13, H, W)`` float32 array in ``[0, 1]``.
        """
        import numpy as np

        with np.load(self.root / relpath) as data:
            return data["image"].astype(np.float32) / self.scale

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Compose and tensorise one synthetic sample.

        Args:
            index: Sample index within the split.

        Returns:
            A dict of tensors plus ``metadata``.
        """
        import numpy as np
        import torch

        spec = json.loads(self.sample_files[index].read_text(encoding="utf-8"))
        meta = spec.get("metadata", {})
        fill = self.cloud_fill_override or meta.get("cloud_fill", "overlay")

        gt = self._load_image(spec["ground_truth"])  # (13,H,W) in [0,1]
        with np.load(self.root / spec["cloud_tile"]) as tile:
            mask = tile["mask"]
            cloud = tile["cloud"].astype(np.float32) / self.scale if "cloud" in tile else None

        # Arrays are already in [0, 1]; scale the constant fill to match.
        cloudy = compose_cloudy(
            gt, mask, cloud, cloud_fill=fill,
            constant_value=self.constant_fill_value / self.scale,
            cloud_value=self.mask_cloud_value)

        bands, h, w = gt.shape
        references = np.zeros((self.max_refs, bands, h, w), dtype=np.float32)
        validity = np.zeros((self.max_refs,), dtype=np.float32)
        for i, ref_path in enumerate(spec["references"][: self.max_refs]):
            references[i] = self._load_image(ref_path)
            validity[i] = 1.0

        mask_f = (mask[0] if mask.ndim == 3 else mask).astype(np.float32)[None, :, :]
        return {
            "cloudy": torch.from_numpy(cloudy),
            "mask": torch.from_numpy(mask_f),
            "references": torch.from_numpy(references),
            "reference_validity_mask": torch.from_numpy(validity),
            "ground_truth": torch.from_numpy(gt),
            "metadata": meta,
        }
