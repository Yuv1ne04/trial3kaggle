"""Migrate an old duplicated-NPZ dataset into the shared-reference layout.

The original builder wrote one self-contained NPZ per sample
(``target``, ``mask``, ``references``, ``metadata``). This converter decomposes
each file into single-copy library patches (target/reference/mask) plus a sample
JSON, deduplicating identical patches via filename existence. It never rewrites
an existing library patch, so migration is idempotent and resumable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..logging_setup import get_logger
from . import ids
from .extract import write_mask, write_reference_image, write_target_image
from .models import PatchKey

logger = get_logger()


def _iso(date: str) -> str:
    """Return ``date`` (``YYYYMMDD``) as ISO ``YYYY-MM-DD`` if possible."""
    try:
        return datetime.strptime(date, "%Y%m%d").date().isoformat()
    except ValueError:
        return date


def _base_meta(key: PatchKey, coords, crs, transform) -> dict:
    """Build target-independent patch metadata for a migrated patch."""
    return {
        "date": key.date, "patch_size": key.size, "cell_index": key.cell_index,
        "patch_coordinates": {"row": coords[0], "col": coords[1]} if coords else {},
        "crs": crs, "transform": transform,
    }


def migrate_dataset(old_root: Path, new_root: Path) -> dict[str, int]:
    """Convert an old duplicated dataset tree into the shared-reference layout.

    Args:
        old_root: Root of the old dataset (``patches_<size>/<split>_npz/*.npz``).
        new_root: Destination root for the shared-reference dataset.

    Returns:
        A counts dict (samples, target/reference/mask patches written).

    Raises:
        FileNotFoundError: If no old NPZ samples are found.
    """
    old_root, new_root = Path(old_root), Path(new_root)
    npz_files = sorted(old_root.rglob("*_npz/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No old NPZ samples found under {old_root}")
    logger.info("Migrating %d old sample(s) from %s", len(npz_files), old_root)

    counts = {"samples": 0, "target_patches": 0, "reference_patches": 0, "mask_patches": 0}
    split_from_dir = {"train_npz": "train", "val_npz": "val", "test_npz": "test"}

    for n, npz_path in enumerate(tqdm(npz_files, desc="Migrate", unit="smp"), 1):
        split = split_from_dir.get(npz_path.parent.name, "train")
        with np.load(npz_path, allow_pickle=False) as data:
            target, mask, references = data["target"], data["mask"], data["references"]
            meta = json.loads(str(data["metadata"]))

        size = int(meta["patch_size"])
        cell = int(meta["patch_index"])
        tdate = str(meta["target_date"])
        ref_dates = [str(d) for d in meta.get("reference_dates", [])]
        coords = meta.get("patch_coordinates", {})
        coord_pair = [coords.get("row", 0), coords.get("col", 0)]
        crs, transform = meta.get("crs"), meta.get("transform")

        tkey = PatchKey(size, tdate, cell)
        if write_target_image(new_root, tkey, target, _base_meta(tkey, coord_pair, crs, transform)):
            counts["target_patches"] += 1
        if write_mask(new_root, tkey, mask, _base_meta(tkey, coord_pair, crs, transform)):
            counts["mask_patches"] += 1
        for i, ref_date in enumerate(ref_dates):
            if i >= references.shape[0]:
                break
            rkey = PatchKey(size, ref_date, cell)
            if write_reference_image(new_root, rkey, references[i],
                                     _base_meta(rkey, coord_pair, crs, transform)):
                counts["reference_patches"] += 1

        sample_id = f"sample_{n:06d}"
        sample_json = {
            "target": tkey.target_relpath(),
            "mask": tkey.mask_relpath(),
            "references": [PatchKey(size, d, cell).reference_relpath() for d in ref_dates],
            "metadata": {
                "sample_id": sample_id, "split": split,
                "target_date": _iso(tdate), "target_date_compact": tdate,
                "reference_dates": [_iso(d) for d in ref_dates],
                "n_references": len(ref_dates),
                "cloud_percentage": meta.get("cloud_percentage"),
                "patch_size": size, "cell_index": cell, "coordinates": coord_pair,
                "crs": crs, "transform": transform,
            },
        }
        out = new_root / ids.sample_relpath(split, sample_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sample_json, indent=2), encoding="utf-8")
        counts["samples"] += 1

    logger.info("Migration complete: %s", counts)
    return counts
