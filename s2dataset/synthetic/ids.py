"""Path scheme for the storage-efficient synthetic dataset.

Nothing redundant is stored. Each 13-band patch lives once in ``patch_library``
(used as ground truth or as a reference); each transplanted real cloud lives
once in ``cloud_tile_library`` (binary mask + optional real cloud reflectance);
a training sample is a small JSON manifest that references them. Paths are
relative to the dataset root for portability (e.g. mounting on Kaggle).
"""

from __future__ import annotations

from pathlib import PurePosixPath

PATCH_LIBRARY_DIR = "patch_library"
CLOUD_TILE_LIBRARY_DIR = "cloud_tile_library"
SAMPLES_DIR = "samples"

SPLIT_FOLDER: dict[str, str] = {"train": "train", "val": "validation", "test": "test"}


def patch_filename(cell_index: int) -> str:
    """Return the npz filename for a grid cell (e.g. ``patch_000254.npz``)."""
    return f"patch_{cell_index:06d}.npz"


def patch_relpath(size: int, date: str, cell_index: int) -> str:
    """Return the ``patch_library`` relative path for a 13-band patch.

    Args:
        size: Patch size in pixels.
        date: Acquisition date (``YYYYMMDD``).
        cell_index: Cell index within the grid.

    Returns:
        A POSIX relative path string.
    """
    return str(PurePosixPath(PATCH_LIBRARY_DIR) / str(size) / date /
               patch_filename(cell_index))


def cloud_tile_relpath(size: int, mask_date: str, cell_index: int) -> str:
    """Return the ``cloud_tile_library`` relative path for a cloud tile.

    Args:
        size: Patch size in pixels.
        mask_date: Source acquisition date of the mask (``YYYYMMDD``).
        cell_index: Cell index of the mask within its acquisition.

    Returns:
        A POSIX relative path string.
    """
    return str(PurePosixPath(CLOUD_TILE_LIBRARY_DIR) / str(size) / mask_date /
               patch_filename(cell_index))


def sample_relpath(split: str, sample_id: str) -> str:
    """Return the relative path of a sample manifest JSON.

    Args:
        split: Internal split label (``train`` / ``val`` / ``test``).
        sample_id: Sample identifier (filename stem).

    Returns:
        A POSIX relative path string.
    """
    folder = SPLIT_FOLDER.get(split, split)
    return str(PurePosixPath(SAMPLES_DIR) / folder / f"{sample_id}.json")
