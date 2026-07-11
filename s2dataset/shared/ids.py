"""Identifier and path scheme for the shared-reference dataset.

Three single-copy libraries are used, exactly as specified:
``target_library``, ``reference_library`` and ``mask_library``. A ``{size}``
sub-directory is inserted under each library so that 128/256/512 datasets can
coexist in one tree (the spec's diagram omits it for brevity; it is required for
the multi-scale capability). Paths are stored *relative to the dataset root* so
the dataset is portable (e.g. mounted at a different path on Kaggle).
"""

from __future__ import annotations

from pathlib import PurePosixPath

#: Top-level library / output directory names within the dataset root.
TARGET_LIBRARY_DIR = "target_library"
REFERENCE_LIBRARY_DIR = "reference_library"
MASK_LIBRARY_DIR = "mask_library"
SAMPLES_DIR = "samples"

#: Mapping of internal split label -> on-disk samples sub-folder.
SPLIT_FOLDER: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


def patch_filename(cell_index: int) -> str:
    """Return the npz filename for a grid cell.

    Args:
        cell_index: Zero-based index of the cell within its (size) grid.

    Returns:
        A filename such as ``"patch_000254.npz"``.
    """
    return f"patch_{cell_index:06d}.npz"


def _relpath(library: str, size: int, date: str, cell_index: int) -> str:
    """Build a ``library/size/date/patch.npz`` POSIX relative path."""
    return str(PurePosixPath(library) / str(size) / date / patch_filename(cell_index))


def target_relpath(size: int, date: str, cell_index: int) -> str:
    """Return the dataset-relative path of a target image patch.

    Args:
        size: Patch size in pixels.
        date: Acquisition date as ``YYYYMMDD``.
        cell_index: Cell index within the grid.

    Returns:
        A POSIX relative path string.
    """
    return _relpath(TARGET_LIBRARY_DIR, size, date, cell_index)


def reference_relpath(size: int, date: str, cell_index: int) -> str:
    """Return the dataset-relative path of a reference image patch.

    Args:
        size: Patch size in pixels.
        date: Acquisition date as ``YYYYMMDD``.
        cell_index: Cell index within the grid.

    Returns:
        A POSIX relative path string.
    """
    return _relpath(REFERENCE_LIBRARY_DIR, size, date, cell_index)


def mask_relpath(size: int, date: str, cell_index: int) -> str:
    """Return the dataset-relative path of a mask patch.

    Args:
        size: Patch size in pixels.
        date: Acquisition date as ``YYYYMMDD``.
        cell_index: Cell index within the grid.

    Returns:
        A POSIX relative path string.
    """
    return _relpath(MASK_LIBRARY_DIR, size, date, cell_index)


def sample_relpath(split: str, sample_id: str) -> str:
    """Return the dataset-relative path of a sample JSON.

    Args:
        split: Internal split label (``train`` / ``val`` / ``test``).
        sample_id: Sample identifier (filename stem).

    Returns:
        A POSIX relative path string.
    """
    folder = SPLIT_FOLDER.get(split, split)
    return str(PurePosixPath(SAMPLES_DIR) / folder / f"{sample_id}.json")
