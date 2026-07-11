"""Windowed reading and single-copy writing of library patches.

Every patch is written to a temp file then atomically replaced (with retries to
survive antivirus/indexer locks on Windows). Writes are idempotent: an existing
library file is never rewritten, which is what makes generation resumable and
guarantees each patch exists once.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from .models import PatchKey


def robust_replace(src: Path, dst: Path, *, attempts: int = 6, delay: float = 0.25) -> None:
    """Atomically replace ``dst`` with ``src``, retrying on transient locks.

    Args:
        src: Source temp path.
        dst: Destination path.
        attempts: Maximum tries before giving up.
        delay: Base backoff delay (seconds), scaled by attempt number.

    Raises:
        PermissionError: If every attempt fails.
    """
    for attempt in range(1, attempts + 1):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay * attempt)


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Write a compressed npz atomically.

    Args:
        path: Final destination path (parents created).
        arrays: Named arrays to store.
        metadata: Metadata dict, stored as a JSON string (no pickle needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata=np.array(json.dumps(metadata)), **arrays)
    tmp = path.with_suffix(".npz.tmp")
    tmp.write_bytes(buffer.getvalue())
    robust_replace(tmp, path)


def read_window(dataset: "rasterio.io.DatasetReader", row: int, col: int, size: int,
                *, band: int | None = None) -> np.ndarray:
    """Read a square window from an open dataset.

    Args:
        dataset: An open Rasterio dataset.
        row: Window top row (pixels).
        col: Window left column (pixels).
        size: Window side length (pixels).
        band: 1-based band to read, or ``None`` to read all bands.

    Returns:
        ``(size, size)`` if ``band`` is given, else ``(bands, size, size)``.
    """
    window = Window(col, row, size, size)
    if band is not None:
        return dataset.read(band, window=window)
    return dataset.read(window=window)


def _write_once(root: Path, relpath: str, arrays: dict[str, np.ndarray],
                metadata: dict[str, Any], *, overwrite: bool) -> bool:
    """Write an npz at ``relpath`` only if absent (or ``overwrite``).

    Args:
        root: Dataset root directory.
        relpath: Dataset-relative destination path.
        arrays: Named arrays to store.
        metadata: Patch metadata.
        overwrite: Force rewrite even if present.

    Returns:
        ``True`` if a file was written, ``False`` if it already existed.
    """
    path = root / relpath
    if path.exists() and not overwrite:
        return False
    _atomic_savez(path, arrays, metadata)
    return True


def write_target_image(root: Path, key: PatchKey, image: np.ndarray,
                       metadata: dict[str, Any], *, overwrite: bool = False) -> bool:
    """Write a 13-band patch to the *target* library (once).

    Args:
        root: Dataset root.
        key: Patch key.
        image: ``(13, size, size)`` array.
        metadata: Target-independent patch metadata.
        overwrite: Force rewrite even if present.

    Returns:
        ``True`` if written, ``False`` if it already existed.
    """
    return _write_once(root, key.target_relpath(), {"image": image}, metadata,
                       overwrite=overwrite)


def write_reference_image(root: Path, key: PatchKey, image: np.ndarray,
                          metadata: dict[str, Any], *, overwrite: bool = False) -> bool:
    """Write a 13-band patch to the *reference* library (once).

    Args:
        root: Dataset root.
        key: Patch key.
        image: ``(13, size, size)`` array.
        metadata: Target-independent patch metadata.
        overwrite: Force rewrite even if present.

    Returns:
        ``True`` if written, ``False`` if it already existed.
    """
    return _write_once(root, key.reference_relpath(), {"image": image}, metadata,
                       overwrite=overwrite)


def write_mask(root: Path, key: PatchKey, mask: np.ndarray,
               metadata: dict[str, Any], *, overwrite: bool = False) -> bool:
    """Write a cloud-mask patch to the *mask* library (once).

    Args:
        root: Dataset root.
        key: Patch key.
        mask: ``(1, size, size)`` array.
        metadata: Target-independent patch metadata.
        overwrite: Force rewrite even if present.

    Returns:
        ``True`` if written, ``False`` if it already existed.
    """
    return _write_once(root, key.mask_relpath(), {"mask": mask}, metadata,
                       overwrite=overwrite)


def patch_metadata(key: PatchKey, row: int, col: int, crs: str | None,
                   transform: tuple[float, ...]) -> dict[str, Any]:
    """Build the target-independent metadata stored inside a library patch.

    Args:
        key: The patch key.
        row: Patch top row (pixels).
        col: Patch left column (pixels).
        crs: CRS authority string.
        transform: 6-element patch affine transform.

    Returns:
        A JSON-serialisable metadata dict.
    """
    return {
        "date": key.date,
        "patch_size": key.size,
        "cell_index": key.cell_index,
        "patch_coordinates": {"row": row, "col": col},
        "crs": crs,
        "transform": list(transform),
    }
