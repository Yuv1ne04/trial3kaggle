"""Library writers for the synthetic dataset (parallel workers).

Two single-copy libraries are materialised with windowed reads:
  * ``patch_library`` — 13-band patches (ground truth and references).
  * ``cloud_tile_library`` — a real cloud mask (and, for the ``overlay`` fill,
    the real cloud reflectance) per unique transplanted cloud.

Both writers are idempotent (skip existing files), so generation is resumable
and every patch/tile is stored exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

from ..shared.extract import _atomic_savez, read_window
from . import ids
from .config import SyntheticConfig


def _crs_string(ds: "rasterio.io.DatasetReader") -> str | None:
    """Return a dataset CRS as ``AUTHORITY:CODE`` or its string form."""
    if ds.crs is None:
        return None
    auth = ds.crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else ds.crs.to_string()


def write_patch_group(
    payload: tuple[str, str, list[tuple[int, int, int]], int, str]
) -> tuple[int, str]:
    """Write the 13-band image patches needed from one acquisition's stack.

    Args:
        payload: ``(date, stack_path, cells, size, root)`` where ``cells`` is a
            list of ``(cell_index, row, col)``.

    Returns:
        ``(written_count, error)``; ``error`` empty on success.
    """
    date, stack_path, cells, size, root_str = payload
    root = Path(root_str)
    written = 0
    try:
        with rasterio.open(stack_path) as ds:
            crs = _crs_string(ds)
            for cell_index, row, col in cells:
                rel = ids.patch_relpath(size, date, cell_index)
                path = root / rel
                if path.exists():
                    continue
                image = read_window(ds, row, col, size)
                transform = tuple(window_transform(
                    Window(col, row, size, size), ds.transform))[:6]
                _atomic_savez(path, {"image": image}, {
                    "date": date, "cell_index": cell_index, "patch_size": size,
                    "patch_coordinates": {"row": row, "col": col},
                    "crs": crs, "transform": list(transform),
                })
                written += 1
        return written, ""
    except Exception as exc:  # noqa: BLE001 - per-group guard
        return written, f"patch {date}/{size}: {type(exc).__name__}: {exc}"


def write_cloud_tile_group(
    payload: tuple[str, str, str, list[tuple[int, int, int]], int, str, str, int]
) -> tuple[int, str]:
    """Write the cloud tiles needed from one mask acquisition.

    A tile stores the binary mask and — when ``cloud_fill`` is ``overlay`` — the
    real cloud reflectance read from the mask's own stack at the same window.

    Args:
        payload: ``(mask_date, mask_path, stack_path, cells, size, root,
            cloud_fill, cloud_value)`` where ``cells`` is
            ``(cell_index, source_row, source_col)`` and ``stack_path`` may be
            ``""`` when no reflectance source is available.

    Returns:
        ``(written_count, error)``; ``error`` empty on success.
    """
    (mask_date, mask_path, stack_path, cells, size, root_str,
     cloud_fill, cloud_value) = payload
    root = Path(root_str)
    written = 0
    try:
        stack_ds = rasterio.open(stack_path) if (cloud_fill == "overlay" and stack_path) else None
        try:
            with rasterio.open(mask_path) as mds:
                for cell_index, row, col in cells:
                    rel = ids.cloud_tile_relpath(size, mask_date, cell_index)
                    path = root / rel
                    if path.exists():
                        continue
                    mask = mds.read(1, window=Window(col, row, size, size))
                    arrays: dict[str, np.ndarray] = {"mask": mask[None, :, :]}
                    if stack_ds is not None:
                        arrays["cloud"] = read_window(stack_ds, row, col, size)
                    _atomic_savez(path, arrays, {
                        "mask_date": mask_date, "cell_index": cell_index,
                        "patch_size": size,
                        "cloud_value": cloud_value, "has_reflectance": stack_ds is not None,
                    })
                    written += 1
        finally:
            if stack_ds is not None:
                stack_ds.close()
        return written, ""
    except Exception as exc:  # noqa: BLE001 - per-group guard
        return written, f"tile {mask_date}/{size}: {type(exc).__name__}: {exc}"
