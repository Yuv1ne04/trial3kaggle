"""Per-patch reference quality evaluation and single-copy reference writing.

A candidate reference is only useful where it is actually clear. For each needed
``(size, reference_date, cell)`` this reads the reference's cloud-mask and stack
windows, decides validity against the configured thresholds, and — if valid —
writes the reference image patch into the reference library exactly once.

Validity is target-independent (it depends only on the reference patch itself),
so it is computed once per ``(size, date, cell)`` and shared by every sample
that wants that reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.windows import Window, transform as window_transform

from .extract import patch_metadata, write_reference_image
from .models import PatchKey


@dataclass(slots=True)
class RefEvalParams:
    """Thresholds and raster conventions for reference evaluation.

    Attributes:
        max_cloud_fraction: Reject a reference patch above this cloud fraction.
        max_nodata_fraction: Reject a reference patch above this NoData fraction.
        stack_nodata: Stack NoData value (for NoData detection).
        mask_cloud_value: Mask value denoting cloud.
    """

    max_cloud_fraction: float
    max_nodata_fraction: float
    stack_nodata: int
    mask_cloud_value: int


#: A worker payload: (size, date, stack_path, mask_path|"", cells, root, params).
GroupPayload = tuple[int, str, str, str, list[tuple[int, int, int]], str, RefEvalParams]


def evaluate_reference_group(payload: GroupPayload) -> tuple[dict[tuple[int, str, int], bool], int, str]:
    """Evaluate and write reference patches for one ``(size, date)`` group.

    Args:
        payload: ``(size, date, stack_path, mask_path, cells, root, params)``
            where ``cells`` is a list of ``(cell_index, row, col)`` and
            ``mask_path`` is ``""`` when no reference mask is available.

    Returns:
        ``(validity, written, error)`` where ``validity`` maps
        ``(size, date, cell)`` -> bool, ``written`` counts new files, and
        ``error`` is empty on success.
    """
    size, date, stack_path, mask_path, cells, root_str, params = payload
    root = Path(root_str)
    validity: dict[tuple[int, str, int], bool] = {}
    written = 0

    if not mask_path:
        # No reference mask -> cannot certify clarity -> treat as invalid.
        for cell_index, _row, _col in cells:
            validity[(size, date, cell_index)] = False
        return validity, 0, ""

    try:
        with rasterio.open(stack_path) as sds, rasterio.open(mask_path) as mds:
            crs = _crs_string(sds)
            for cell_index, row, col in cells:
                key = PatchKey(size, date, cell_index)
                window = Window(col, row, size, size)
                mask_win = mds.read(1, window=window)
                cloud_frac = float((mask_win == params.mask_cloud_value).mean())

                image = sds.read(window=window)
                nodata_frac = float((image == params.stack_nodata).all(axis=0).mean())

                is_valid = (cloud_frac <= params.max_cloud_fraction
                            and nodata_frac <= params.max_nodata_fraction)
                validity[(size, date, cell_index)] = is_valid
                if not is_valid:
                    continue
                if (root / key.reference_relpath()).exists():
                    continue
                transform = tuple(window_transform(window, sds.transform))[:6]
                meta = patch_metadata(key, row, col, crs, transform)
                meta["reference_cloud_fraction"] = round(cloud_frac, 6)
                if write_reference_image(root, key, image, meta):
                    written += 1
        return validity, written, ""
    except Exception as exc:  # noqa: BLE001 - per-group guard
        return validity, written, f"{date}/{size}: {type(exc).__name__}: {exc}"


def _crs_string(ds: "rasterio.io.DatasetReader") -> str | None:
    """Return a dataset CRS as ``AUTHORITY:CODE`` or string form."""
    if ds.crs is None:
        return None
    auth = ds.crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else ds.crs.to_string()
