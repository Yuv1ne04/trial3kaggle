"""Patch-grid generation and per-patch filtering.

The grid is computed from raster dimensions and the configured size/stride;
filtering is evaluated from only the target and mask windows, so reference
windows are read solely for patches that are kept (memory-efficient).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DatasetConfig


def generate_grid(
    width: int,
    height: int,
    size: int,
    stride: int,
    *,
    drop_partial: bool,
) -> list[tuple[int, int]]:
    """Generate the list of patch top-left ``(row, col)`` origins.

    Supports overlapping (``stride < size``) and non-overlapping
    (``stride == size``) tilings. Edge handling is controlled by
    ``drop_partial``.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        size: Patch side length in pixels.
        stride: Step between successive origins.
        drop_partial: When ``True`` patches that would exceed the raster are
            dropped; when ``False`` the final row/column is shifted inward so
            every patch stays fully inside the raster.

    Returns:
        A list of unique ``(row, col)`` origins in row-major order.
    """
    if size > width or size > height:
        return []

    def _starts(extent: int) -> list[int]:
        positions = list(range(0, extent - size + 1, stride))
        last = extent - size
        if not drop_partial and (not positions or positions[-1] != last):
            positions.append(last)
        return positions

    rows = _starts(height)
    cols = _starts(width)
    return [(r, c) for r in rows for c in cols]


@dataclass(slots=True)
class PatchMetrics:
    """Quality metrics computed for a candidate patch.

    Attributes:
        cloud_fraction: Fraction of cloud pixels ``[0, 1]``.
        nodata_fraction: Fraction of NoData pixels ``[0, 1]``.
        background_fraction: Fraction of ocean/background pixels ``[0, 1]``.
        valid_fraction: Fraction of valid (non-NoData) pixels ``[0, 1]``.
    """

    cloud_fraction: float
    nodata_fraction: float
    background_fraction: float
    valid_fraction: float


def compute_metrics(
    target_patch: np.ndarray,
    mask_patch: np.ndarray,
    config: DatasetConfig,
) -> PatchMetrics:
    """Compute filtering metrics from a target and mask window.

    Args:
        target_patch: Target stack window, shape ``(13, H, W)``.
        mask_patch: Cloud-mask window, shape ``(H, W)``.
        config: Active dataset configuration.

    Returns:
        The computed :class:`PatchMetrics`.
    """
    total = float(mask_patch.size)
    nodata = (target_patch == config.stack_nodata).all(axis=0)
    background = (target_patch <= config.filters.background_reflectance).all(axis=0)
    cloud = mask_patch == config.mask_cloud_value

    nodata_fraction = float(nodata.sum()) / total
    return PatchMetrics(
        cloud_fraction=float(cloud.sum()) / total,
        nodata_fraction=nodata_fraction,
        background_fraction=float(background.sum()) / total,
        valid_fraction=1.0 - nodata_fraction,
    )


def passes_filters(metrics: PatchMetrics, config: DatasetConfig) -> bool:
    """Return whether a patch satisfies every configured filter.

    Args:
        metrics: The patch's computed metrics.
        config: Active dataset configuration.

    Returns:
        ``True`` if the patch should be kept, else ``False``.
    """
    filters = config.filters
    return (
        metrics.nodata_fraction <= filters.max_nodata_fraction
        and metrics.background_fraction <= filters.max_background_fraction
        and metrics.valid_fraction >= filters.min_valid_fraction
        and filters.min_cloud_fraction <= metrics.cloud_fraction <= filters.max_cloud_fraction
    )
