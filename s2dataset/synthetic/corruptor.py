"""Synthetic cloud composition utilities.

Composition is deferred to load time (the PyTorch dataset), so nothing corrupted
is stored on disk. This module provides the single compose function used by the
loader plus the coverage/difficulty/QC helpers used at manifest time.
"""

from __future__ import annotations

import numpy as np

from .config import SyntheticConfig


def compose_cloudy(
    ground_truth: np.ndarray,
    mask: np.ndarray,
    cloud_reflectance: np.ndarray | None,
    *,
    cloud_fill: str,
    constant_value: int,
    cloud_value: int,
) -> np.ndarray:
    """Compose a synthetic cloudy image from a clear patch and a real mask.

    The ground truth is never mutated (a copy is returned). Cloudy pixels are
    filled per ``cloud_fill``: real cloud reflectance (``overlay``), a constant,
    or zero.

    Args:
        ground_truth: Clear patch, shape ``(13, H, W)``.
        mask: Cloud mask, shape ``(1, H, W)`` or ``(H, W)``.
        cloud_reflectance: Optional ``(13, H, W)`` real cloud reflectance for the
            ``overlay`` fill.
        cloud_fill: ``"overlay"``, ``"constant"`` or ``"zero"``.
        constant_value: Fill value used when ``cloud_fill`` is ``"constant"``.
        cloud_value: Mask value denoting cloud.

    Returns:
        The synthetic cloudy image, shape ``(13, H, W)``.
    """
    mask2d = mask[0] if mask.ndim == 3 else mask
    cloud = mask2d == cloud_value
    cloudy = ground_truth.copy()
    if cloud_fill == "overlay" and cloud_reflectance is not None:
        cloudy[:, cloud] = cloud_reflectance[:, cloud]
    elif cloud_fill == "zero":
        cloudy[:, cloud] = 0
    else:
        cloudy[:, cloud] = constant_value
    return cloudy


def applied_coverage(mask: np.ndarray, ground_truth: np.ndarray,
                     config: SyntheticConfig) -> float:
    """Compute cloud coverage over the valid (non-NoData) ground-truth pixels.

    Args:
        mask: Cloud mask, shape ``(1, H, W)`` or ``(H, W)``.
        ground_truth: Clear patch, shape ``(13, H, W)``.
        config: Active synthetic configuration.

    Returns:
        Cloud coverage in ``[0, 1]``.
    """
    mask2d = mask[0] if mask.ndim == 3 else mask
    valid = ~(ground_truth == config.stack_nodata).all(axis=0)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0
    return float(((mask2d == config.mask_cloud_value) & valid).sum()) / valid_count


def difficulty_for_coverage(coverage: float, config: SyntheticConfig) -> str:
    """Return the curriculum band name a coverage value falls into.

    Args:
        coverage: Cloud coverage ``[0, 1]``.
        config: Active synthetic configuration.

    Returns:
        The band name, or ``"unbinned"`` if it matches none.
    """
    for band in config.curriculum.bins:
        if band.min_coverage <= coverage <= band.max_coverage:
            return band.name
    return "unbinned"


def coverage_in_qc_range(coverage: float, config: SyntheticConfig) -> tuple[bool, str]:
    """Check applied coverage against the QC coverage bounds.

    Args:
        coverage: Applied cloud coverage.
        config: Active synthetic configuration.

    Returns:
        ``(ok, reason)``; ``reason`` empty when ``ok``.
    """
    qc = config.qc
    if coverage < qc.min_cloud_coverage:
        return False, f"coverage {coverage:.3f} < min"
    if coverage > qc.max_cloud_coverage:
        return False, f"coverage {coverage:.3f} > max"
    return True, ""
