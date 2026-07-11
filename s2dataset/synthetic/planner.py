"""Stage 1: discover clear ground-truth patches per acquisition (windowed)."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

from ..alignment import verify_alignment
from ..logging_setup import get_logger
from ..models import SampleSpec
from ..patching import generate_grid
from ..shared.extract import read_window
from .config import SyntheticConfig
from .models import GroundTruthPatch

logger = get_logger()


def _crs_string(ds: "rasterio.io.DatasetReader") -> str | None:
    """Return a dataset CRS as ``AUTHORITY:CODE`` or its string form."""
    if ds.crs is None:
        return None
    auth = ds.crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else ds.crs.to_string()


def plan_clear_patches(
    spec: SampleSpec, config: SyntheticConfig
) -> list[GroundTruthPatch]:
    """Find every clear cell of one acquisition usable as ground truth.

    A cell qualifies if its native cloud, NoData and background fractions are
    below the clear-filter thresholds and enough valid pixels remain. References
    are verified aligned once for the whole acquisition; if they are not aligned
    (e.g. a wrong-tile reference), the acquisition is skipped.

    Args:
        spec: The acquisition's sample specification (with references/calendar).
        config: Active synthetic configuration.

    Returns:
        The clear ground-truth patches (possibly empty).
    """
    alignment = verify_alignment(spec)
    if not alignment.aligned:
        logger.warning("Skipping %s: references misaligned (%s)",
                       spec.target_date, "; ".join(alignment.issues[:2]))
        return []
    if len(spec.reference_dates) < config.min_references:
        logger.debug("Skipping %s: only %d reference(s)",
                     spec.target_date, len(spec.reference_dates))
        return []

    cf = config.clear_filter
    size = config.patch_size
    patches: list[GroundTruthPatch] = []
    try:
        with rasterio.open(spec.target_stack) as tds, rasterio.open(spec.target_mask) as mds:
            crs = _crs_string(tds)
            grid = generate_grid(tds.width, tds.height, size, size, drop_partial=True)
            for cell_index, (row, col) in enumerate(grid):
                target = read_window(tds, row, col, size)
                mask = read_window(mds, row, col, size, band=1)

                total = float(mask.size)
                nodata = (target == config.stack_nodata).all(axis=0)
                background = (target <= config.background_reflectance).all(axis=0)
                valid_frac = 1.0 - float(nodata.sum()) / total
                cloud_frac = float((mask == config.mask_cloud_value).sum()) / total
                bg_frac = float(background.sum()) / total

                if (cloud_frac > cf.max_cloud_fraction
                        or (1.0 - valid_frac) > cf.max_nodata_fraction
                        or bg_frac > cf.max_background_fraction
                        or valid_frac < cf.min_valid_fraction):
                    continue

                transform = tuple(window_transform(
                    Window(col, row, size, size), tds.transform))[:6]
                patches.append(GroundTruthPatch(
                    date=spec.target_date, cell_index=cell_index, row=row, col=col,
                    size=size, stack_path=str(spec.target_stack),
                    reference_dates=list(spec.reference_dates),
                    reference_stacks=[str(p) for p in spec.reference_stacks],
                    native_cloud_fraction=round(cloud_frac, 6),
                    nodata_fraction=round(1.0 - valid_frac, 6),
                    background_fraction=round(bg_frac, 6),
                    season=spec.season, year=spec.year, month=spec.month,
                    day_of_year=spec.day_of_year, crs=crs, transform=list(transform),
                ))
    except Exception as exc:  # noqa: BLE001 - per-acquisition guard
        logger.exception("Failed planning clear patches for %s", spec.target_date)
        return []
    return patches
