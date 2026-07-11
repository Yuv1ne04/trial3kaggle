"""Per-target planning: filter cells, write target/mask patches, emit plans.

Reuses the existing alignment, grid and filtering logic. For each kept target
cell this writes the target's *image* (target library) and *mask* (mask library)
patches and emits a :class:`TargetPlan` carrying the ranked *candidate*
references. Reference quality is evaluated — and reference patches materialised —
later, in the builder's reference pass, so each reference is written once.
"""

from __future__ import annotations

import time

import rasterio
from rasterio.windows import Window, transform as window_transform

from ..alignment import verify_alignment
from ..config import DatasetConfig
from ..logging_setup import get_logger
from ..models import SampleSpec
from ..patching import compute_metrics, generate_grid, passes_filters
from .extract import patch_metadata, read_window, write_mask, write_target_image
from .models import PatchKey, PlanResult, TargetPlan

logger = get_logger()


def _crs_string(ds: "rasterio.io.DatasetReader") -> str | None:
    """Return a dataset CRS as ``AUTHORITY:CODE`` or its string form."""
    if ds.crs is None:
        return None
    auth = ds.crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else ds.crs.to_string()


def plan_target(spec: SampleSpec, config: DatasetConfig) -> PlanResult:
    """Plan one target across all scales, writing its target/mask patches.

    Never raises for per-target problems; alignment failures and errors are
    reported in the result so one bad target cannot abort the build.

    Args:
        spec: The target's sample specification.
        config: Active dataset configuration.

    Returns:
        A :class:`PlanResult` with the target's plans and write counts.
    """
    start = time.perf_counter()
    result = PlanResult(target_date=spec.target_date, status="failed")
    root = config.output_dir

    alignment = verify_alignment(spec)
    if not alignment.aligned:
        result.status = "aborted"
        result.message = "Alignment failed: " + "; ".join(alignment.issues)
        logger.error("Aborting %s: %s", spec.target_date, result.message)
        result.duration_sec = round(time.perf_counter() - start, 3)
        return result

    try:
        with rasterio.open(spec.target_stack) as tds, rasterio.open(spec.target_mask) as mds:
            crs = _crs_string(tds)
            for scale in config.patch_scales():
                grid = generate_grid(
                    tds.width, tds.height, scale.size, scale.stride,
                    drop_partial=scale.drop_partial,
                )
                for cell_index, (row, col) in enumerate(grid):
                    if _plan_cell(spec, config, root, tds, mds, crs,
                                  scale.size, cell_index, row, col, result):
                        result.image_patches_written += 1
        result.status = "planned"
        result.message = f"{len(result.plans)} cell(s) planned"
    except Exception as exc:  # noqa: BLE001 - per-target guard
        result.status = "failed"
        result.message = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed planning %s", spec.target_date)

    result.duration_sec = round(time.perf_counter() - start, 3)
    return result


def _plan_cell(
    spec: SampleSpec,
    config: DatasetConfig,
    root,
    tds: "rasterio.io.DatasetReader",
    mds: "rasterio.io.DatasetReader",
    crs: str | None,
    size: int,
    cell_index: int,
    row: int,
    col: int,
    result: PlanResult,
) -> bool:
    """Filter one cell and, if kept, write target/mask patches and a plan.

    Args:
        spec: The target's sample specification.
        config: Active dataset configuration.
        root: Dataset root directory.
        tds: Open target stack dataset.
        mds: Open mask dataset.
        crs: CRS authority string.
        size: Patch size for the current scale.
        cell_index: Cell index within the grid.
        row: Patch top row (pixels).
        col: Patch left column (pixels).
        result: Accumulating plan result (mutated in place).

    Returns:
        ``True`` if a new target image patch file was written.
    """
    target_patch = read_window(tds, row, col, size)
    mask_patch = read_window(mds, row, col, size, band=1)

    metrics = compute_metrics(target_patch, mask_patch, config)
    if not passes_filters(metrics, config):
        return False

    transform = tuple(window_transform(Window(col, row, size, size), tds.transform))[:6]
    key = PatchKey(size, spec.target_date, cell_index)
    meta = patch_metadata(key, row, col, crs, transform)

    written = write_target_image(root, key, target_patch, meta)
    if write_mask(root, key, mask_patch[None, :, :], meta):
        result.mask_patches_written += 1

    result.plans.append(TargetPlan(
        target_date=spec.target_date,
        split=spec.split,
        size=size,
        cell_index=cell_index,
        row=row,
        col=col,
        candidate_dates=list(spec.reference_dates),
        cloud_fraction=metrics.cloud_fraction,
        nodata_fraction=metrics.nodata_fraction,
        valid_fraction=metrics.valid_fraction,
        cloud_percentage=spec.cloud_percentage,
        season=spec.season,
        year=spec.year,
        month=spec.month,
        day_of_year=spec.day_of_year,
        crs=crs,
        transform=list(transform),
    ))
    return written
