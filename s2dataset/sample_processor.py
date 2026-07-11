"""Per-target sample processing: windowed patch extraction and writing.

``process_target`` is a self-contained, picklable worker. It opens each raster
once and reads only patch-sized windows, so peak memory is bounded by the patch
size regardless of the 10980x10980 source tiles. Reference windows are read only
for patches that pass filtering.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

from .alignment import verify_alignment
from .config import EXPECTED_BANDS, DatasetConfig
from .logging_setup import get_logger
from .models import PatchRecord, SampleOutcome, SampleSpec
from .patching import compute_metrics, generate_grid, passes_filters
from .writers import SampleWriter

logger = get_logger()


def process_target(spec: SampleSpec, config: DatasetConfig) -> SampleOutcome:
    """Extract, filter, validate and write all patches for one target.

    Never raises for per-target problems; failures and alignment aborts are
    captured in the returned outcome so one bad acquisition cannot abort a run.

    Args:
        spec: The sample specification to process.
        config: Active dataset configuration.

    Returns:
        A populated :class:`SampleOutcome`.
    """
    start = time.perf_counter()
    outcome = SampleOutcome(
        target_date=spec.target_date, split=spec.split, status="failed"
    )

    alignment = verify_alignment(spec)
    if not alignment.aligned:
        outcome.status = "aborted"
        outcome.message = "Alignment failed: " + "; ".join(alignment.issues)
        logger.error("Aborting %s: %s", spec.target_date, outcome.message)
        outcome.duration_sec = round(time.perf_counter() - start, 3)
        return outcome

    scales = config.patch_scales()

    try:
        target_ds = rasterio.open(spec.target_stack)
        mask_ds = rasterio.open(spec.target_mask)
        ref_ds = [rasterio.open(p) for p in spec.reference_stacks]
        try:
            # The rasters are opened once; every scale reads its own windows
            # from them, so no full image is ever loaded into memory.
            for scale in scales:
                writer = SampleWriter(config, config.scale_output_dir(scale.size))
                grid = generate_grid(
                    target_ds.width, target_ds.height, scale.size, scale.stride,
                    drop_partial=scale.drop_partial,
                )
                outcome.patches_examined += len(grid)
                for patch_index, (row, col) in enumerate(grid):
                    record = _process_patch(
                        spec, config, writer, scale.size, patch_index, row, col,
                        target_ds, mask_ds, ref_ds,
                    )
                    if record is not None:
                        outcome.records.append(record)
            outcome.patches_written = len(outcome.records)
            outcome.status = "processed"
            outcome.message = (
                f"{outcome.patches_written}/{outcome.patches_examined} patches kept "
                f"across {len(scales)} scale(s)"
            )
        finally:
            target_ds.close()
            mask_ds.close()
            for ds in ref_ds:
                ds.close()
    except Exception as exc:  # noqa: BLE001 - per-target guard
        outcome.status = "failed"
        outcome.message = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed processing %s", spec.target_date)

    outcome.duration_sec = round(time.perf_counter() - start, 3)
    return outcome


def _process_patch(
    spec: SampleSpec,
    config: DatasetConfig,
    writer: SampleWriter,
    size: int,
    patch_index: int,
    row: int,
    col: int,
    target_ds: "rasterio.io.DatasetReader",
    mask_ds: "rasterio.io.DatasetReader",
    ref_ds: list["rasterio.io.DatasetReader"],
) -> PatchRecord | None:
    """Read, filter, validate and (if kept) write a single patch.

    Args:
        spec: The owning sample specification.
        config: Active dataset configuration.
        writer: The sample writer for this scale's output subtree.
        size: Patch side length for the current scale.
        patch_index: Index of this patch within the target image (per scale).
        row: Patch top row (pixels).
        col: Patch left column (pixels).
        target_ds: Open target dataset.
        mask_ds: Open mask dataset.
        ref_ds: Open reference datasets.

    Returns:
        A :class:`PatchRecord` if the patch was written, else ``None``.
    """
    window = Window(col, row, size, size)

    target_patch = target_ds.read(window=window)  # (13, H, W)
    mask_patch = mask_ds.read(1, window=window)  # (H, W)

    metrics = compute_metrics(target_patch, mask_patch, config)
    if not passes_filters(metrics, config):
        return None

    references = np.stack([ds.read(window=window) for ds in ref_ds], axis=0)

    if not _validate_arrays(target_patch, mask_patch, references, size, config):
        logger.error(
            "Validation failed for %s patch %d (size %d) at (%d,%d); skipping",
            spec.target_date, patch_index, size, row, col,
        )
        return None

    mask_out = mask_patch[np.newaxis, :, :]
    patch_tf = tuple(window_transform(window, target_ds.transform))[:6]
    crs = _crs_string(target_ds)
    metadata = _build_metadata(
        spec, config, size, patch_index, row, col, metrics, crs, patch_tf
    )

    sample_id = f"sample_{spec.target_date}_s{size}_{patch_index:04d}"
    geotiff_dir, npz_path = writer.write(
        sample_id, spec.split, target_patch, mask_out, references, metadata
    )

    return PatchRecord(
        sample_id=sample_id,
        split=spec.split,
        target_date=spec.target_date,
        reference_dates=list(spec.reference_dates),
        patch_index=patch_index,
        row=row,
        col=col,
        patch_size=size,
        cloud_fraction=round(metrics.cloud_fraction, 6),
        nodata_fraction=round(metrics.nodata_fraction, 6),
        valid_fraction=round(metrics.valid_fraction, 6),
        geotiff_dir=geotiff_dir,
        npz_path=npz_path,
    )


def _validate_arrays(
    target: np.ndarray,
    mask: np.ndarray,
    references: np.ndarray,
    size: int,
    config: DatasetConfig,
) -> bool:
    """Validate patch array shapes and band counts before writing.

    Args:
        target: Target patch ``(13, H, W)``.
        mask: Mask patch ``(H, W)``.
        references: Reference patches ``(N, 13, H, W)``.
        size: Expected patch side length.
        config: Active dataset configuration.

    Returns:
        ``True`` if every shape/band-count expectation holds.
    """
    return (
        target.shape == (EXPECTED_BANDS, size, size)
        and mask.shape == (size, size)
        and references.shape == (config.n_references, EXPECTED_BANDS, size, size)
    )


def _build_metadata(
    spec: SampleSpec,
    config: DatasetConfig,
    size: int,
    patch_index: int,
    row: int,
    col: int,
    metrics: Any,
    crs: str | None,
    transform: tuple[float, ...],
) -> dict[str, Any]:
    """Assemble the per-sample metadata dictionary.

    Args:
        spec: The owning sample specification.
        config: Active dataset configuration.
        size: Patch side length for the current scale.
        patch_index: Patch index within the target image (per scale).
        row: Patch top row (pixels).
        col: Patch left column (pixels).
        metrics: Computed patch metrics.
        crs: CRS authority string.
        transform: 6-element patch affine transform.

    Returns:
        The metadata dictionary written into each sample.
    """
    return {
        "target_date": spec.target_date,
        "reference_dates": list(spec.reference_dates),
        "cloud_percentage": spec.cloud_percentage,
        "patch_cloud_fraction": round(metrics.cloud_fraction, 6),
        "patch_nodata_fraction": round(metrics.nodata_fraction, 6),
        "patch_valid_fraction": round(metrics.valid_fraction, 6),
        "patch_coordinates": {"row": row, "col": col},
        "patch_index": patch_index,
        "patch_size": size,
        "crs": crs,
        "transform": list(transform),
        "season": spec.season,
        "year": spec.year,
        "month": spec.month,
        "day_of_year": spec.day_of_year,
        "n_references": config.n_references,
    }


def _crs_string(ds: "rasterio.io.DatasetReader") -> str | None:
    """Return a dataset's CRS as ``AUTHORITY:CODE`` or string form."""
    if ds.crs is None:
        return None
    auth = ds.crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else ds.crs.to_string()
