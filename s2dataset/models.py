"""Data models for the dataset builder (picklable for parallel processing)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SampleSpec:
    """A target acquisition and its resolved reference set, ready to process.

    Attributes:
        target_date: Target acquisition date as ``YYYYMMDD``.
        target_stack: Path to the target's 13-band stack.
        target_mask: Path to the target's cloud mask.
        reference_dates: Reference acquisition dates as ``YYYYMMDD`` (ordered).
        reference_stacks: Reference stack paths, aligned to ``reference_dates``.
        reference_masks: Reference cloud-mask paths, aligned to
            ``reference_dates`` (``None`` where a mask is unavailable).
        split: Assigned split (``"train"`` / ``"val"`` / ``"test"``).
        cloud_percentage: Whole-image cloud percentage of the target.
        season: Target season label.
        year: Target year.
        month: Target month.
        day_of_year: Target ordinal day of year.
    """

    target_date: str
    target_stack: Path
    target_mask: Path
    reference_dates: list[str]
    reference_stacks: list[Path]
    reference_masks: list[Path | None] = field(default_factory=list)
    split: str = "train"
    cloud_percentage: float | None = None
    season: str | None = None
    year: int | None = None
    month: int | None = None
    day_of_year: int | None = None


@dataclass(slots=True)
class AlignmentReport:
    """Result of verifying geospatial alignment across a sample's rasters.

    Attributes:
        aligned: Whether every raster shares CRS, transform, resolution and shape.
        width: Common raster width (when aligned).
        height: Common raster height (when aligned).
        crs: Common CRS authority string (when aligned).
        issues: Human-readable descriptions of any mismatches.
    """

    aligned: bool
    width: int | None = None
    height: int | None = None
    crs: str | None = None
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PatchRecord:
    """Index record describing one written patch sample.

    Attributes:
        sample_id: Zero-padded global sample id (e.g. ``"sample_000001"``).
        split: Assigned split.
        target_date: Target date as ``YYYYMMDD``.
        reference_dates: Reference dates used.
        patch_index: Index of the patch within its target image.
        row: Patch top row (pixels, in the source raster).
        col: Patch left column (pixels, in the source raster).
        patch_size: Patch side length in pixels.
        cloud_fraction: Cloud fraction within the patch ``[0, 1]``.
        nodata_fraction: NoData fraction within the patch ``[0, 1]``.
        valid_fraction: Valid-pixel fraction within the patch ``[0, 1]``.
        geotiff_dir: Output GeoTIFF folder path (or empty).
        npz_path: Output NPZ path (or empty).
    """

    sample_id: str
    split: str
    target_date: str
    reference_dates: list[str]
    patch_index: int
    row: int
    col: int
    patch_size: int
    cloud_fraction: float
    nodata_fraction: float
    valid_fraction: float
    geotiff_dir: str = ""
    npz_path: str = ""

    def to_row(self) -> dict[str, Any]:
        """Return a flat CSV-friendly representation."""
        data = asdict(self)
        data["reference_dates"] = ";".join(self.reference_dates)
        return data


@dataclass(slots=True)
class SampleOutcome:
    """Aggregate outcome of processing one target acquisition.

    Attributes:
        target_date: Target date as ``YYYYMMDD``.
        split: Assigned split.
        status: ``"processed"``, ``"skipped"``, ``"aborted"`` or ``"failed"``.
        patches_written: Number of patches written for this target.
        patches_examined: Number of candidate patches considered.
        records: Per-patch index records produced.
        message: Status/error detail.
        duration_sec: Wall-clock seconds taken.
    """

    target_date: str
    split: str
    status: str
    patches_written: int = 0
    patches_examined: int = 0
    records: list[PatchRecord] = field(default_factory=list)
    message: str = ""
    duration_sec: float | None = None


@dataclass(slots=True)
class PatchSample:
    """In-memory patch arrays for one sample (used by writers).

    Attributes:
        target: Target stack patch, shape ``(13, H, W)``.
        mask: Cloud-mask patch, shape ``(1, H, W)``.
        references: Reference patches, shape ``(N, 13, H, W)``.
        metadata: Sample metadata dictionary.
    """

    target: Any  # numpy.ndarray (13, H, W)
    mask: Any  # numpy.ndarray (1, H, W)
    references: Any  # numpy.ndarray (N, 13, H, W)
    metadata: dict[str, Any]
