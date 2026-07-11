"""Data models for the storage-efficient synthetic supervision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from . import ids


def _iso(date: str) -> str:
    """Return ``date`` (``YYYYMMDD``) as ISO ``YYYY-MM-DD`` if possible."""
    try:
        return datetime.strptime(date, "%Y%m%d").date().isoformat()
    except ValueError:
        return date


@dataclass(slots=True)
class MaskEntry:
    """A real cloud-mask patch available for transplantation.

    Attributes:
        path: Absolute path to the source mask patch npz.
        date: Source acquisition date (``YYYYMMDD``).
        cell_index: Cell index within the grid.
        cloud_fraction: Cloud fraction of the mask patch ``[0, 1]``.
        season: Source season label.
        month: Source month (1-12).
        source_row: Row of the mask within its source acquisition (pixels).
        source_col: Column of the mask within its source acquisition (pixels).
    """

    path: str
    date: str
    cell_index: int
    cloud_fraction: float
    season: str | None
    month: int | None
    source_row: int = 0
    source_col: int = 0

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the entry."""
        return asdict(self)


@dataclass(slots=True)
class GroundTruthPatch:
    """A clear observed patch usable as supervised ground truth.

    Attributes:
        date: Acquisition date of the clear patch (``YYYYMMDD``).
        cell_index: Cell index within the grid.
        row: Patch top row in the source raster (pixels).
        col: Patch left column in the source raster (pixels).
        size: Patch side length in pixels.
        stack_path: Path to the clear patch's source stack.
        reference_dates: Available reference dates (``YYYYMMDD``).
        reference_stacks: Reference stack paths, aligned to ``reference_dates``.
        native_cloud_fraction: The clear patch's own (low) cloud fraction.
        nodata_fraction: NoData fraction of the clear patch.
        background_fraction: Ocean/background fraction of the clear patch.
        season: Season label.
        year: Year.
        month: Month (1-12).
        day_of_year: Ordinal day of year.
        split: Assigned split (``train`` / ``val`` / ``test``).
        crs: CRS authority string.
        transform: 6-element patch affine transform.
    """

    date: str
    cell_index: int
    row: int
    col: int
    size: int
    stack_path: str
    reference_dates: list[str]
    reference_stacks: list[str]
    native_cloud_fraction: float
    nodata_fraction: float
    background_fraction: float
    season: str | None
    year: int | None
    month: int | None
    day_of_year: int | None
    split: str = "train"
    crs: str | None = None
    transform: list[float] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for sidecar persistence."""
        return asdict(self)


@dataclass(slots=True)
class SyntheticManifest:
    """A synthetic training sample expressed purely as references (no pixels).

    Attributes:
        sample_id: Unique sample identifier.
        split: Split label.
        gt: The ground-truth patch.
        mask: The transplanted real cloud mask.
        difficulty: Curriculum band name.
        augmentation_index: Variant index within the GT patch.
        seed: Deterministic seed for this sample.
        applied_cloud_coverage: Cloud coverage of the transplanted mask ``[0,1]``.
    """

    sample_id: str
    split: str
    gt: GroundTruthPatch
    mask: MaskEntry
    difficulty: str
    augmentation_index: int
    seed: int
    applied_cloud_coverage: float

    def to_manifest_json(self, cloud_fill: str) -> dict[str, Any]:
        """Build the on-disk sample manifest (spec format, references only).

        Args:
            cloud_fill: The cloud-fill mode recorded for the loader.

        Returns:
            A JSON dict referencing library patches + a nested ``metadata`` block.
        """
        size = self.gt.size
        return {
            "ground_truth": ids.patch_relpath(size, self.gt.date, self.gt.cell_index),
            "cloud_tile": ids.cloud_tile_relpath(size, self.mask.date, self.mask.cell_index),
            "references": [
                ids.patch_relpath(size, d, self.gt.cell_index)
                for d in self.gt.reference_dates
            ],
            "metadata": {
                "sample_id": self.sample_id,
                "split": self.split,
                "target_date": _iso(self.gt.date),
                "ground_truth_date": _iso(self.gt.date),
                "applied_cloud_mask_date": _iso(self.mask.date),
                "cloud_percentage": round(self.applied_cloud_coverage * 100.0, 4),
                "applied_cloud_coverage": round(self.applied_cloud_coverage, 6),
                "difficulty": self.difficulty,
                "cloud_fill": cloud_fill,
                "reference_dates": [_iso(d) for d in self.gt.reference_dates],
                "n_references": len(self.gt.reference_dates),
                "patch_coordinates": [self.gt.row, self.gt.col],
                "cell_index": self.gt.cell_index,
                "patch_size": size,
                "random_seed": self.seed,
                "augmentation_index": self.augmentation_index,
                "season": self.gt.season,
                "year": self.gt.year,
                "month": self.gt.month,
                "day_of_year": self.gt.day_of_year,
                "crs": self.gt.crs,
                "transform": self.gt.transform,
                "native_gt_cloud_fraction": self.gt.native_cloud_fraction,
            },
        }


@dataclass(slots=True)
class GenOutcome:
    """Reporting record for one manifest (written or skipped).

    Attributes:
        status: ``"written"``, ``"rejected"`` or ``"failed"``.
        split: Assigned split.
        gt_date: Ground-truth date.
        mask_date: Applied mask date.
        cell_index: Cell index.
        augmentation_index: Variant index.
        difficulty: Curriculum band.
        applied_cloud_coverage: Coverage of the applied mask.
        n_references: Number of references.
        season: GT season.
        month: GT month.
        message: Detail.
    """

    status: str
    split: str
    gt_date: str
    mask_date: str
    cell_index: int
    augmentation_index: int
    difficulty: str
    applied_cloud_coverage: float | None = None
    n_references: int | None = None
    season: str | None = None
    month: int | None = None
    message: str = ""

    def to_row(self) -> dict[str, Any]:
        """Return a flat CSV-friendly representation."""
        return asdict(self)
