"""Data models for the shared-reference dataset (picklable for multiprocessing)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from . import ids


@dataclass(slots=True, frozen=True)
class PatchKey:
    """A grid cell on a given date at a given size — the unit of single-copy storage.

    The same ``(size, date, cell_index)`` always maps to the same library file,
    which is how duplication is eliminated.

    Attributes:
        size: Patch size in pixels.
        date: Acquisition date as ``YYYYMMDD``.
        cell_index: Cell index within the (size) grid.
    """

    size: int
    date: str
    cell_index: int

    def target_relpath(self) -> str:
        """Return the target-library relative path for this key."""
        return ids.target_relpath(self.size, self.date, self.cell_index)

    def reference_relpath(self) -> str:
        """Return the reference-library relative path for this key."""
        return ids.reference_relpath(self.size, self.date, self.cell_index)

    def mask_relpath(self) -> str:
        """Return the mask-library relative path for this key."""
        return ids.mask_relpath(self.size, self.date, self.cell_index)


def _iso(date: str) -> str:
    """Convert a ``YYYYMMDD`` date to ISO ``YYYY-MM-DD`` (or pass through)."""
    try:
        return datetime.strptime(date, "%Y%m%d").date().isoformat()
    except ValueError:
        return date


@dataclass(slots=True)
class SampleManifest:
    """A training sample expressed purely as references into the libraries.

    Attributes:
        target_date: Target acquisition date as ``YYYYMMDD``.
        split: Split label (``train`` / ``val`` / ``test``).
        size: Patch size in pixels.
        cell_index: Cell index within the (size) grid.
        row: Patch top row in the source raster (pixels).
        col: Patch left column in the source raster (pixels).
        reference_dates: Validated reference dates (``YYYYMMDD``), 2..maximum.
        cloud_fraction: Cloud fraction of the target patch ``[0, 1]``.
        nodata_fraction: NoData fraction of the target patch ``[0, 1]``.
        valid_fraction: Valid-pixel fraction of the target patch ``[0, 1]``.
        cloud_percentage: Whole-image target cloud percentage.
        season: Target season label.
        year: Target year.
        month: Target month.
        day_of_year: Target ordinal day of year.
        crs: CRS authority string.
        transform: 6-element patch affine transform.
    """

    target_date: str
    split: str
    size: int
    cell_index: int
    row: int
    col: int
    reference_dates: list[str]
    cloud_fraction: float
    nodata_fraction: float
    valid_fraction: float
    cloud_percentage: float | None
    season: str | None
    year: int | None
    month: int | None
    day_of_year: int | None
    crs: str | None
    transform: list[float]

    def target_key(self) -> PatchKey:
        """Return the :class:`PatchKey` of this sample's target patch."""
        return PatchKey(self.size, self.target_date, self.cell_index)

    def reference_keys(self) -> list[PatchKey]:
        """Return the :class:`PatchKey` list of this sample's references."""
        return [PatchKey(self.size, d, self.cell_index) for d in self.reference_dates]

    def to_sample_json(self, sample_id: str) -> dict[str, Any]:
        """Build the on-disk sample JSON payload (spec format).

        Args:
            sample_id: The sample's unique identifier.

        Returns:
            A dict with ``target`` / ``mask`` / ``references`` relative paths and
            a nested ``metadata`` object.
        """
        return {
            "target": self.target_key().target_relpath(),
            "mask": self.target_key().mask_relpath(),
            "references": [k.reference_relpath() for k in self.reference_keys()],
            "metadata": {
                "sample_id": sample_id,
                "split": self.split,
                "target_date": _iso(self.target_date),
                "target_date_compact": self.target_date,
                "reference_dates": [_iso(d) for d in self.reference_dates],
                "n_references": len(self.reference_dates),
                "cloud_percentage": self.cloud_percentage,
                "patch_size": self.size,
                "cell_index": self.cell_index,
                "coordinates": [self.row, self.col],
                "patch_cloud_fraction": round(self.cloud_fraction, 6),
                "patch_nodata_fraction": round(self.nodata_fraction, 6),
                "patch_valid_fraction": round(self.valid_fraction, 6),
                "season": self.season,
                "year": self.year,
                "month": self.month,
                "day_of_year": self.day_of_year,
                "crs": self.crs,
                "transform": list(self.transform),
            },
        }

    def to_record(self) -> dict[str, Any]:
        """Return a flat dict for the manifest sidecar (JSONL persistence)."""
        return asdict(self)


@dataclass(slots=True)
class TargetPlan:
    """A planned target patch with its *candidate* references (pre-validation).

    Attributes:
        manifest_fields: The target-side fields needed to build a manifest once
            references are validated (everything except ``reference_dates``).
        candidate_dates: Ranked candidate reference dates for this cell.
    """

    target_date: str
    split: str
    size: int
    cell_index: int
    row: int
    col: int
    candidate_dates: list[str]
    cloud_fraction: float
    nodata_fraction: float
    valid_fraction: float
    cloud_percentage: float | None
    season: str | None
    year: int | None
    month: int | None
    day_of_year: int | None
    crs: str | None
    transform: list[float]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for sidecar persistence."""
        return asdict(self)


@dataclass(slots=True)
class PlanResult:
    """Output of planning a single target across all configured scales.

    Attributes:
        target_date: Target acquisition date as ``YYYYMMDD``.
        status: ``"planned"``, ``"aborted"`` or ``"failed"``.
        plans: Target plans (one per kept cell per scale).
        image_patches_written: Target image patches newly written.
        mask_patches_written: Mask patches newly written.
        message: Status/error detail.
        duration_sec: Wall-clock seconds taken.
    """

    target_date: str
    status: str
    plans: list[TargetPlan] = field(default_factory=list)
    image_patches_written: int = 0
    mask_patches_written: int = 0
    message: str = ""
    duration_sec: float | None = None
