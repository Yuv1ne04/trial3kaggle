"""Geospatial alignment verification across a sample's rasters.

Every target, mask and reference for a sample must share CRS, geotransform,
resolution and pixel dimensions, otherwise windowed patch reads would not
correspond to the same ground location. This module checks that up front and
reports any inconsistency so the sample can be aborted gracefully.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.crs import CRS

from .config import EXPECTED_BANDS
from .models import AlignmentReport, SampleSpec

#: Tolerance (in CRS units) for comparing affine transform coefficients.
_TRANSFORM_TOL = 1e-6


def _crs_string(crs: CRS | None) -> str | None:
    """Return a CRS as ``AUTHORITY:CODE`` or its string form."""
    if crs is None:
        return None
    auth = crs.to_authority()
    return f"{auth[0]}:{auth[1]}" if auth else crs.to_string()


def verify_alignment(spec: SampleSpec) -> AlignmentReport:
    """Verify that all rasters of a sample are geospatially aligned.

    Opens only the headers (no pixel data) of the target stack, the mask and
    every reference stack, and checks CRS, transform, resolution and shape.

    Args:
        spec: The sample specification to verify.

    Returns:
        An :class:`AlignmentReport`; ``aligned`` is ``False`` with populated
        ``issues`` if any raster disagrees or cannot be opened.
    """
    issues: list[str] = []
    paths: list[tuple[str, Path]] = [
        ("target", spec.target_stack),
        ("mask", spec.target_mask),
    ]
    paths.extend(
        (f"reference[{i}] {d}", p)
        for i, (d, p) in enumerate(zip(spec.reference_dates, spec.reference_stacks))
    )

    reference_profile: dict[str, object] | None = None
    width = height = None
    crs_str: str | None = None

    for label, path in paths:
        try:
            with rasterio.open(path) as ds:
                profile = {
                    "crs": _crs_string(ds.crs),
                    "transform": tuple(ds.transform)[:6],
                    "width": ds.width,
                    "height": ds.height,
                    "res": (round(ds.res[0], 6), round(ds.res[1], 6)),
                    "count": ds.count,
                }
        except Exception as exc:  # noqa: BLE001 - report instead of raising
            issues.append(f"{label}: cannot open ({exc})")
            continue

        # Band-count expectations (mask is single-band; stacks are 13-band).
        if label == "mask":
            if profile["count"] < 1:
                issues.append(f"{label}: has no bands")
        elif profile["count"] != EXPECTED_BANDS:
            issues.append(
                f"{label}: expected {EXPECTED_BANDS} bands, found {profile['count']}"
            )

        if reference_profile is None:
            reference_profile = profile
            width, height = profile["width"], profile["height"]
            crs_str = profile["crs"]
            continue

        issues.extend(_compare(label, reference_profile, profile))

    return AlignmentReport(
        aligned=not issues,
        width=width,
        height=height,
        crs=crs_str,
        issues=issues,
    )


def _compare(
    label: str,
    expected: dict[str, object],
    actual: dict[str, object],
) -> list[str]:
    """Compare one raster's geospatial profile against the reference profile.

    Args:
        label: Human-readable label of the raster being checked.
        expected: The first raster's profile (the reference).
        actual: This raster's profile.

    Returns:
        A list of mismatch descriptions (empty if fully aligned).
    """
    problems: list[str] = []
    if actual["crs"] != expected["crs"]:
        problems.append(f"{label}: CRS {actual['crs']} != {expected['crs']}")
    if (actual["width"], actual["height"]) != (expected["width"], expected["height"]):
        problems.append(
            f"{label}: shape {actual['width']}x{actual['height']} != "
            f"{expected['width']}x{expected['height']}"
        )
    if actual["res"] != expected["res"]:
        problems.append(f"{label}: resolution {actual['res']} != {expected['res']}")
    if any(
        abs(a - b) > _TRANSFORM_TOL
        for a, b in zip(actual["transform"], expected["transform"])  # type: ignore[arg-type]
    ):
        problems.append(f"{label}: transform differs from target")
    return problems
