"""Vegetation / water spectral indices and provisional surface classification.

All functions operate on reflectance tensors ``(B, 13, H, W)`` in ``[0, 1]`` with
the fixed band order from :mod:`s2audit`. Denominators are safeguarded: a
per-pixel validity mask marks where an index is numerically meaningful (the
denominator magnitude exceeds ``eps``), so downstream metrics can ignore
unstable pixels instead of averaging in garbage.

Indices (channel indices in parentheses):
    * NDVI  = (NIR - Red) / (NIR + Red)                 (7, 3)   canopy vigour
    * NDRE  = (NIR - RE1) / (NIR + RE1)                 (7, 4)   chlorophyll/N
    * EVI   = 2.5 (NIR-Red)/(NIR + 6Red - 7.5Blue + 1)  (7,3,1)  dense-canopy
    * NDWI  = (Green - NIR) / (Green + NIR)             (2, 7)   open water
    * NDMI  = (NIR - SWIR1) / (NIR + SWIR1)             (7, 11)  canopy moisture
    * NDSI  = (Green - SWIR1) / (Green + SWIR1)         (2, 11)  snow/cloud aid
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import BAND_INDEX

_B = BAND_INDEX
BLUE, GREEN, RED, RE1, NIR, CIRRUS, SWIR1, SWIR2 = (
    _B["B02"], _B["B03"], _B["B04"], _B["B05"], _B["B08"], _B["B10"], _B["B11"], _B["B12"])


@dataclass
class Index:
    """A computed index with a per-pixel validity mask.

    Attributes:
        value: The index value ``(B, 1, H, W)`` (0 where invalid).
        valid: Boolean validity ``(B, 1, H, W)`` (denominator well-conditioned).
    """

    value: torch.Tensor
    valid: torch.Tensor


def _normalized_difference(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-6,
                           denom_floor: float = 2e-3) -> Index:
    """Safeguarded normalized difference ``(a - b) / (a + b)``.

    Args:
        a: First band ``(B, 1, H, W)``.
        b: Second band ``(B, 1, H, W)``.
        eps: Additive stabiliser preventing division by zero.
        denom_floor: Pixels whose ``|a + b|`` is below this are marked invalid
            (both bands near zero -> the ratio is dominated by noise).

    Returns:
        The :class:`Index`.
    """
    denom = a + b
    valid = denom.abs() >= denom_floor
    value = torch.where(valid, (a - b) / (denom + eps), torch.zeros_like(denom))
    # A normalized difference is physically in [-1, 1]; values outside it only
    # arise from unphysical (negative) predicted reflectance. Clamp so the index
    # error stays bounded and interpretable (the unphysical output is surfaced
    # separately by the reflectance-range diagnostic).
    return Index(value.clamp(-1.0, 1.0), valid)


def _band(x: torch.Tensor, idx: int) -> torch.Tensor:
    """Return channel ``idx`` as ``(B, 1, H, W)``."""
    return x[:, idx:idx + 1]


def ndvi(x: torch.Tensor) -> Index:
    """Normalized Difference Vegetation Index."""
    return _normalized_difference(_band(x, NIR), _band(x, RED))


def ndre(x: torch.Tensor) -> Index:
    """Normalized Difference Red-Edge index (B08, B05)."""
    return _normalized_difference(_band(x, NIR), _band(x, RE1))


def ndwi(x: torch.Tensor) -> Index:
    """McFeeters NDWI (open water; Green, NIR). Positive over water."""
    return _normalized_difference(_band(x, GREEN), _band(x, NIR))


def ndmi(x: torch.Tensor) -> Index:
    """Normalized Difference Moisture Index (B08, B11)."""
    return _normalized_difference(_band(x, NIR), _band(x, SWIR1))


def ndsi(x: torch.Tensor) -> Index:
    """Normalized Difference Snow Index (Green, SWIR1)."""
    return _normalized_difference(_band(x, GREEN), _band(x, SWIR1))


def evi(x: torch.Tensor, *, eps: float = 1e-6, denom_floor: float = 2e-3) -> Index:
    """Enhanced Vegetation Index (reflectance form, L=1, C1=6, C2=7.5, G=2.5).

    Args:
        x: Reflectance ``(B, 13, H, W)``.
        eps: Additive stabiliser.
        denom_floor: Validity floor on ``|denominator|``.

    Returns:
        The :class:`Index` (clamped to a physical ``[-1, 1]`` range).
    """
    nir, red, blue = _band(x, NIR), _band(x, RED), _band(x, BLUE)
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    valid = denom.abs() >= denom_floor
    value = torch.where(valid, 2.5 * (nir - red) / (denom + eps), torch.zeros_like(denom))
    return Index(value.clamp(-1.0, 1.0), valid)


#: The operational index registry used by the vegetation-metrics report.
VEGETATION_INDICES = {"ndvi": ndvi, "ndre": ndre, "evi": evi, "ndwi": ndwi, "ndmi": ndmi}


# --------------------------------------------------------------------------- #
# Provisional surface / validity classification (no external land mask needed) #
# --------------------------------------------------------------------------- #

def background_mask(x: torch.Tensor, *, threshold: float = 1.5e-3) -> torch.Tensor:
    """Boolean mask of NoData / background pixels (all bands near zero).

    Args:
        x: Reflectance ``(B, 13, H, W)``.
        threshold: Reflectance below which every band counts as background.

    Returns:
        Boolean ``(B, 1, H, W)`` (True = background / NoData).
    """
    return (x < threshold).all(dim=1, keepdim=True)


def water_mask(x: torch.Tensor, *, ndwi_threshold: float = 0.0) -> torch.Tensor:
    """Boolean open-water mask from NDWI on (clean) reflectance.

    Args:
        x: Reflectance ``(B, 13, H, W)`` (use the clean ground truth).
        ndwi_threshold: NDWI above which a valid pixel is classified water.

    Returns:
        Boolean ``(B, 1, H, W)`` (True = water).
    """
    w = ndwi(x)
    bg = background_mask(x)
    return (w.value > ndwi_threshold) & w.valid & (~bg)


def land_mask(x: torch.Tensor, *, ndwi_threshold: float = 0.0) -> torch.Tensor:
    """Boolean land mask = valid, non-background, non-water pixels.

    Args:
        x: Reflectance ``(B, 13, H, W)`` (clean ground truth).
        ndwi_threshold: NDWI threshold separating water from land.

    Returns:
        Boolean ``(B, 1, H, W)`` (True = land).
    """
    bg = background_mask(x)
    water = water_mask(x, ndwi_threshold=ndwi_threshold)
    return (~bg) & (~water)


def surface_category(x: torch.Tensor, *, land_dominant: float = 0.7,
                     ocean_dominant: float = 0.3) -> list[str]:
    """Classify each patch as land-dominant / mixed / ocean-dominant.

    The fraction is over *valid* (non-background) pixels, so a patch that is
    mostly NoData is judged on the surface it actually shows.

    Args:
        x: Reflectance ``(B, 13, H, W)`` (clean ground truth).
        land_dominant: Land-fraction at/above which a patch is land-dominant.
        ocean_dominant: Land-fraction at/below which a patch is ocean-dominant.

    Returns:
        A list of category strings, one per batch item.
    """
    bg = background_mask(x)
    land = land_mask(x)
    valid = (~bg).flatten(1).sum(dim=1).clamp(min=1)
    land_frac = land.flatten(1).sum(dim=1).float() / valid.float()
    out = []
    for f in land_frac.tolist():
        if f >= land_dominant:
            out.append("land_dominant")
        elif f <= ocean_dominant:
            out.append("ocean_dominant")
        else:
            out.append("mixed")
    return out
