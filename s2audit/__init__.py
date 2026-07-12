"""Baseline Scientific Audit & Evaluation for the S2 cloud-reconstruction system.

This package answers a single question before any advanced model development:
*are the current baseline metrics scientifically trustworthy for operational
sugar-cane monitoring in Mauritius?*

It never retrains, never mutates the checkpoint, and never regenerates the
dataset. Everything operates by streaming the existing manifests / libraries and
writing a self-contained ``audit/`` report tree. The modules are import-safe
without heavy optional dependencies (matplotlib / s2cloudless are imported
lazily) so the metric and manifest logic can be unit-tested anywhere.

Band order (fixed, 0-indexed) throughout::

    0:B01 1:B02 2:B03 3:B04 4:B05 5:B06 6:B07 7:B08 8:B8A 9:B09 10:B10 11:B11 12:B12
       aero blue green red  re1  re2  re3  nir  re4  vap  cir  sw1  sw2
"""

from __future__ import annotations

__all__ = ["BAND_ORDER", "BAND_INDEX"]

#: Fixed Sentinel-2 band order produced by ``s2stacker``.
BAND_ORDER: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

#: Name -> channel-index lookup.
BAND_INDEX: dict[str, int] = {name: i for i, name in enumerate(BAND_ORDER)}
