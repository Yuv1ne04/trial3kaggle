"""Output writers for the two dataset formats and per-sample metadata.

Format 1 — a georeferenced GeoTIFF folder per sample (target, mask, references,
metadata.json). Format 2 — a single compressed ``.npz`` per sample for fast
PyTorch loading.

The NPZ is written to a temp file then atomically replaced. The GeoTIFF folder
is written in place (a ``.complete`` sentinel marks success), avoiding fragile
Windows directory renames that fail under antivirus/search-indexer file locks.
Replacements retry briefly to ride out transient locks on indexed drives.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from .config import DatasetConfig

#: Sentinel filename marking a fully-written GeoTIFF sample directory.
COMPLETE_MARKER = ".complete"


def _robust_replace(src: Path, dst: Path, *, attempts: int = 6, delay: float = 0.25) -> None:
    """Atomically replace ``dst`` with ``src``, retrying on transient locks.

    Windows raises ``PermissionError`` (WinError 5/32) when antivirus or the
    search indexer momentarily holds a handle on a freshly written file. A short
    exponential backoff rides these out instead of failing the whole sample.

    Args:
        src: Source path (the temp file).
        dst: Destination path.
        attempts: Maximum number of tries.
        delay: Base delay (seconds), multiplied by the attempt number.

    Raises:
        PermissionError: If every attempt fails.
    """
    for attempt in range(1, attempts + 1):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay * attempt)


def patch_geotiff_profile(
    crs: str,
    transform: tuple[float, ...],
    count: int,
    dtype: str,
    size: int,
    config: DatasetConfig,
    *,
    nodata: float | None,
) -> dict[str, Any]:
    """Build a creation profile for a single georeferenced patch GeoTIFF.

    Args:
        crs: CRS authority string (e.g. ``"EPSG:32740"``).
        transform: 6-element affine transform of the patch window.
        count: Number of bands.
        dtype: Pixel data type.
        size: Patch side length in pixels.
        config: Active dataset configuration (compression).
        nodata: NoData value, or ``None``.

    Returns:
        A Rasterio creation profile dictionary.
    """
    return {
        "driver": "GTiff",
        "width": size,
        "height": size,
        "count": count,
        "dtype": dtype,
        "crs": CRS.from_string(crs) if crs else None,
        "transform": Affine(*transform),
        "nodata": nodata,
        "compress": config.compress,
        "tiled": True,
        "blockxsize": min(256, size),
        "blockysize": min(256, size),
    }


class SampleWriter:
    """Writes one patch sample in the configured output format(s).

    The writer targets a single scale's output subtree; the orchestrator
    creates one writer per patch size.
    """

    def __init__(self, config: DatasetConfig, base_dir: Path) -> None:
        """Initialise the writer.

        Args:
            config: Active dataset configuration.
            base_dir: Output root for this scale (e.g.
                ``<output_dir>/patches_256``). Splits and ``*_npz`` folders are
                created beneath it.
        """
        self.config = config
        self.base_dir = base_dir

    def write(
        self,
        sample_id: str,
        split: str,
        target: np.ndarray,
        mask: np.ndarray,
        references: np.ndarray,
        metadata: dict[str, Any],
    ) -> tuple[str, str]:
        """Write a sample in all enabled formats.

        Args:
            sample_id: Unique sample identifier (folder/file stem).
            split: Split name (``train`` / ``val`` / ``test``).
            target: Target array, shape ``(13, H, W)``.
            mask: Mask array, shape ``(1, H, W)``.
            references: Reference array, shape ``(N, 13, H, W)``.
            metadata: Sample metadata dictionary.

        Returns:
            A tuple ``(geotiff_dir, npz_path)``; entries are empty strings when
            the corresponding format is disabled.
        """
        geotiff_dir = ""
        npz_path = ""
        if self.config.write_geotiff:
            geotiff_dir = str(
                self._write_geotiff(sample_id, split, target, mask, references, metadata)
            )
        if self.config.write_npz:
            npz_path = str(
                self._write_npz(sample_id, split, target, mask, references, metadata)
            )
        return geotiff_dir, npz_path

    def _write_geotiff(
        self,
        sample_id: str,
        split: str,
        target: np.ndarray,
        mask: np.ndarray,
        references: np.ndarray,
        metadata: dict[str, Any],
    ) -> Path:
        """Write the GeoTIFF-folder format for one sample.

        Args:
            sample_id: Unique sample identifier.
            split: Split name.
            target: Target array ``(13, H, W)``.
            mask: Mask array ``(1, H, W)``.
            references: Reference array ``(N, 13, H, W)``.
            metadata: Sample metadata.

        Returns:
            The sample's GeoTIFF directory.
        """
        out_dir = self.base_dir / split / sample_id
        # Write in place rather than renaming a directory: directory renames
        # fail intermittently on Windows when an indexer/AV holds a file handle.
        # A trailing ``.complete`` marker signals a fully-written sample.
        marker = out_dir / COMPLETE_MARKER
        if out_dir.exists():
            marker.unlink(missing_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        crs = metadata["crs"]
        transform = tuple(metadata["transform"])
        size = int(target.shape[-1])

        target_profile = patch_geotiff_profile(
            crs, transform, 13, str(target.dtype), size, self.config,
            nodata=self.config.stack_nodata,
        )
        with rasterio.open(out_dir / "target.tif", "w", **target_profile) as dst:
            dst.write(target)

        mask_profile = patch_geotiff_profile(
            crs, transform, 1, str(mask.dtype), size, self.config,
            nodata=self.config.mask_nodata_value,
        )
        with rasterio.open(out_dir / "mask.tif", "w", **mask_profile) as dst:
            dst.write(mask)

        for i in range(references.shape[0]):
            ref_profile = patch_geotiff_profile(
                crs, transform, 13, str(references.dtype), size, self.config,
                nodata=self.config.stack_nodata,
            )
            with rasterio.open(
                out_dir / f"reference_{i + 1}.tif", "w", **ref_profile
            ) as dst:
                dst.write(references[i])

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        marker.write_text("ok", encoding="utf-8")
        return out_dir

    def _write_npz(
        self,
        sample_id: str,
        split: str,
        target: np.ndarray,
        mask: np.ndarray,
        references: np.ndarray,
        metadata: dict[str, Any],
    ) -> Path:
        """Write the PyTorch-ready NPZ format for one sample.

        Metadata is stored as a JSON string array so the file loads without
        ``allow_pickle``.

        Args:
            sample_id: Unique sample identifier.
            split: Split name.
            target: Target array ``(13, H, W)``.
            mask: Mask array ``(1, H, W)``.
            references: Reference array ``(N, 13, H, W)``.
            metadata: Sample metadata.

        Returns:
            The written NPZ path.
        """
        out_dir = self.base_dir / f"{split}_npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{sample_id}.npz"
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                target=target,
                mask=mask,
                references=references,
                metadata=np.array(json.dumps(metadata)),
            )
        _robust_replace(tmp, path)
        return path
