"""Configuration objects and YAML loading for the dataset builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: Number of bands expected in every stack (and therefore every target/reference).
EXPECTED_BANDS: int = 13


@dataclass(slots=True)
class PatchParams:
    """Patch-extraction parameters.

    Attributes:
        size: Square patch side length in pixels.
        stride: Step (pixels) between successive patch origins. Equal to
            ``size`` gives non-overlapping patches; smaller gives overlap.
        drop_partial: When ``True`` patches that would extend past the image
            edge are dropped; when ``False`` they are shifted inward to remain
            fully inside the raster.
    """

    size: int = 256
    stride: int = 256
    drop_partial: bool = True

    def __post_init__(self) -> None:
        """Validate patch parameters.

        Raises:
            ValueError: If size or stride is not positive.
        """
        if self.size <= 0:
            raise ValueError("patch size must be > 0")
        if self.stride <= 0:
            raise ValueError("patch stride must be > 0")


@dataclass(slots=True, frozen=True)
class PatchScale:
    """A single resolved patch scale (one entry of a multi-scale run).

    Attributes:
        size: Square patch side length in pixels.
        stride: Step (pixels) between successive patch origins.
        drop_partial: Edge handling (see :class:`PatchParams`).
    """

    size: int
    stride: int
    drop_partial: bool

    @property
    def name(self) -> str:
        """Return the directory-friendly scale name (e.g. ``"patches_256"``)."""
        return f"patches_{self.size}"


@dataclass(slots=True)
class FilterParams:
    """Patch-filtering thresholds. A patch is kept only if it passes all.

    Attributes:
        max_nodata_fraction: Maximum fraction of NoData pixels allowed.
        max_background_fraction: Maximum fraction of ocean/background pixels
            allowed (background detected as near-zero across all bands).
        min_cloud_fraction: Minimum target cloud fraction required (so samples
            actually contain something to reconstruct).
        max_cloud_fraction: Maximum target cloud fraction allowed (so samples
            retain enough clear context).
        min_valid_fraction: Minimum fraction of valid (non-NoData) pixels.
        background_reflectance: Per-band reflectance at/below which a pixel is
            considered background when all bands fall below it.
    """

    max_nodata_fraction: float = 0.30
    max_background_fraction: float = 0.50
    min_cloud_fraction: float = 0.05
    max_cloud_fraction: float = 0.90
    min_valid_fraction: float = 0.50
    background_reflectance: int = 1

    def __post_init__(self) -> None:
        """Validate that all fractions lie in ``[0, 1]``.

        Raises:
            ValueError: If any fraction is out of range.
        """
        for name in (
            "max_nodata_fraction",
            "max_background_fraction",
            "min_cloud_fraction",
            "max_cloud_fraction",
            "min_valid_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(slots=True)
class ReferenceParams:
    """Reference-selection policy for the shared-reference builder.

    Up to ``maximum`` references are attached per sample, but only those that
    are clear enough *at the patch* are kept; a sample needs at least
    ``minimum`` valid references or it is dropped.

    Attributes:
        maximum: Preferred/maximum references per sample (default 4).
        minimum: Minimum valid references required to emit a sample (default 2).
        max_cloud_fraction: A reference patch is rejected if its cloud fraction
            exceeds this (so references actually provide clear context).
        max_nodata_fraction: A reference patch is rejected if its NoData fraction
            exceeds this.
    """

    maximum: int = 4
    minimum: int = 2
    max_cloud_fraction: float = 0.20
    max_nodata_fraction: float = 0.30

    def __post_init__(self) -> None:
        """Validate the reference policy.

        Raises:
            ValueError: If counts are non-positive, ``minimum`` > ``maximum`` or
                fractions are out of ``[0, 1]``.
        """
        if self.minimum < 1 or self.maximum < 1:
            raise ValueError("reference minimum/maximum must be >= 1")
        if self.minimum > self.maximum:
            raise ValueError("reference minimum must be <= maximum")
        for name in ("max_cloud_fraction", "max_nodata_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(slots=True)
class SplitParams:
    """Train/validation/test split fractions (by acquisition date).

    Attributes:
        train: Fraction of dates assigned to training.
        val: Fraction assigned to validation.
        test: Fraction assigned to testing.
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        """Validate and normalise the split fractions.

        Raises:
            ValueError: If any fraction is negative or all are zero.
        """
        values = [self.train, self.val, self.test]
        if any(v < 0 for v in values):
            raise ValueError("split fractions must be non-negative")
        total = sum(values)
        if total <= 0:
            raise ValueError("split fractions must sum to a positive value")
        self.train /= total
        self.val /= total
        self.test /= total


@dataclass(slots=True)
class DatasetConfig:
    """Full configuration for a dataset-building run.

    Attributes:
        stacks_dir: Directory of 13-band stacks (``YYYYMMDD_stack.tif``).
        masks_dir: Directory of cloud masks (``YYYYMMDD_cloudmask.tif``).
        reference_database: Path to ``reference_database.json``.
        temporal_database: Path to ``temporal_database.csv``.
        cloud_statistics: Path to ``cloud_statistics.csv``.
        output_dir: Root output directory for the dataset.
        metadata_dir: Directory for index/statistics outputs.
        logs_dir: Directory for the log file.
        n_references: Number of references per sample.
        patch: Single-scale patch parameters (fallback when ``patch_sizes`` is
            not set; preserves backward compatibility).
        patch_sizes: Optional list of patch sizes for multi-scale generation.
            When set, the builder produces one output subtree per size and this
            takes precedence over ``patch.size``.
        stride: Global stride applied to every patch size; ``None`` falls back
            to ``stride_ratio``/per-size defaults.
        stride_ratio: Stride as a fraction of each patch size (e.g. ``0.5`` for
            50% overlap); used when ``strides`` and ``stride`` are unset.
        strides: Explicit per-size strides aligned with ``patch_sizes`` (highest
            precedence); ``None`` to derive strides from the rules above.
        drop_partial: Global edge handling for every scale; ``None`` falls back
            to ``patch.drop_partial``.
        filters: Patch-filtering thresholds.
        split: Train/val/test split fractions.
        write_geotiff: Whether to emit the GeoTIFF folder format.
        write_npz: Whether to emit the PyTorch ``.npz`` format.
        compress: GDAL compression for GeoTIFF outputs.
        num_workers: Parallel worker processes (one target acquisition each).
        mask_cloud_value: Mask value denoting cloud.
        mask_nodata_value: Mask value denoting NoData/invalid.
        stack_nodata: NoData value of the stacks (for valid-pixel detection).
        resume: Skip targets already recorded in the checkpoint.
        log_filename: Log filename written into ``logs_dir``.
        index_filename: Dataset index CSV filename.
        statistics_filename: Dataset statistics JSON filename.
        checkpoint_filename: Resume-checkpoint filename (under ``output_dir``).
    """

    stacks_dir: Path
    masks_dir: Path
    reference_database: Path = Path("metadata/reference_database.json")
    temporal_database: Path = Path("metadata/temporal_database.csv")
    cloud_statistics: Path = Path("metadata/cloud_statistics.csv")
    output_dir: Path = Path("dataset")
    metadata_dir: Path = Path("metadata")
    logs_dir: Path = Path("logs")
    n_references: int = 4
    references: ReferenceParams = field(default_factory=ReferenceParams)
    patch: PatchParams = field(default_factory=PatchParams)
    patch_sizes: list[int] | None = None
    stride: int | None = None
    stride_ratio: float | None = None
    strides: list[int] | None = None
    drop_partial: bool | None = None
    filters: FilterParams = field(default_factory=FilterParams)
    split: SplitParams = field(default_factory=SplitParams)
    write_geotiff: bool = True
    write_npz: bool = True
    compress: str = "DEFLATE"
    num_workers: int = 4
    mask_cloud_value: int = 1
    mask_nodata_value: int = 255
    stack_nodata: int = 0
    tile: str | None = None
    resume: bool = True
    log_filename: str = "patch_generation.log"
    index_filename: str = "dataset_index.csv"
    statistics_filename: str = "dataset_statistics.json"
    checkpoint_filename: str = "_checkpoint.json"

    def __post_init__(self) -> None:
        """Normalise paths, build nested params and validate scalars.

        Raises:
            ValueError: If ``n_references`` < 1 or no output format is enabled.
        """
        self.stacks_dir = Path(self.stacks_dir).expanduser().resolve()
        self.masks_dir = Path(self.masks_dir).expanduser().resolve()
        self.reference_database = Path(self.reference_database).expanduser().resolve()
        self.temporal_database = Path(self.temporal_database).expanduser().resolve()
        self.cloud_statistics = Path(self.cloud_statistics).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.metadata_dir = Path(self.metadata_dir).expanduser().resolve()
        self.logs_dir = Path(self.logs_dir).expanduser().resolve()

        if isinstance(self.patch, dict):
            self.patch = PatchParams(**self.patch)
        if isinstance(self.filters, dict):
            self.filters = FilterParams(**self.filters)
        if isinstance(self.split, dict):
            self.split = SplitParams(**self.split)
        if isinstance(self.references, dict):
            self.references = ReferenceParams(**self.references)

        if self.n_references < 1:
            raise ValueError("n_references must be >= 1")
        if not (self.write_geotiff or self.write_npz):
            raise ValueError("At least one output format must be enabled")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        self._validate_multiscale()

    def _validate_multiscale(self) -> None:
        """Validate the multi-scale patch settings.

        Raises:
            ValueError: If any patch size is non-positive, the explicit
                ``strides`` length does not match ``patch_sizes``, or
                ``stride_ratio`` is non-positive.
        """
        if self.patch_sizes is not None:
            if not self.patch_sizes:
                raise ValueError("patch_sizes must not be empty when provided")
            if any(int(s) <= 0 for s in self.patch_sizes):
                raise ValueError("every patch size must be > 0")
            self.patch_sizes = [int(s) for s in self.patch_sizes]
        if self.strides is not None:
            if self.patch_sizes is None or len(self.strides) != len(self.patch_sizes):
                raise ValueError("strides must align 1:1 with patch_sizes")
            if any(int(s) <= 0 for s in self.strides):
                raise ValueError("every stride must be > 0")
            self.strides = [int(s) for s in self.strides]
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.stride_ratio is not None and self.stride_ratio <= 0:
            raise ValueError("stride_ratio must be > 0")

    def patch_scales(self) -> list["PatchScale"]:
        """Resolve the configured patch sizes into concrete scales.

        Resolution precedence for each size's stride: explicit ``strides`` >
        ``stride_ratio`` > global ``stride`` > non-overlapping default (for
        multi-scale) or ``patch.stride`` (single-scale fallback).

        Returns:
            One :class:`PatchScale` per configured patch size, in order.
        """
        drop = self.drop_partial if self.drop_partial is not None else self.patch.drop_partial
        sizes = self.patch_sizes if self.patch_sizes else [self.patch.size]
        scales: list[PatchScale] = []
        for index, size in enumerate(sizes):
            scales.append(PatchScale(size=size, stride=self._resolve_stride(index, size),
                                     drop_partial=drop))
        return scales

    def _resolve_stride(self, index: int, size: int) -> int:
        """Resolve the stride for one patch size by the documented precedence.

        Args:
            index: Position of the size within ``patch_sizes``.
            size: The patch size in pixels.

        Returns:
            The stride in pixels (always ``>= 1``).
        """
        if self.strides is not None:
            return self.strides[index]
        if self.stride_ratio is not None:
            return max(1, round(size * self.stride_ratio))
        if self.stride is not None:
            return self.stride
        return size if self.patch_sizes else self.patch.stride

    def scale_output_dir(self, size: int) -> Path:
        """Return the output subtree for a given patch size.

        Args:
            size: The patch size in pixels.

        Returns:
            ``<output_dir>/patches_<size>``.
        """
        return self.output_dir / f"patches_{size}"

    @property
    def log_path(self) -> Path:
        """Return the absolute path of the log file."""
        return self.logs_dir / self.log_filename

    @property
    def index_path(self) -> Path:
        """Return the absolute path of the dataset index CSV."""
        return self.metadata_dir / self.index_filename

    @property
    def statistics_path(self) -> Path:
        """Return the absolute path of the dataset statistics JSON."""
        return self.metadata_dir / self.statistics_filename

    @property
    def checkpoint_path(self) -> Path:
        """Return the absolute path of the resume checkpoint."""
        return self.output_dir / self.checkpoint_filename

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        data = asdict(self)
        return _stringify_paths(data)

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> "DatasetConfig":
        """Build a configuration from a YAML file, applying overrides.

        Unknown top-level keys are ignored. Nested ``patch``/``filters``/``split``
        mappings are mapped onto their dataclasses. Keyword ``overrides`` (e.g.
        from the CLI) take precedence over the file.

        Args:
            path: Path to the YAML config file.
            **overrides: Field values overriding the file (``None`` ignored).

        Returns:
            A populated :class:`DatasetConfig`.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the file is not valid YAML or not a mapping.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")

        valid_keys = {f.name for f in fields(cls)}
        merged: dict[str, Any] = {k: v for k, v in raw.items() if k in valid_keys}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**merged)


def _stringify_paths(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert ``Path`` values to strings for JSON serialisation.

    Args:
        data: A dictionary possibly containing ``Path`` values.

    Returns:
        The same structure with paths converted to strings.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = _stringify_paths(value)
        else:
            result[key] = value
    return result
