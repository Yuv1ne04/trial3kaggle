"""Configuration for the synthetic supervision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..config import DatasetConfig


@dataclass(slots=True)
class CurriculumBin:
    """One curriculum difficulty band defined by applied cloud coverage.

    Attributes:
        name: Band name (e.g. ``"easy"``).
        min_coverage: Inclusive lower bound of cloud coverage.
        max_coverage: Inclusive upper bound of cloud coverage.
        weight: Relative sampling weight when assignment is ``"weighted"``.
    """

    name: str
    min_coverage: float
    max_coverage: float
    weight: float = 1.0


@dataclass(slots=True)
class CurriculumParams:
    """Curriculum-learning configuration.

    Attributes:
        enabled: When ``True`` each variant targets a curriculum band; otherwise
            the global ``[min,max]`` coverage range is used for all variants.
        assignment: How variants are assigned to bands: ``"cycle"`` (variant i ->
            band i mod K), ``"random"`` (uniform), or ``"weighted"`` (by weight).
        bins: The difficulty bands (defaults: easy 10-20%, medium 20-40%,
            hard 40-70%).
    """

    enabled: bool = True
    assignment: str = "cycle"
    bins: list[CurriculumBin] = field(default_factory=lambda: [
        CurriculumBin("easy", 0.10, 0.20),
        CurriculumBin("medium", 0.20, 0.40),
        CurriculumBin("hard", 0.40, 0.70),
    ])

    def __post_init__(self) -> None:
        """Coerce dict bins and validate the assignment mode.

        Raises:
            ValueError: If ``assignment`` is unknown or bins are empty.
        """
        self.bins = [CurriculumBin(**b) if isinstance(b, dict) else b for b in self.bins]
        if not self.bins:
            raise ValueError("curriculum requires at least one bin")
        if self.assignment not in {"cycle", "random", "weighted"}:
            raise ValueError("assignment must be cycle/random/weighted")


@dataclass(slots=True)
class ClearFilterParams:
    """Thresholds selecting *clear* patches usable as ground truth.

    Attributes:
        max_cloud_fraction: Maximum native cloud fraction to accept as clear GT.
        max_nodata_fraction: Maximum NoData fraction.
        max_background_fraction: Maximum ocean/background fraction.
        min_valid_fraction: Minimum valid-pixel fraction.
    """

    max_cloud_fraction: float = 0.05
    max_nodata_fraction: float = 0.30
    max_background_fraction: float = 0.50
    min_valid_fraction: float = 0.50


@dataclass(slots=True)
class SyntheticQCParams:
    """Post-corruption quality-control thresholds.

    Attributes:
        min_cloud_coverage: Reject if applied coverage below this.
        max_cloud_coverage: Reject if applied coverage above this.
        max_nodata_fraction: Reject if GT NoData exceeds this.
        max_background_fraction: Reject if GT background/ocean exceeds this.
    """

    min_cloud_coverage: float = 0.05
    max_cloud_coverage: float = 0.90
    max_nodata_fraction: float = 0.30
    max_background_fraction: float = 0.50


@dataclass(slots=True)
class MaskSamplingParams:
    """Real-cloud-mask sampling policy.

    Attributes:
        strategy: ``"random"``, ``"similar_coverage"``, ``"similar_season"``,
            ``"similar_month"`` or ``"weighted"``.
        different_date: Require the sampled mask to come from a different date
            than the ground-truth patch.
        max_reuse: Maximum times a single mask patch may be reused across the
            whole dataset (0 = unlimited).
        coverage_tolerance: Half-width of the coverage window for
            ``"similar_coverage"``.
        weight_recency: Weight term favouring rarely-used masks in ``"weighted"``.
        weight_coverage_match: Weight term favouring coverage-matched masks.
    """

    strategy: str = "similar_coverage"
    different_date: bool = True
    max_reuse: int = 50
    coverage_tolerance: float = 0.07
    weight_recency: float = 1.0
    weight_coverage_match: float = 1.0

    def __post_init__(self) -> None:
        """Validate the sampling strategy.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        allowed = {"random", "similar_coverage", "similar_season",
                   "similar_month", "weighted"}
        if self.strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}")


@dataclass(slots=True)
class SyntheticConfig:
    """Full configuration for a synthetic-supervision run.

    Attributes:
        stacks_dir: Directory of 13-band stacks (source of clear GT + references).
        masks_dir: Directory of cloud masks (per acquisition).
        mask_library_dir: Directory of pre-extracted mask patches (the real-mask
            pool); when absent the pool is built from ``masks_dir`` windows.
        reference_database: Path to ``reference_database.json``.
        temporal_database: Path to ``temporal_database.csv``.
        cloud_statistics: Path to ``cloud_statistics.csv``.
        output_dir: Root output directory (``synthetic_dataset``).
        metadata_dir: Directory for statistics/summary/index outputs.
        logs_dir: Directory for the log file.
        tile: Optional MGRS tile filter (e.g. ``"T40KEC"``).
        patch_size: Patch side length (single scale for supervision).
        variants_per_patch: Synthetic variants generated per clear patch.
        min_references: Minimum references required to emit a sample.
        max_references: Maximum references stored (padded in the input npz).
        cloud_fill: ``"overlay"`` (paste real cloud reflectance), ``"constant"``
            or ``"zero"`` for cloudy pixels of the synthetic input.
        constant_fill_value: Reflectance value used when ``cloud_fill`` is
            ``"constant"`` (DN units, pre-scaling).
        curriculum: Curriculum-learning configuration.
        clear_filter: Clear-GT selection thresholds.
        qc: Post-corruption QC thresholds.
        mask_sampling: Real-cloud-mask sampling policy.
        split: Train/val/test fractions (by date).
        seed: Master random seed.
        num_workers: Parallel worker processes.
        resume: Skip work already recorded in the checkpoint.
        stack_nodata: Stack NoData value.
        mask_cloud_value: Mask value denoting cloud.
        mask_nodata_value: Mask value denoting NoData.
        background_reflectance: All-band reflectance <= this counts as background.
        log_filename / statistics_filename / summary_filename / index_filename /
        checkpoint_filename / mask_pool_filename: Output artefact names.
    """

    stacks_dir: Path
    masks_dir: Path
    mask_library_dir: Path | None = None
    reference_database: Path = Path("metadata/reference_database.json")
    temporal_database: Path = Path("metadata/temporal_database.csv")
    cloud_statistics: Path = Path("metadata/cloud_statistics.csv")
    output_dir: Path = Path("synthetic_dataset")
    metadata_dir: Path = Path("metadata")
    logs_dir: Path = Path("logs")
    tile: str | None = "T40KEC"
    patch_size: int = 256
    variants_per_patch: int = 3
    min_references: int = 2
    max_references: int = 4
    cloud_fill: str = "overlay"
    constant_fill_value: int = 8000
    curriculum: CurriculumParams = field(default_factory=CurriculumParams)
    clear_filter: ClearFilterParams = field(default_factory=ClearFilterParams)
    qc: SyntheticQCParams = field(default_factory=SyntheticQCParams)
    mask_sampling: MaskSamplingParams = field(default_factory=MaskSamplingParams)
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 1234
    num_workers: int = 4
    resume: bool = True
    stack_nodata: int = 0
    mask_cloud_value: int = 1
    mask_nodata_value: int = 255
    background_reflectance: int = 1
    log_filename: str = "synthetic_generation.log"
    statistics_filename: str = "synthetic_dataset_statistics.json"
    summary_filename: str = "synthetic_generation_summary.json"
    index_filename: str = "synthetic_dataset_index.csv"
    checkpoint_filename: str = "_synthetic_checkpoint.json"
    mask_pool_filename: str = "_mask_pool.json"

    def __post_init__(self) -> None:
        """Normalise paths, coerce nested params and validate scalars.

        Raises:
            ValueError: If counts/fill/splits are invalid.
        """
        for name in ("stacks_dir", "masks_dir", "reference_database",
                     "temporal_database", "cloud_statistics", "output_dir",
                     "metadata_dir", "logs_dir"):
            setattr(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.mask_library_dir is not None:
            self.mask_library_dir = Path(self.mask_library_dir).expanduser().resolve()

        if isinstance(self.curriculum, dict):
            self.curriculum = CurriculumParams(**self.curriculum)
        if isinstance(self.clear_filter, dict):
            self.clear_filter = ClearFilterParams(**self.clear_filter)
        if isinstance(self.qc, dict):
            self.qc = SyntheticQCParams(**self.qc)
        if isinstance(self.mask_sampling, dict):
            self.mask_sampling = MaskSamplingParams(**self.mask_sampling)
        self.split = tuple(float(x) for x in self.split)

        if self.variants_per_patch < 1:
            raise ValueError("variants_per_patch must be >= 1")
        if self.min_references < 1 or self.max_references < self.min_references:
            raise ValueError("require 1 <= min_references <= max_references")
        if self.cloud_fill not in {"overlay", "constant", "zero"}:
            raise ValueError("cloud_fill must be overlay/constant/zero")
        if abs(sum(self.split) - 1.0) > 1e-6 and sum(self.split) > 0:
            total = sum(self.split)
            self.split = tuple(x / total for x in self.split)

    def source_dataset_config(self) -> DatasetConfig:
        """Build a :class:`DatasetConfig` for reusing the reference loader.

        Returns:
            A :class:`DatasetConfig` pointing at the same source inputs, with the
            reference maximum set to this run's ``max_references`` and the tile
            filter applied.
        """
        return DatasetConfig(
            stacks_dir=self.stacks_dir,
            masks_dir=self.masks_dir,
            reference_database=self.reference_database,
            temporal_database=self.temporal_database,
            cloud_statistics=self.cloud_statistics,
            output_dir=self.output_dir,
            metadata_dir=self.metadata_dir,
            logs_dir=self.logs_dir,
            references={"maximum": self.max_references, "minimum": self.min_references},
            patch_sizes=[self.patch_size],
            tile=self.tile,
            stack_nodata=self.stack_nodata,
            mask_cloud_value=self.mask_cloud_value,
            mask_nodata_value=self.mask_nodata_value,
        )

    @property
    def log_path(self) -> Path:
        """Return the absolute path of the log file."""
        return self.logs_dir / self.log_filename

    @property
    def statistics_path(self) -> Path:
        """Return the absolute path of the statistics JSON."""
        return self.metadata_dir / self.statistics_filename

    @property
    def summary_path(self) -> Path:
        """Return the absolute path of the generation-summary JSON."""
        return self.metadata_dir / self.summary_filename

    @property
    def index_path(self) -> Path:
        """Return the absolute path of the index CSV."""
        return self.metadata_dir / self.index_filename

    @property
    def checkpoint_path(self) -> Path:
        """Return the absolute path of the resume checkpoint."""
        return self.output_dir / self.checkpoint_filename

    @property
    def mask_pool_path(self) -> Path:
        """Return the absolute path of the cached mask pool."""
        return self.output_dir / self.mask_pool_filename

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        def _clean(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_clean(v) for v in value]
            return value
        return {k: _clean(v) for k, v in asdict(self).items()}

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: Any) -> "SyntheticConfig":
        """Build a configuration from a YAML file, applying overrides.

        Args:
            path: Path to the YAML config file.
            **overrides: Field values overriding the file (``None`` ignored).

        Returns:
            A populated :class:`SyntheticConfig`.

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
        valid = {f.name for f in fields(cls)}
        merged = {k: v for k, v in raw.items() if k in valid}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**merged)
