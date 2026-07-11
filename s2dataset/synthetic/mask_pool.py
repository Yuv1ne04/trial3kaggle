"""Real cloud-mask pool construction and configurable sampling strategies."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..logging_setup import get_logger
from .config import CurriculumBin, SyntheticConfig
from .models import GroundTruthPatch, MaskEntry

logger = get_logger()


def _month_of(date: str) -> int | None:
    """Return the month (1-12) parsed from a ``YYYYMMDD`` date, or ``None``."""
    try:
        return int(date[4:6])
    except (ValueError, IndexError):
        return None


def build_mask_pool(
    config: SyntheticConfig,
    season_by_date: dict[str, str],
    *,
    max_pool: int = 40000,
) -> list[MaskEntry]:
    """Build (and cache) the pool of real cloud-mask patches.

    The pool is drawn from the pre-extracted ``mask_library`` of the shared
    dataset (real Mauritius cloud masks). Each entry records its cloud fraction,
    season and month so downstream sampling can match coverage/season/month.

    Args:
        config: Active synthetic configuration.
        season_by_date: Mapping of ``YYYYMMDD`` -> season label.
        max_pool: Cap on pool size (randomly subsampled if exceeded).

    Returns:
        A list of :class:`MaskEntry`.

    Raises:
        FileNotFoundError: If no mask library is available to sample from.
    """
    if config.mask_pool_path.exists():
        cached = json.loads(config.mask_pool_path.read_text(encoding="utf-8"))
        pool = [MaskEntry(**e) for e in cached]
        logger.info("Loaded cached mask pool: %d entr(ies)", len(pool))
        return pool

    lib = config.mask_library_dir
    if lib is None or not lib.exists():
        raise FileNotFoundError(
            "No mask_library_dir available to build the real-mask pool; "
            "point mask_library_dir at the shared dataset's mask_library."
        )

    files = sorted((lib / str(config.patch_size)).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No mask patches under {lib}/{config.patch_size}")
    if len(files) > max_pool:
        rng = random.Random(config.seed)
        files = rng.sample(files, max_pool)

    pool: list[MaskEntry] = []
    for path in tqdm(files, desc="Mask pool", unit="mask"):
        entry = _mask_entry_from_file(path, config, season_by_date)
        if entry is not None:
            pool.append(entry)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.mask_pool_path.write_text(
        json.dumps([e.to_record() for e in pool]), encoding="utf-8")
    logger.info("Built mask pool: %d entr(ies) (cached)", len(pool))
    return pool


def _mask_entry_from_file(
    path: Path, config: SyntheticConfig, season_by_date: dict[str, str]
) -> MaskEntry | None:
    """Read one mask patch and build a :class:`MaskEntry`.

    Args:
        path: Path to the mask patch npz (``mask_library/<size>/<date>/patch.npz``).
        config: Active synthetic configuration.
        season_by_date: Season lookup.

    Returns:
        A :class:`MaskEntry`, or ``None`` if the file is unreadable/empty.
    """
    try:
        with np.load(path) as data:
            mask = data["mask"]
            row = col = 0
            if "metadata" in data:
                meta = json.loads(str(data["metadata"]))
                coords = meta.get("patch_coordinates", {})
                row, col = int(coords.get("row", 0)), int(coords.get("col", 0))
    except Exception:  # noqa: BLE001 - skip unreadable pool entries
        return None
    valid = mask != config.mask_nodata_value
    valid_count = int(valid.sum())
    if valid_count == 0:
        return None
    cloud = float((mask == config.mask_cloud_value).sum()) / valid_count
    date = path.parent.name
    cell = _cell_from_name(path.name)
    return MaskEntry(
        path=str(path), date=date, cell_index=cell, cloud_fraction=cloud,
        season=season_by_date.get(date), month=_month_of(date),
        source_row=row, source_col=col,
    )


def _cell_from_name(name: str) -> int:
    """Extract the trailing cell index from a ``patch_000254.npz`` filename."""
    stem = name.split(".")[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


class MaskSampler:
    """Samples real cloud masks under a strategy, curriculum and reuse budget.

    A single sampler instance lives in the orchestrator process so its reuse
    counter is global across the whole dataset.
    """

    def __init__(self, pool: list[MaskEntry], config: SyntheticConfig) -> None:
        """Initialise the sampler.

        Args:
            pool: The real-mask pool.
            config: Active synthetic configuration.
        """
        self.pool = pool
        self.config = config
        self.params = config.mask_sampling
        self.rng = random.Random(config.seed)
        self.reuse: Counter[str] = Counter()

    def _eligible(self, gt: GroundTruthPatch, low: float, high: float) -> list[MaskEntry]:
        """Return masks matching coverage/date/reuse constraints.

        Args:
            gt: The ground-truth patch (for the different-date rule).
            low: Inclusive lower coverage bound.
            high: Inclusive upper coverage bound.

        Returns:
            The eligible mask entries.
        """
        cap = self.params.max_reuse
        out = []
        for m in self.pool:
            if not (low <= m.cloud_fraction <= high):
                continue
            if self.params.different_date and m.date == gt.date:
                continue
            if cap and self.reuse[m.path] >= cap:
                continue
            out.append(m)
        return out

    def sample(self, gt: GroundTruthPatch, band: CurriculumBin) -> MaskEntry | None:
        """Sample one mask for a ground-truth patch and difficulty band.

        Eligibility is progressively relaxed (band -> widened coverage -> ignore
        reuse cap) so generation degrades gracefully when a band is sparse.

        Args:
            gt: The ground-truth patch.
            band: The target curriculum band.

        Returns:
            A chosen :class:`MaskEntry`, or ``None`` if none is available.
        """
        candidates = self._eligible(gt, band.min_coverage, band.max_coverage)
        if not candidates:
            candidates = self._eligible(gt, max(0.02, band.min_coverage - 0.1),
                                        min(0.95, band.max_coverage + 0.1))
        if not candidates:
            saved = self.params.max_reuse
            self.params.max_reuse = 0
            candidates = self._eligible(gt, band.min_coverage, band.max_coverage)
            self.params.max_reuse = saved
        if not candidates:
            return None

        chosen = self._choose(gt, band, candidates)
        self.reuse[chosen.path] += 1
        return chosen

    def _choose(self, gt: GroundTruthPatch, band: CurriculumBin,
                candidates: list[MaskEntry]) -> MaskEntry:
        """Select one candidate according to the configured strategy.

        Args:
            gt: The ground-truth patch.
            band: The target curriculum band.
            candidates: Eligible mask entries.

        Returns:
            The chosen mask entry.
        """
        strategy = self.params.strategy
        if strategy == "random":
            return self.rng.choice(candidates)
        if strategy == "similar_season":
            same = [m for m in candidates if m.season and m.season == gt.season]
            return self.rng.choice(same or candidates)
        if strategy == "similar_month":
            same = [m for m in candidates if m.month and m.month == gt.month]
            return self.rng.choice(same or candidates)

        center = 0.5 * (band.min_coverage + band.max_coverage)
        weights = [self._weight(m, gt, center, strategy) for m in candidates]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(candidates)
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _weight(self, m: MaskEntry, gt: GroundTruthPatch, center: float,
                strategy: str) -> float:
        """Compute a sampling weight for a candidate (coverage/weighted modes).

        Args:
            m: Candidate mask entry.
            gt: Ground-truth patch.
            center: Target coverage centre for the band.
            strategy: Active strategy.

        Returns:
            A non-negative weight.
        """
        coverage_match = 1.0 / (1.0 + abs(m.cloud_fraction - center) /
                                max(self.params.coverage_tolerance, 1e-3))
        if strategy == "similar_coverage":
            return coverage_match
        # weighted: favour coverage match, rarely-used masks and season match.
        recency = 1.0 / (1.0 + self.reuse[m.path])
        season_bonus = 1.5 if (m.season and m.season == gt.season) else 1.0
        return (self.params.weight_coverage_match * coverage_match
                + self.params.weight_recency * recency) * season_bonus

    def reuse_statistics(self) -> dict[str, Any]:
        """Return summary statistics of mask reuse across the run.

        Returns:
            A dict with total/unique masks used and reuse distribution.
        """
        used = self.reuse
        counts = list(used.values())
        return {
            "unique_masks_used": len(used),
            "pool_size": len(self.pool),
            "total_applications": int(sum(counts)),
            "max_reuse_observed": int(max(counts)) if counts else 0,
            "mean_reuse": round(float(np.mean(counts)), 3) if counts else 0.0,
        }
