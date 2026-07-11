"""Orchestrator for the storage-efficient synthetic supervision pipeline.

Stages:
  0. Build the real-mask pool.
  1. Plan clear ground-truth patches (parallel by acquisition); split by date.
  2. Write the ``patch_library`` once for every needed (date, cell) — ground
     truth and references (parallel by date; idempotent).
  3. Assign a real mask to every (patch, variant) under the curriculum + reuse
     budget (deterministic, main process).
  4. Write the ``cloud_tile_library`` once per unique transplanted mask.
  5. Write the sample manifests (tiny JSON), the index, statistics and summary.

Nothing corrupted is materialised — the PyTorch dataset composes the cloudy
input at load time. The build is resumable (planning persisted; writes skip
existing files).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data_loader import DatasetLoader, _index_by_date
from ..logging_setup import configure_logging, get_logger
from ..models import SampleSpec
from . import ids
from .config import SyntheticConfig
from .corruptor import coverage_in_qc_range, difficulty_for_coverage
from .generator import write_cloud_tile_group, write_patch_group
from .mask_pool import MaskSampler, build_mask_pool
from .models import GenOutcome, GroundTruthPatch, MaskEntry, SyntheticManifest

GT_SIDECAR = "_gt_patches.jsonl"
CHECKPOINT = "_synthetic_checkpoint.json"


def _plan_worker(payload: tuple[SampleSpec, SyntheticConfig]) -> list[GroundTruthPatch]:
    """Picklable wrapper running clear-patch planning in a worker."""
    from .planner import plan_clear_patches

    spec, config = payload
    return plan_clear_patches(spec, config)


class SyntheticSupervisionBuilder:
    """Builds the manifest-based synthetic dataset (patch + cloud-tile libraries)."""

    def __init__(self, config: SyntheticConfig) -> None:
        """Initialise the builder.

        Args:
            config: Active synthetic configuration.
        """
        self.config = config
        self._logger = get_logger()
        self._gt_path = config.output_dir / GT_SIDECAR
        self._checkpoint = config.output_dir / CHECKPOINT

    def run(self) -> dict[str, Any]:
        """Run the full synthetic build.

        Returns:
            The statistics dictionary (also written to disk).

        Raises:
            FileNotFoundError: If required inputs are missing.
            ValueError: If no clear ground-truth patches are found.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(self.config.log_path)
        self._logger = get_logger()
        started = datetime.now(timezone.utc)
        self._logger.info("=" * 60)
        self._logger.info("Synthetic supervision (manifest architecture) starting")
        self._logger.info(
            "patch=%d | variants=%d | fill=%s | strategy=%s | workers=%d",
            self.config.patch_size, self.config.variants_per_patch,
            self.config.cloud_fill, self.config.mask_sampling.strategy,
            self.config.num_workers,
        )

        source = self.config.source_dataset_config()
        specs = DatasetLoader(source).build_specs()
        season_by_date = {s.target_date: (s.season or "") for s in specs}
        stack_index = _index_by_date(self.config.stacks_dir, "_stack.tif",
                                     tile=self.config.tile)
        mask_index = _index_by_date(self.config.masks_dir, "_cloudmask.tif")

        pool = build_mask_pool(self.config, season_by_date)
        if not pool:
            raise ValueError("Empty mask pool; cannot generate synthetic clouds")

        gt_patches = self._plan_pass(specs)
        if not gt_patches:
            raise ValueError("No clear ground-truth patches found")
        self._assign_splits(gt_patches)

        self._write_patch_library(gt_patches, stack_index)
        manifests, sampler = self._assign_masks(gt_patches, pool)
        self._write_cloud_tiles(manifests, mask_index, stack_index)
        outcomes = self._write_manifests(manifests)

        statistics = self._compute_statistics(outcomes, gt_patches, sampler)
        summary = self._summary(statistics, started, len(gt_patches), len(manifests))
        self._write_index(outcomes)
        self._write_json(self.config.statistics_path, statistics)
        self._write_json(self.config.summary_path, summary)

        self._logger.info(
            "Done: %d manifest(s) from %d clear patch(es); dataset size %s",
            statistics["written_samples"], len(gt_patches),
            statistics["storage"]["dataset_size_human"],
        )
        return statistics

    # ----- stage 1: plan -------------------------------------------------------

    def _plan_pass(self, specs: list[SampleSpec]) -> list[GroundTruthPatch]:
        """Plan clear ground-truth patches (parallel), persisting for resume.

        Args:
            specs: All acquisition specs.

        Returns:
            The clear ground-truth patches.
        """
        planned = self._load_checkpoint()
        patches = self._load_gt() if self.config.resume else []
        if not self.config.resume:
            self._gt_path.unlink(missing_ok=True)
            self._checkpoint.unlink(missing_ok=True)
            planned, patches = set(), []

        pending = [s for s in specs if s.target_date not in planned]
        self._logger.info("Planning: %d acq, %d done, %d pending",
                          len(specs), len(planned), len(pending))
        if pending:
            payloads = [(s, self.config) for s in pending]
            if self.config.num_workers == 1:
                from .planner import plan_clear_patches
                for spec in tqdm(pending, desc="Plan GT", unit="acq"):
                    patches.extend(self._record_plan(
                        spec.target_date, plan_clear_patches(spec, self.config), planned))
            else:
                with ProcessPoolExecutor(max_workers=self.config.num_workers) as ex:
                    futs = {ex.submit(_plan_worker, p): p[0].target_date for p in payloads}
                    for fut in tqdm(as_completed(futs), total=len(futs),
                                    desc="Plan GT", unit="acq"):
                        patches.extend(self._record_plan(futs[fut], fut.result(), planned))
        self._logger.info("Total clear ground-truth patches: %d", len(patches))
        return patches

    def _record_plan(self, date: str, patches: list[GroundTruthPatch],
                     planned: set[str]) -> list[GroundTruthPatch]:
        """Persist a planned acquisition's patches and mark it done.

        Args:
            date: Acquisition date.
            patches: The clear patches found.
            planned: Mutable set of planned dates.

        Returns:
            The patches (unchanged).
        """
        with self._gt_path.open("a", encoding="utf-8") as handle:
            for p in patches:
                handle.write(json.dumps(p.to_record()) + "\n")
        planned.add(date)
        self._checkpoint.write_text(json.dumps({"planned": sorted(planned)}),
                                    encoding="utf-8")
        return patches

    def _assign_splits(self, patches: list[GroundTruthPatch]) -> None:
        """Assign train/val/test by acquisition date (no temporal leakage).

        Args:
            patches: The clear patches (mutated in place).
        """
        dates = sorted({p.date for p in patches})
        n = len(dates)
        n_train = int(round(n * self.config.split[0]))
        n_val = int(round(n * self.config.split[1]))
        train = set(dates[:n_train])
        val = set(dates[n_train:n_train + n_val])
        for p in patches:
            p.split = "train" if p.date in train else "val" if p.date in val else "test"
        self._logger.info("Temporal split: %d/%d/%d dates (train/val/test)",
                          len(train), len(val), n - len(train) - len(val))

    # ----- stage 2: patch library ----------------------------------------------

    def _write_patch_library(self, patches: list[GroundTruthPatch],
                             stack_index: dict[str, Path]) -> None:
        """Write every needed 13-band patch once (ground truth + references).

        Args:
            patches: The clear ground-truth patches.
            stack_index: Date -> stack path lookup.
        """
        cellpos: dict[int, tuple[int, int]] = {}
        needed: dict[str, set[int]] = defaultdict(set)
        for p in patches:
            cellpos[p.cell_index] = (p.row, p.col)
            needed[p.date].add(p.cell_index)
            for ref_date in p.reference_dates:
                needed[ref_date].add(p.cell_index)

        groups = []
        for date, cells in needed.items():
            stack = stack_index.get(date)
            if stack is None:
                continue
            cell_list = [(c, *cellpos[c]) for c in sorted(cells)]
            groups.append((date, str(stack), cell_list, self.config.patch_size,
                           str(self.config.output_dir)))
        self._run_groups(groups, write_patch_group, "Patch library")

    # ----- stage 3: assign masks -----------------------------------------------

    def _assign_masks(self, patches: list[GroundTruthPatch], pool: list[MaskEntry]):
        """Assign a real mask to each (patch, variant) deterministically.

        Args:
            patches: The clear ground-truth patches.
            pool: The real-mask pool.

        Returns:
            ``(manifests, sampler)``.
        """
        sampler = MaskSampler(pool, self.config)
        bins = self.config.curriculum.bins
        rng = random.Random(self.config.seed)
        patches.sort(key=lambda p: (p.split, p.date, p.cell_index))
        manifests: list[SyntheticManifest] = []
        counter = 0
        for patch in patches:
            for variant in range(self.config.variants_per_patch):
                band = self._pick_band(variant, bins, rng)
                mask = sampler.sample(patch, band)
                if mask is None:
                    continue
                ok, _ = coverage_in_qc_range(mask.cloud_fraction, self.config)
                if not ok:
                    continue
                counter += 1
                difficulty = difficulty_for_coverage(mask.cloud_fraction, self.config)
                manifests.append(SyntheticManifest(
                    sample_id=f"sample_{counter:06d}", split=patch.split, gt=patch,
                    mask=mask, difficulty=difficulty, augmentation_index=variant,
                    seed=self.config.seed + counter,
                    applied_cloud_coverage=mask.cloud_fraction,
                ))
        self._logger.info("Assigned %d synthetic manifest(s)", len(manifests))
        return manifests, sampler

    def _pick_band(self, variant: int, bins, rng: random.Random):
        """Choose a curriculum band for a variant index.

        Args:
            variant: Variant index.
            bins: Curriculum bands.
            rng: Seeded RNG.

        Returns:
            The chosen band.
        """
        cur = self.config.curriculum
        if not cur.enabled:
            return bins[0] if len(bins) == 1 else rng.choice(bins)
        if cur.assignment == "cycle":
            return bins[variant % len(bins)]
        if cur.assignment == "weighted":
            return rng.choices(bins, weights=[b.weight for b in bins], k=1)[0]
        return rng.choice(bins)

    # ----- stage 4: cloud tiles -------------------------------------------------

    def _write_cloud_tiles(self, manifests: list[SyntheticManifest],
                           mask_index: dict[str, Path],
                           stack_index: dict[str, Path]) -> None:
        """Write each unique transplanted cloud tile once.

        Args:
            manifests: All sample manifests.
            mask_index: Date -> mask path lookup.
            stack_index: Date -> stack path lookup (for overlay reflectance).
        """
        by_date: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
        for m in manifests:
            by_date[m.mask.date][m.mask.cell_index] = (m.mask.source_row, m.mask.source_col)

        groups = []
        for date, cells in by_date.items():
            mask_path = mask_index.get(date)
            if mask_path is None:
                self._logger.warning("Mask date %s has no cloud mask file; skipping", date)
                continue
            stack = stack_index.get(date)
            cell_list = [(c, r, col) for c, (r, col) in sorted(cells.items())]
            groups.append((date, str(mask_path), str(stack) if stack else "", cell_list,
                           self.config.patch_size, str(self.config.output_dir),
                           self.config.cloud_fill, self.config.mask_cloud_value))
        self._run_groups(groups, write_cloud_tile_group, "Cloud tiles")

    # ----- stage 5: manifests ---------------------------------------------------

    def _write_manifests(self, manifests: list[SyntheticManifest]) -> list[GenOutcome]:
        """Write the tiny per-sample JSON manifests.

        Args:
            manifests: All sample manifests.

        Returns:
            Per-sample reporting outcomes.
        """
        outcomes: list[GenOutcome] = []
        for m in tqdm(manifests, desc="Manifests", unit="smp"):
            rel = ids.sample_relpath(m.split, m.sample_id)
            path = self.config.output_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(m.to_manifest_json(self.config.cloud_fill),
                                       indent=2), encoding="utf-8")
            outcomes.append(GenOutcome(
                status="written", split=m.split, gt_date=m.gt.date, mask_date=m.mask.date,
                cell_index=m.gt.cell_index, augmentation_index=m.augmentation_index,
                difficulty=m.difficulty, applied_cloud_coverage=m.applied_cloud_coverage,
                n_references=len(m.gt.reference_dates), season=m.gt.season, month=m.gt.month,
                message="OK"))
        self._logger.info("Wrote %d manifest(s)", len(outcomes))
        return outcomes

    # ----- helpers -------------------------------------------------------------

    def _run_groups(self, groups: list, worker, desc: str) -> None:
        """Run a group-writing worker over groups (parallel), logging errors.

        Args:
            groups: The worker payloads.
            worker: The module-level worker function.
            desc: Progress-bar label.
        """
        if not groups:
            return
        written = 0
        if self.config.num_workers == 1:
            for g in tqdm(groups, desc=desc, unit="grp"):
                w, err = worker(g)
                written += w
                if err:
                    self._logger.error("%s error: %s", desc, err)
        else:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as ex:
                futs = [ex.submit(worker, g) for g in groups]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="grp"):
                    w, err = fut.result()
                    written += w
                    if err:
                        self._logger.error("%s error: %s", desc, err)
        self._logger.info("%s: wrote %d new file(s)", desc, written)

    def _load_checkpoint(self) -> set[str]:
        """Load the set of planned acquisition dates from the checkpoint."""
        if not self.config.resume or not self._checkpoint.exists():
            return set()
        try:
            return set(json.loads(self._checkpoint.read_text(encoding="utf-8"))
                       .get("planned", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _load_gt(self) -> list[GroundTruthPatch]:
        """Load planned ground-truth patches from the sidecar."""
        if not self._gt_path.exists():
            return []
        out = []
        with self._gt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(GroundTruthPatch(**json.loads(line)))
        return out

    # ----- reporting -----------------------------------------------------------

    def _write_index(self, outcomes: list[GenOutcome]) -> None:
        """Write the per-sample index CSV.

        Args:
            outcomes: All reporting outcomes.
        """
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([o.to_row() for o in outcomes]).to_csv(
            self.config.index_path, index=False, encoding="utf-8")
        self._logger.info("Wrote index: %s (%d rows)", self.config.index_path, len(outcomes))

    def _compute_statistics(self, outcomes: list[GenOutcome],
                            gt_patches: list[GroundTruthPatch],
                            sampler: MaskSampler) -> dict[str, Any]:
        """Compute dataset + storage statistics with an old-vs-new comparison.

        Args:
            outcomes: All reporting outcomes.
            gt_patches: The clear ground-truth patches.
            sampler: The mask sampler (for reuse statistics).

        Returns:
            A JSON-serialisable statistics dictionary.
        """
        written = [o for o in outcomes if o.status == "written"]
        cov = [o.applied_cloud_coverage for o in written if o.applied_cloud_coverage is not None]
        edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist = np.histogram(cov, bins=edges)[0].tolist() if cov else []
        ref_hist = Counter(o.n_references for o in written if o.n_references)

        root = self.config.output_dir
        patch_b, patch_n = _dir_size(root / ids.PATCH_LIBRARY_DIR)
        tile_b, tile_n = _dir_size(root / ids.CLOUD_TILE_LIBRARY_DIR)
        sample_b, sample_n = _dir_size(root / ids.SAMPLES_DIR)
        total = patch_b + tile_b + sample_b
        # Old self-contained design stored, per sample, cloudy + clear + all
        # references (~ measured at ~5.5 MB/sample compressed).
        old_estimate = int(len(written) * 5.5 * 1024 * 1024)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "architecture": "manifest_shared_library",
            "clear_ground_truth_patches": len(gt_patches),
            "written_samples": len(written),
            "split_distribution": dict(Counter(o.split for o in written)),
            "curriculum_distribution": dict(Counter(o.difficulty for o in written)),
            "average_cloud_coverage": round(float(np.mean(cov)), 4) if cov else None,
            "cloud_coverage_histogram": {"edges": edges, "counts": hist},
            "reference_count_distribution": {str(k): ref_hist[k] for k in sorted(ref_hist)},
            "season_distribution": dict(Counter(o.season for o in written if o.season)),
            "month_distribution": {str(k): v for k, v in sorted(
                Counter(o.month for o in written if o.month).items())},
            "mask_reuse": sampler.reuse_statistics(),
            "reference_availability": {
                "patches_with_min_refs": sum(
                    1 for p in gt_patches
                    if len(p.reference_dates) >= self.config.min_references),
                "total_patches": len(gt_patches),
            },
            "storage": {
                "unique_patches": patch_n, "cloud_tiles": tile_n, "manifests": sample_n,
                "patch_library_bytes": patch_b, "cloud_tile_library_bytes": tile_b,
                "dataset_size_bytes": total, "dataset_size_human": _human(total),
                "old_self_contained_estimate_human": _human(old_estimate),
                "savings_human": _human(max(0, old_estimate - total)),
            },
            "config": self.config.to_dict(),
        }

    def _summary(self, statistics: dict[str, Any], started: datetime,
                 n_patches: int, n_manifests: int) -> dict[str, Any]:
        """Build the high-level generation summary.

        Args:
            statistics: The computed statistics.
            started: Build start time.
            n_patches: Number of clear ground-truth patches.
            n_manifests: Number of manifests.

        Returns:
            A JSON-serialisable summary dictionary.
        """
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "generated_at": statistics["generated_at"],
            "elapsed_seconds": round(elapsed, 1),
            "clear_ground_truth_patches": n_patches,
            "variants_per_patch": self.config.variants_per_patch,
            "written_samples": statistics["written_samples"],
            "effective_augmentation_factor": round(
                statistics["written_samples"] / n_patches, 3) if n_patches else 0,
            "curriculum_distribution": statistics["curriculum_distribution"],
            "dataset_size_human": statistics["storage"]["dataset_size_human"],
            "savings_vs_self_contained": statistics["storage"]["savings_human"],
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write a dict as pretty JSON.

        Args:
            path: Destination path.
            payload: The dict to serialise.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Wrote %s", path)


def _dir_size(path: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for a directory tree.

    Args:
        path: Directory to measure.

    Returns:
        Total size in bytes and file count (zeros if absent).
    """
    if not path.exists():
        return 0, 0
    total = count = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
            count += 1
    return total, count


def _human(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        A string such as ``"12.3 GB"``.
    """
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"
