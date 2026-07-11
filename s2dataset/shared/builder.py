"""Three-pass orchestrator for the shared-reference dataset.

Pass 1 (parallel by target): plan each target — filter cells, write the
target image (target library) and mask (mask library), emit per-cell plans with
ranked *candidate* references.
Pass 2 (parallel by reference date): evaluate each needed reference patch's
quality and, if valid, write it once into the reference library; returns a
target-independent validity map.
Pass 3: finalise samples — keep each plan's valid references (2..maximum),
drop plans with fewer than the minimum, write sample JSONs, the index, the
statistics and the generation summary.

All passes are idempotent (skip existing files) and checkpoint-backed, so the
build is fully resumable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..config import DatasetConfig
from ..data_loader import DatasetLoader, _index_by_date
from ..logging_setup import configure_logging, get_logger
from ..models import SampleSpec
from ..splitter import assign_splits
from . import ids
from .models import PlanResult, SampleManifest, TargetPlan
from .refeval import RefEvalParams, evaluate_reference_group

PLAN_SIDECAR = "_plans.jsonl"
CHECKPOINT_FILE = "_shared_checkpoint.json"
SUMMARY_FILE = "generation_summary.json"


def _plan_worker(payload: tuple[SampleSpec, DatasetConfig]) -> PlanResult:
    """Picklable wrapper running :func:`plan_target` in a worker process."""
    from .planner import plan_target

    spec, config = payload
    return plan_target(spec, config)


class SharedDatasetBuilder:
    """Builds the shared-reference dataset (3 libraries + JSON samples)."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise the builder.

        Args:
            config: Active dataset configuration.
        """
        self.config = config
        self._logger = get_logger()
        self._plan_path = config.output_dir / PLAN_SIDECAR
        self._checkpoint_path = config.output_dir / CHECKPOINT_FILE
        self._summary_path = config.metadata_dir / SUMMARY_FILE

    def run(self) -> dict[str, Any]:
        """Run the full shared-reference build.

        Returns:
            The dataset statistics dictionary (also written to disk).

        Raises:
            FileNotFoundError: If required inputs are missing.
            ValueError: If no samples are produced.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(self.config.log_path)
        self._logger = get_logger()
        started = datetime.now(timezone.utc)
        self._logger.info("=" * 60)
        self._logger.info("Shared-reference dataset build starting")
        scales = self.config.patch_scales()
        refs = self.config.references
        self._logger.info(
            "scales=%s | references=%d..%d | workers=%d",
            ", ".join(f"{s.size}px/str{s.stride}" for s in scales),
            refs.minimum, refs.maximum, self.config.num_workers,
        )

        specs = DatasetLoader(self.config).build_specs()
        assign_splits(specs, self.config)
        stack_index = _index_by_date(self.config.stacks_dir, "_stack.tif")
        mask_index = _index_by_date(self.config.masks_dir, "_cloudmask.tif")

        plans = self._plan_pass(specs)
        if not plans:
            raise ValueError("No target patches produced (check filters/inputs)")

        estimate = self._estimate_storage(plans)
        self._logger.info("Pre-generation storage estimate: %s", estimate["estimated_human"])

        validity = self._reference_pass(plans, stack_index, mask_index)
        manifests = self._finalise(plans, validity)
        if not manifests:
            raise ValueError("No samples met the minimum-reference requirement")

        self._write_samples(manifests)
        statistics = self._compute_statistics(manifests, estimate)
        self._write_index(manifests)
        self._write_statistics(statistics)
        self._write_summary(statistics, estimate, started, len(plans), len(manifests))

        self._logger.info(
            "Build complete: %d samples, %s on disk (saved ~%s vs duplicated)",
            statistics["total_samples"],
            statistics["storage"]["new_actual_human"],
            statistics["storage"]["savings_human"],
        )
        return statistics

    # ----- pass 1: planning ----------------------------------------------------

    def _plan_pass(self, specs: list[SampleSpec]) -> list[TargetPlan]:
        """Plan all targets (parallel), persisting plans for resume.

        Args:
            specs: All target sample specs.

        Returns:
            All target plans (prior + newly planned).
        """
        completed = self._load_checkpoint()
        plans = self._load_plans() if self.config.resume else []
        if not self.config.resume:
            self._reset_state()
            completed, plans = set(), []

        pending = [s for s in specs if s.target_date not in completed]
        self._logger.info("Planning: %d target(s), %d done, %d to plan",
                          len(specs), len(completed), len(pending))
        if not pending:
            return plans

        if self.config.num_workers == 1:
            from .planner import plan_target
            for spec in tqdm(pending, desc="Plan", unit="tgt"):
                plans.extend(self._handle_plan(plan_target(spec, self.config), completed))
        else:
            payloads = [(s, self.config) for s in pending]
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as ex:
                futures = {ex.submit(_plan_worker, p): p[0] for p in payloads}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Plan", unit="tgt"):
                    spec = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = PlanResult(spec.target_date, "failed",
                                            message=f"Worker crashed: {exc}")
                    plans.extend(self._handle_plan(result, completed))
        return plans

    def _handle_plan(self, result: PlanResult, completed: set[str]) -> list[TargetPlan]:
        """Persist a plan result and return its plans.

        Args:
            result: The plan result for one target.
            completed: Mutable set of completed target dates.

        Returns:
            The result's plans (empty for aborted/failed targets).
        """
        if result.status == "planned":
            self._append_plans(result.plans)
            completed.add(result.target_date)
            self._persist_checkpoint(completed)
            self._logger.info("%s: %d cell(s), +%d img/+%d mask (%.1fs)",
                              result.target_date, len(result.plans),
                              result.image_patches_written, result.mask_patches_written,
                              result.duration_sec or 0.0)
            return result.plans
        self._logger.error("%s: %s (%s)", result.target_date, result.message, result.status)
        return []

    # ----- pass 2: reference evaluation ---------------------------------------

    def _reference_pass(
        self, plans: list[TargetPlan], stack_index: dict[str, Path],
        mask_index: dict[str, Path],
    ) -> dict[tuple[int, str, int], bool]:
        """Evaluate + write valid reference patches; return a validity map.

        Args:
            plans: All target plans.
            stack_index: Date -> stack path lookup.
            mask_index: Date -> mask path lookup.

        Returns:
            Mapping of ``(size, date, cell_index)`` -> validity.
        """
        cell_pos: dict[tuple[int, int], tuple[int, int]] = {}
        needed: dict[tuple[int, str], set[int]] = defaultdict(set)
        for plan in plans:
            cell_pos[(plan.size, plan.cell_index)] = (plan.row, plan.col)
            for date in plan.candidate_dates:
                needed[(plan.size, date)].add(plan.cell_index)

        params = RefEvalParams(
            max_cloud_fraction=self.config.references.max_cloud_fraction,
            max_nodata_fraction=self.config.references.max_nodata_fraction,
            stack_nodata=self.config.stack_nodata,
            mask_cloud_value=self.config.mask_cloud_value,
        )
        groups = []
        for (size, date), cells in needed.items():
            stack = stack_index.get(date)
            if stack is None:
                continue
            mask = mask_index.get(date)
            cell_list = [(c, *cell_pos[(size, c)]) for c in sorted(cells)]
            groups.append((size, date, str(stack), str(mask) if mask else "",
                           cell_list, str(self.config.output_dir), params))

        self._logger.info("Reference pass: %d (size,date) group(s) to evaluate", len(groups))
        validity: dict[tuple[int, str, int], bool] = {}
        written = 0
        if not groups:
            return validity

        if self.config.num_workers == 1:
            for group in tqdm(groups, desc="References", unit="grp"):
                v, w, err = evaluate_reference_group(group)
                validity.update(v)
                written += w
                if err:
                    self._logger.error("Reference error: %s", err)
        else:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as ex:
                futures = [ex.submit(evaluate_reference_group, g) for g in groups]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="References", unit="grp"):
                    v, w, err = future.result()
                    validity.update(v)
                    written += w
                    if err:
                        self._logger.error("Reference error: %s", err)
        self._logger.info("Reference pass: wrote %d reference patch(es)", written)
        return validity

    # ----- pass 3: finalise samples -------------------------------------------

    def _finalise(self, plans: list[TargetPlan],
                  validity: dict[tuple[int, str, int], bool]) -> list[SampleManifest]:
        """Keep each plan's valid references and emit manifests.

        Args:
            plans: All target plans.
            validity: Reference validity map from the reference pass.

        Returns:
            Sample manifests meeting the minimum-reference requirement.
        """
        refs = self.config.references
        manifests: list[SampleManifest] = []
        dropped = 0
        for plan in plans:
            valid_dates = [
                d for d in plan.candidate_dates
                if validity.get((plan.size, d, plan.cell_index), False)
            ][: refs.maximum]
            if len(valid_dates) < refs.minimum:
                dropped += 1
                continue
            manifests.append(SampleManifest(
                target_date=plan.target_date, split=plan.split, size=plan.size,
                cell_index=plan.cell_index, row=plan.row, col=plan.col,
                reference_dates=valid_dates,
                cloud_fraction=plan.cloud_fraction, nodata_fraction=plan.nodata_fraction,
                valid_fraction=plan.valid_fraction, cloud_percentage=plan.cloud_percentage,
                season=plan.season, year=plan.year, month=plan.month,
                day_of_year=plan.day_of_year, crs=plan.crs, transform=plan.transform,
            ))
        self._logger.info("Finalise: %d sample(s) kept, %d dropped (< %d valid refs)",
                          len(manifests), dropped, refs.minimum)
        return manifests

    def _write_samples(self, manifests: list[SampleManifest]) -> None:
        """Write one JSON file per sample.

        Args:
            manifests: All sample manifests.
        """
        manifests.sort(key=lambda m: (m.split, m.target_date, m.size, m.cell_index))
        for index, manifest in enumerate(manifests, start=1):
            sample_id = f"sample_{index:06d}"
            path = self.config.output_dir / ids.sample_relpath(manifest.split, sample_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest.to_sample_json(sample_id), indent=2),
                            encoding="utf-8")
        self._logger.info("Wrote %d sample JSON file(s)", len(manifests))

    def _write_index(self, manifests: list[SampleManifest]) -> None:
        """Write the dataset index CSV (one row per sample).

        Args:
            manifests: All sample manifests.
        """
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, m in enumerate(manifests, start=1):
            sample_id = f"sample_{index:06d}"
            rows.append({
                "sample_id": sample_id, "split": m.split, "target_date": m.target_date,
                "patch_size": m.size, "cell_index": m.cell_index, "row": m.row, "col": m.col,
                "n_references": len(m.reference_dates),
                "reference_dates": ";".join(m.reference_dates),
                "cloud_fraction": round(m.cloud_fraction, 6),
                "valid_fraction": round(m.valid_fraction, 6),
                "sample_json": ids.sample_relpath(m.split, sample_id),
                "target_patch": m.target_key().target_relpath(),
            })
        pd.DataFrame(rows).to_csv(self.config.index_path, index=False, encoding="utf-8")
        self._logger.info("Wrote index: %s (%d rows)", self.config.index_path, len(rows))

    # ----- statistics & storage ------------------------------------------------

    def _estimate_storage(self, plans: list[TargetPlan]) -> dict[str, Any]:
        """Estimate library storage before the reference pass.

        Uses one already-written target patch per size as a size sample and
        projects unique-patch counts from the plans.

        Args:
            plans: All target plans.

        Returns:
            A pre-generation estimate dict.
        """
        sizes = sorted({p.size for p in plans})
        avg_bytes = {s: self._sample_patch_bytes(s) for s in sizes}
        target_cells = {s: len({(p.target_date, p.cell_index) for p in plans if p.size == s})
                        for s in sizes}
        ref_cells = {
            s: len({(d, p.cell_index) for p in plans if p.size == s for d in p.candidate_dates})
            for s in sizes
        }
        estimated = 0
        per_size = {}
        for s in sizes:
            bytes_s = int((target_cells[s] + ref_cells[s]) * avg_bytes[s])
            estimated += bytes_s
            per_size[str(s)] = {
                "target_patches": target_cells[s],
                "candidate_reference_patches": ref_cells[s],
                "avg_patch_bytes": int(avg_bytes[s]),
                "estimated_bytes": bytes_s,
            }
        return {"estimated_bytes": estimated, "estimated_human": _human(estimated),
                "per_patch_size": per_size}

    def _sample_patch_bytes(self, size: int) -> float:
        """Return an average on-disk bytes/patch for a size (from written files).

        Args:
            size: Patch size in pixels.

        Returns:
            Mean file size in bytes (falls back to a raw-based estimate).
        """
        lib = self.config.output_dir / ids.TARGET_LIBRARY_DIR / str(size)
        sizes = [p.stat().st_size for p in lib.rglob("*.npz")][:64]
        if sizes:
            return sum(sizes) / len(sizes)
        return 13 * size * size * 2 * 0.65  # uint16 reflectance @ ~0.65 ratio

    def _compute_statistics(self, manifests: list[SampleManifest],
                            estimate: dict[str, Any]) -> dict[str, Any]:
        """Compute dataset + storage statistics with old-vs-new comparison.

        Args:
            manifests: All sample manifests.
            estimate: The pre-generation storage estimate.

        Returns:
            A JSON-serialisable statistics dictionary.
        """
        sizes = sorted({m.size for m in manifests})
        per_size = {}
        ref_hist_all: Counter[int] = Counter()
        for size in sizes:
            subset = [m for m in manifests if m.size == size]
            by_split: Counter[str] = Counter(m.split for m in subset)
            ref_hist = Counter(len(m.reference_dates) for m in subset)
            ref_hist_all.update(ref_hist)
            per_size[str(size)] = {
                "total_samples": len(subset),
                "training_samples": by_split["train"],
                "validation_samples": by_split["val"],
                "testing_samples": by_split["test"],
                "reference_count_distribution": {str(k): ref_hist[k] for k in sorted(ref_hist)},
                "average_references": round(
                    sum(len(m.reference_dates) for m in subset) / len(subset), 3) if subset else 0,
            }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "architecture": "shared_reference_three_library",
            "patch_sizes": sizes,
            "total_samples": len(manifests),
            "reference_policy": {"minimum": self.config.references.minimum,
                                 "maximum": self.config.references.maximum},
            "reference_count_distribution": {str(k): ref_hist_all[k]
                                             for k in sorted(ref_hist_all)},
            "per_patch_size": per_size,
            "storage": self._storage_report(manifests, estimate),
            "config": self.config.to_dict(),
        }

    def _storage_report(self, manifests: list[SampleManifest],
                        estimate: dict[str, Any]) -> dict[str, Any]:
        """Measure actual library size and compare with the duplicated design.

        Args:
            manifests: All sample manifests.
            estimate: The pre-generation estimate.

        Returns:
            A storage report dict.
        """
        root = self.config.output_dir
        tgt_b, tgt_n = _dir_size(root / ids.TARGET_LIBRARY_DIR)
        ref_b, ref_n = _dir_size(root / ids.REFERENCE_LIBRARY_DIR)
        msk_b, msk_n = _dir_size(root / ids.MASK_LIBRARY_DIR)
        smp_b, smp_n = _dir_size(root / ids.SAMPLES_DIR)
        new_total = tgt_b + ref_b + msk_b + smp_b

        # Duplicated design: each sample embedded (1 target + its references)
        # image patches. Use measured avg patch bytes per size.
        avg_by_size = {int(k): v["avg_patch_bytes"] for k, v in estimate["per_patch_size"].items()}
        old_estimated = 0
        old_instances = 0
        for m in manifests:
            per = avg_by_size.get(m.size, 0)
            instances = 1 + len(m.reference_dates)
            old_instances += instances
            old_estimated += instances * per
        old_estimated += msk_b  # masks would also be stored once per sample (>= this)

        unique_images = tgt_n + ref_n
        raw_uint16 = sum(13 * m.size * m.size * 2 for m in manifests) + \
            sum(13 * m.size * m.size * 2 * len(m.reference_dates) for m in manifests)
        return {
            "target_patches": tgt_n, "reference_patches": ref_n, "mask_patches": msk_n,
            "sample_json_files": smp_n,
            "target_library_bytes": tgt_b, "reference_library_bytes": ref_b,
            "mask_library_bytes": msk_b,
            "new_actual_bytes": new_total, "new_actual_human": _human(new_total),
            "duplicated_image_instances": old_instances,
            "deduplication_factor": round(old_instances / unique_images, 2)
            if unique_images else None,
            "old_estimated_bytes": int(old_estimated),
            "old_estimated_human": _human(int(old_estimated)),
            "savings_human": _human(max(0, int(old_estimated) - new_total)),
            "compression_ratio_vs_raw_uint16": round(new_total / raw_uint16, 4)
            if raw_uint16 else None,
            "pre_generation_estimate_human": estimate["estimated_human"],
        }

    def _write_statistics(self, statistics: dict[str, Any]) -> None:
        """Write the dataset statistics JSON.

        Args:
            statistics: The statistics dictionary.
        """
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.config.statistics_path.write_text(json.dumps(statistics, indent=2),
                                                encoding="utf-8")
        self._logger.info("Wrote statistics: %s", self.config.statistics_path)

    def _write_summary(self, statistics: dict[str, Any], estimate: dict[str, Any],
                       started: datetime, n_plans: int, n_samples: int) -> None:
        """Write the high-level generation summary JSON.

        Args:
            statistics: The dataset statistics.
            estimate: The pre-generation storage estimate.
            started: Build start time.
            n_plans: Number of planned target cells.
            n_samples: Number of finalised samples.
        """
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "patch_sizes": statistics["patch_sizes"],
            "reference_policy": statistics["reference_policy"],
            "planned_target_cells": n_plans,
            "final_samples": n_samples,
            "reference_count_distribution": statistics["reference_count_distribution"],
            "storage_estimate": estimate,
            "storage_actual": statistics["storage"],
        }
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._logger.info("Wrote generation summary: %s", self._summary_path)

    # ----- persistence ---------------------------------------------------------

    def _load_checkpoint(self) -> set[str]:
        """Load the set of planned target dates from the checkpoint."""
        if not self._checkpoint_path.exists():
            return set()
        try:
            return set(json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                       .get("planned", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _persist_checkpoint(self, completed: set[str]) -> None:
        """Atomically persist the planned-targets checkpoint.

        Args:
            completed: Planned target dates.
        """
        tmp = self._checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"planned": sorted(completed)}, indent=2), encoding="utf-8")
        tmp.replace(self._checkpoint_path)

    def _load_plans(self) -> list[TargetPlan]:
        """Load target plans written by previous runs from the sidecar."""
        if not self._plan_path.exists():
            return []
        out: list[TargetPlan] = []
        with self._plan_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(TargetPlan(**json.loads(line)))
        self._logger.info("Loaded %d plan(s) from previous run(s)", len(out))
        return out

    def _append_plans(self, plans: list[TargetPlan]) -> None:
        """Append target plans to the crash-safe sidecar.

        Args:
            plans: Plans to append.
        """
        if not plans:
            return
        with self._plan_path.open("a", encoding="utf-8") as handle:
            for plan in plans:
                handle.write(json.dumps(plan.to_record()) + "\n")

    def _reset_state(self) -> None:
        """Remove checkpoint and plan sidecar for a fresh build."""
        for path in (self._checkpoint_path, self._plan_path):
            path.unlink(missing_ok=True)


def _dir_size(path: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for a directory tree.

    Args:
        path: Directory to measure.

    Returns:
        Total size in bytes and number of files (zeros if absent).
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
