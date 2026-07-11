"""Orchestration: split, parallel extraction, checkpoint/resume, reporting."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DatasetConfig
from .data_loader import DatasetLoader
from .logging_setup import configure_logging, get_logger
from .models import PatchRecord, SampleOutcome, SampleSpec
from .sample_processor import process_target
from .splitter import assign_splits

#: Filename of the crash-safe per-patch records sidecar (under output_dir).
RECORDS_SIDECAR = "_records.jsonl"


def _worker(payload: tuple[SampleSpec, DatasetConfig]) -> SampleOutcome:
    """Top-level, picklable worker wrapper for the process pool.

    Args:
        payload: A ``(spec, config)`` tuple.

    Returns:
        The :class:`SampleOutcome` for the target.
    """
    spec, config = payload
    return process_target(spec, config)


class DatasetBuilder:
    """Builds the training dataset from stacks, masks and the reference DB."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise the builder.

        Args:
            config: Active dataset configuration.
        """
        self.config = config
        self._logger = get_logger()
        self._records_path = config.output_dir / RECORDS_SIDECAR

    def run(self) -> dict[str, Any]:
        """Run the full dataset-building pipeline.

        Returns:
            The dataset statistics dictionary (also written to disk).

        Raises:
            FileNotFoundError: If required inputs are missing.
            ValueError: If no processable targets are found.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(self.config.log_path)
        self._logger = get_logger()
        self._logger.info("=" * 60)
        self._logger.info("Dataset build starting")
        scales = self.config.patch_scales()
        self._logger.info(
            "scales=%s | refs=%d | workers=%d | formats=%s%s",
            ", ".join(f"{s.size}px/str{s.stride}" for s in scales),
            self.config.n_references, self.config.num_workers,
            "GeoTIFF " if self.config.write_geotiff else "",
            "NPZ" if self.config.write_npz else "",
        )

        specs = DatasetLoader(self.config).build_specs()
        assign_splits(specs, self.config)

        completed = self._load_checkpoint()
        prior_records = self._load_prior_records() if self.config.resume else []
        if not self.config.resume:
            self._reset_state()
            completed = set()
            prior_records = []

        pending = [s for s in specs if s.target_date not in completed]
        self._logger.info(
            "%d target(s) total, %d already complete, %d to process",
            len(specs), len(completed), len(pending),
        )

        new_records = self._execute(pending, completed)
        all_records = prior_records + new_records

        statistics = self._compute_statistics(all_records, specs)
        self._write_index(all_records)
        self._write_statistics(statistics)
        self._logger.info(
            "Dataset build complete: %d sample(s) across %d patch size(s)",
            statistics["overall"]["total_samples"],
            len(statistics["patch_sizes"]),
        )
        return statistics

    def _execute(
        self, pending: list[SampleSpec], completed: set[str]
    ) -> list[PatchRecord]:
        """Process pending targets, persisting progress for crash-safe resume.

        Args:
            pending: Targets not yet completed.
            completed: Mutable set of completed target dates (updated in place).

        Returns:
            All patch records produced in this run.
        """
        records: list[PatchRecord] = []
        if not pending:
            return records

        if self.config.num_workers == 1:
            iterator = (process_target(s, self.config) for s in pending)
            for outcome in tqdm(iterator, total=len(pending), desc="Targets", unit="img"):
                records.extend(self._handle_outcome(outcome, completed))
            return records

        payloads = [(s, self.config) for s in pending]
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(_worker, p): p[0] for p in payloads}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Targets", unit="img"
            ):
                spec = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001 - worker crash guard
                    outcome = SampleOutcome(
                        target_date=spec.target_date, split=spec.split,
                        status="failed", message=f"Worker crashed: {exc}",
                    )
                records.extend(self._handle_outcome(outcome, completed))
        return records

    def _handle_outcome(
        self, outcome: SampleOutcome, completed: set[str]
    ) -> list[PatchRecord]:
        """Log an outcome, persist its records and checkpoint, and return records.

        Args:
            outcome: The target's processing outcome.
            completed: Mutable set of completed target dates.

        Returns:
            The outcome's patch records (empty for aborted/failed targets).
        """
        if outcome.status in ("processed", "skipped"):
            self._append_records(outcome.records)
            completed.add(outcome.target_date)
            self._persist_checkpoint(completed)
            self._logger.info(
                "%s [%s]: %s (%.1fs)",
                outcome.target_date, outcome.split, outcome.message,
                outcome.duration_sec or 0.0,
            )
        else:
            # Aborted/failed targets are not checkpointed, so they are retried
            # on the next run rather than silently skipped.
            self._logger.error(
                "%s [%s]: %s", outcome.target_date, outcome.split, outcome.message
            )
        return outcome.records

    # ----- persistence helpers -------------------------------------------------

    def _load_checkpoint(self) -> set[str]:
        """Load the set of completed target dates from the checkpoint file.

        Returns:
            The completed target dates (empty if no checkpoint exists).
        """
        path = self.config.checkpoint_path
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return set(data.get("completed", []))
        except (json.JSONDecodeError, OSError) as exc:
            self._logger.warning("Could not read checkpoint %s: %s", path, exc)
            return set()

    def _persist_checkpoint(self, completed: set[str]) -> None:
        """Atomically write the checkpoint of completed target dates.

        Args:
            completed: The completed target dates.
        """
        path = self.config.checkpoint_path
        tmp = path.with_suffix(".json.tmp")
        payload = {
            "completed": sorted(completed),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_prior_records(self) -> list[PatchRecord]:
        """Load patch records produced by previous runs from the sidecar.

        Returns:
            The previously written patch records (empty if none).
        """
        if not self._records_path.exists():
            return []
        records: list[PatchRecord] = []
        with self._records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    data["reference_dates"] = data.get("reference_dates", [])
                    records.append(PatchRecord(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    self._logger.warning("Skipping bad record line: %s", exc)
        self._logger.info("Loaded %d record(s) from previous run(s)", len(records))
        return records

    def _append_records(self, records: list[PatchRecord]) -> None:
        """Append patch records to the crash-safe sidecar file.

        Args:
            records: The records to append.
        """
        if not records:
            return
        with self._records_path.open("a", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "target_date": record.target_date,
                    "reference_dates": record.reference_dates,
                    "patch_index": record.patch_index,
                    "row": record.row,
                    "col": record.col,
                    "patch_size": record.patch_size,
                    "cloud_fraction": record.cloud_fraction,
                    "nodata_fraction": record.nodata_fraction,
                    "valid_fraction": record.valid_fraction,
                    "geotiff_dir": record.geotiff_dir,
                    "npz_path": record.npz_path,
                }
                handle.write(json.dumps(payload) + "\n")

    def _reset_state(self) -> None:
        """Remove checkpoint and records sidecar for a fresh (non-resume) run."""
        for path in (self.config.checkpoint_path, self._records_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                self._logger.warning("Could not remove %s: %s", path, exc)
        self._logger.info("Resume disabled: cleared previous checkpoint/records")

    # ----- reporting -----------------------------------------------------------

    def _write_index(self, records: list[PatchRecord]) -> None:
        """Write the dataset index CSV.

        Args:
            records: All patch records.
        """
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        rows = [r.to_row() for r in records]
        columns = [
            "sample_id", "split", "target_date", "reference_dates", "patch_index",
            "row", "col", "patch_size", "cloud_fraction", "nodata_fraction",
            "valid_fraction", "geotiff_dir", "npz_path",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame = frame.sort_values(["split", "sample_id"]).reset_index(drop=True)
        frame.to_csv(self.config.index_path, index=False, encoding="utf-8")
        self._logger.info("Wrote dataset index: %s (%d rows)",
                          self.config.index_path, len(frame))

    def _compute_statistics(
        self, records: list[PatchRecord], specs: list[SampleSpec]
    ) -> dict[str, Any]:
        """Compute the dataset statistics summary, broken down by patch size.

        Args:
            records: All patch records.
            specs: All sample specs (unused here but kept for extensibility).

        Returns:
            A JSON-serialisable statistics dictionary with an overall block and
            a ``per_patch_size`` mapping (one stats block per configured size).
        """
        stride_by_size = {s.size: s.stride for s in self.config.patch_scales()}
        sizes = sorted({r.patch_size for r in records}) or list(stride_by_size)

        per_size: dict[str, Any] = {}
        for size in sizes:
            subset = [r for r in records if r.patch_size == size]
            per_size[str(size)] = self._stats_block(
                subset, size, stride_by_size.get(size)
            )

        overall = self._stats_block(records, None, None)
        overall.pop("patch_size", None)
        overall.pop("stride", None)
        overall.pop("average_valid_pixels", None)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "patch_sizes": sizes,
            "overall": overall,
            "per_patch_size": per_size,
            "average_references": self.config.n_references,
            "split_fractions": {
                "train": round(self.config.split.train, 4),
                "val": round(self.config.split.val, 4),
                "test": round(self.config.split.test, 4),
            },
            "config": self.config.to_dict(),
        }

    def _stats_block(
        self, records: list[PatchRecord], size: int | None, stride: int | None
    ) -> dict[str, Any]:
        """Compute one statistics block for a subset of records.

        Args:
            records: The records in this group (e.g. one patch size).
            size: The patch size for this group, or ``None`` for an overall
                block where a single size is not meaningful.
            stride: The stride for this group, or ``None``.

        Returns:
            A statistics dictionary for the group.
        """
        by_split = {"train": 0, "val": 0, "test": 0}
        for record in records:
            by_split[record.split] = by_split.get(record.split, 0) + 1

        clouds = np.array([r.cloud_fraction for r in records]) if records else np.array([])
        nodata = np.array([r.nodata_fraction for r in records]) if records else np.array([])
        valids = np.array([r.valid_fraction for r in records]) if records else np.array([])

        def _mean_pct(arr: np.ndarray) -> float | None:
            return round(float(arr.mean()) * 100.0, 4) if arr.size else None

        avg_valid_pixels = (
            round(float(valids.mean()) * size * size, 2)
            if valids.size and size is not None
            else None
        )
        return {
            "total_samples": len(records),
            "training_samples": by_split.get("train", 0),
            "validation_samples": by_split.get("val", 0),
            "testing_samples": by_split.get("test", 0),
            "average_cloud_coverage_percent": _mean_pct(clouds),
            "average_cloud_percentage": _mean_pct(clouds),
            "average_nodata_percentage": _mean_pct(nodata),
            "average_valid_pixels": avg_valid_pixels,
            "patch_size": size,
            "stride": stride,
        }

    def _write_statistics(self, statistics: dict[str, Any]) -> None:
        """Write the dataset statistics JSON.

        Args:
            statistics: The statistics dictionary.
        """
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.config.statistics_path.write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
        self._logger.info("Wrote dataset statistics: %s", self.config.statistics_path)
