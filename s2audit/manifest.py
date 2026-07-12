"""Streaming reader for the synthetic sample manifests (metadata only).

Reads each ``samples/<split>/*.json`` once, extracting the light metadata every
audit part needs (split assignment, dates, cell, difficulty, coverage, and the
``(date, cell)`` keys of the ground truth, references and applied cloud tile).
Never loads image arrays. Directory access uses ``os.scandir`` so the 89k-file
test split does not trigger an expensive recursive glob.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_PATCH_KEY = re.compile(r"/(\d{8})/patch_(\d+)\.npz$")


def parse_patch_key(relpath: str) -> tuple[str, int] | None:
    """Extract ``(date, cell_index)`` from a library patch relative path."""
    m = _PATCH_KEY.search(relpath.replace("\\", "/"))
    return (m.group(1), int(m.group(2))) if m else None


@dataclass
class SampleRecord:
    """Light per-sample metadata for auditing (no arrays)."""

    split: str
    sample_id: str
    path: str
    target_date: str | None
    gt_date: str | None
    cell: int | None
    difficulty: str | None
    coverage: float | None
    n_references: int | None
    gt_key: tuple[str, int] | None
    ref_keys: list[tuple[str, int]]
    cloud_key: tuple[str, int] | None
    season: str | None


def _record(split: str, path: Path, spec: dict) -> SampleRecord:
    meta = spec.get("metadata", {})
    gt_key = parse_patch_key(spec.get("ground_truth", "") or "")
    ref_keys = [k for k in (parse_patch_key(r) for r in spec.get("references", [])) if k]
    cloud_key = parse_patch_key(spec.get("cloud_tile", "") or "")
    return SampleRecord(
        split=split,
        sample_id=meta.get("sample_id", path.stem),
        path=str(path),
        target_date=meta.get("target_date"),
        gt_date=meta.get("ground_truth_date"),
        cell=meta.get("cell_index"),
        difficulty=meta.get("difficulty"),
        coverage=meta.get("applied_cloud_coverage", meta.get("cloud_percentage")),
        n_references=meta.get("n_references"),
        gt_key=gt_key,
        ref_keys=ref_keys,
        cloud_key=cloud_key,
        season=meta.get("season"),
    )


def scan_split(root: Path | str, split: str, *, max_samples: int = 0) -> Iterator[SampleRecord]:
    """Yield :class:`SampleRecord` for a split, optionally capped.

    Args:
        root: Dataset root.
        split: ``train`` / ``validation`` / ``test``.
        max_samples: Cap (0 = all). Files are visited in sorted name order for
            reproducibility.

    Yields:
        One :class:`SampleRecord` per manifest.
    """
    split_dir = Path(root) / "samples" / split
    if not split_dir.is_dir():
        return
    names = sorted(e.name for e in os.scandir(split_dir) if e.name.endswith(".json"))
    if max_samples:
        names = names[:max_samples]
    for name in names:
        path = split_dir / name
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        yield _record(split, path, spec)


def available_splits(root: Path | str) -> list[str]:
    """Return the split folders that exist under ``samples/``."""
    base = Path(root) / "samples"
    if not base.is_dir():
        return []
    return sorted(e.name for e in os.scandir(base) if e.is_dir())
