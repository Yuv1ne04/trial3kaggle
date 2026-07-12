"""Data-leakage audit across train / validation / test splits (Part 9).

Verifies the split is scientifically clean for an operational forecast setting:
targets and ground-truth patches must not cross splits, augmentation variants of
one patch must stay together, and the cloud-mask transplant must not import
target signal. References are *past* observations, so a test reference that also
appears as a training target is operationally normal - it is reported as a note,
not a failure, unless the same scene is a *target* in two splits.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import available_splits, scan_split


def _status(fail: bool, warn: bool) -> str:
    return "FAIL" if fail else ("WARNING" if warn else "PASS")


def audit_leakage(root: Path | str, *, max_samples: int = 0) -> dict[str, Any]:
    """Run the cross-split leakage checks.

    Args:
        root: Dataset root.
        max_samples: Per-split cap for a quick scan (0 = all).

    Returns:
        A report dict (also written as ``data_leakage_audit.json``).
    """
    splits = available_splits(root)
    target_split: dict[str, set] = defaultdict(set)   # target_date -> splits
    gt_split: dict[tuple, set] = defaultdict(set)      # (gt_date,cell) -> splits
    gt_variants: dict[tuple, set] = defaultdict(set)   # (gt_date,cell) -> sample splits
    ref_keys_by_split: dict[str, set] = defaultdict(set)
    gt_targets_by_split: dict[str, set] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    cloud_same_scene = 0

    for split in splits:
        for rec in scan_split(root, split, max_samples=max_samples):
            counts[split] += 1
            if rec.target_date:
                target_split[rec.target_date].add(split)
            if rec.gt_key:
                gt_split[rec.gt_key].add(split)
                gt_variants[rec.gt_key].add(split)
                gt_targets_by_split[split].add(rec.gt_key)
            for rk in rec.ref_keys:
                ref_keys_by_split[split].add(rk)
            # Cloud transplant must come from a different scene than the GT.
            if rec.cloud_key and rec.gt_key and rec.cloud_key == rec.gt_key:
                cloud_same_scene += 1

    # 1) Target acquisition dates crossing splits.
    crossing_target_dates = sorted(d for d, s in target_split.items() if len(s) > 1)
    # 2) Ground-truth (date,cell) crossing splits.
    crossing_gt = sorted((f"{d}_{c}") for (d, c), s in gt_split.items() if len(s) > 1)
    # 3) Augmentation variants of one patch in >1 split.
    split_variant_patches = sorted(f"{d}_{c}" for (d, c), s in gt_variants.items() if len(s) > 1)
    # 4) Test references that are targets in another split (note, not fail).
    ref_target_overlap = {}
    for split in splits:
        others = set().union(*(gt_targets_by_split[o] for o in splits if o != split)) \
            if len(splits) > 1 else set()
        overlap = ref_keys_by_split[split] & others
        if overlap:
            ref_target_overlap[split] = len(overlap)

    checks = {
        "target_dates_cross_split": {
            "status": _status(bool(crossing_target_dates), False),
            "n_crossing": len(crossing_target_dates),
            "examples": crossing_target_dates[:10],
            "detail": "A target acquisition date must belong to exactly one split.",
        },
        "ground_truth_patch_cross_split": {
            "status": _status(bool(crossing_gt), False),
            "n_crossing": len(crossing_gt),
            "examples": crossing_gt[:10],
            "detail": "A (date, cell) ground-truth patch must not be a target in two splits.",
        },
        "augmentation_variants_same_split": {
            "status": _status(bool(split_variant_patches), False),
            "n_split": len(split_variant_patches),
            "detail": "All synthetic variants of one ground-truth patch must share a split.",
        },
        "reference_target_overlap": {
            "status": _status(False, bool(ref_target_overlap)),
            "per_split_overlap": ref_target_overlap,
            "detail": ("References are past observations; overlap with another "
                       "split's targets is operationally expected. Reported as a "
                       "note. It becomes a concern only if a scene is a *target* "
                       "in two splits (see ground_truth_patch_cross_split)."),
        },
        "cloud_mask_target_leakage": {
            "status": _status(False, cloud_same_scene > 0),
            "n_same_scene_transplant": cloud_same_scene,
            "detail": ("Applied cloud tile should come from a different scene than "
                       "the ground truth, so no target reflectance is imported."),
        },
    }

    fails = [k for k, v in checks.items() if v["status"] == "FAIL"]
    warns = [k for k, v in checks.items() if v["status"] == "WARNING"]
    overall = _status(bool(fails), bool(warns))

    return {
        "overall_status": overall,
        "splits_scanned": {s: counts[s] for s in splits},
        "max_samples_per_split": max_samples or "all",
        "note_validation_present": "validation" in splits,
        "checks": checks,
        "failed_checks": fails,
        "warning_checks": warns,
    }
