"""Part 5 - manifest-based ground-truth filtering with a 4-state distinction.

The local audit only graded 800 unique patches, so most of the dataset is
UNAUDITED. This module never silently treats unaudited patches as PASS: it keeps
four explicit states (PASS / REVIEW / REJECT / UNAUDITED) and applies a
documented policy. It emits an exclusion set consumable by the existing
``SyntheticDataset(gt_filter=...)`` hook and reports the exact composition used.

Policies:
    * ``audited_pass_only``  - keep only audited PASS patches.
    * ``conservative``       - keep audited PASS plus UNAUDITED patches whose
      recorded native cloud fraction is <= ``native_threshold``; always drop
      REVIEW/REJECT and high-cloud unaudited patches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from s2audit.manifest import parse_patch_key, scan_split


def load_audit_status(audit_manifest: str | Path) -> dict[str, str]:
    """Return ``{patch_id: PASS|REVIEW|REJECT}`` from an audit filter manifest."""
    data = json.loads(Path(audit_manifest).read_text(encoding="utf-8"))
    return dict(data.get("status_by_patch", {}))


def _native_fraction_map(root: Path) -> dict[str, float]:
    """Stream ``_gt_patches.jsonl`` into ``{patch_id: native_cloud_fraction}``."""
    reg = root / "_gt_patches.jsonl"
    out: dict[str, float] = {}
    if not reg.is_file():
        return out
    with reg.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = f"{r.get('date')}_{r.get('cell_index')}"
            out[pid] = r.get("native_cloud_fraction", 1.0)
    return out


def build_exclusion(root: str | Path, audit_manifest: str | Path | None, *,
                    policy: str = "conservative", native_threshold: float = 0.01
                    ) -> dict[str, Any]:
    """Build the exclusion set + 4-state counts for a training policy.

    Args:
        root: Dataset root.
        audit_manifest: Path to ``ground_truth_filter_manifest.json`` (or None).
        policy: ``"conservative"`` or ``"audited_pass_only"``.
        native_threshold: Native cloud-fraction ceiling for keeping UNAUDITED.

    Returns:
        A dict with ``exclude_patch_ids`` (list), ``counts`` (4-state totals),
        ``kept`` (how many of each state are kept), and the policy metadata.
    """
    root = Path(root)
    status = load_audit_status(audit_manifest) if audit_manifest else {}
    native = _native_fraction_map(root)

    all_ids = set(native) | set(status)
    exclude: list[str] = []
    counts = {"PASS": 0, "REVIEW": 0, "REJECT": 0, "UNAUDITED": 0}
    kept = {"PASS": 0, "REVIEW": 0, "REJECT": 0, "UNAUDITED": 0}

    for pid in all_ids:
        st = status.get(pid, "UNAUDITED")
        counts[st] += 1
        keep = False
        if st == "PASS":
            keep = True
        elif st in ("REVIEW", "REJECT"):
            keep = False
        else:  # UNAUDITED
            keep = policy == "conservative" and native.get(pid, 1.0) <= native_threshold
        if keep:
            kept[st] += 1
        else:
            exclude.append(pid)

    return {
        "policy": policy,
        "native_threshold": native_threshold,
        "counts_total": counts,
        "counts_kept": kept,
        "n_excluded": len(exclude),
        "exclude_patch_ids": exclude,
    }


def write_training_filter(exclusion: dict, output_path: str | Path) -> Path:
    """Write a training filter manifest consumable by ``SyntheticDataset``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "policy": exclusion["policy"],
        "native_threshold": exclusion["native_threshold"],
        "counts_total": exclusion["counts_total"],
        "counts_kept": exclusion["counts_kept"],
        "exclude_patch_ids": exclusion["exclude_patch_ids"],
    }, indent=1), encoding="utf-8")
    return path


def tally_used_states(root: str | Path, split: str, sample_files: list[str] | None,
                      status: dict[str, str], native: dict[str, float],
                      native_threshold: float) -> dict[str, int]:
    """Count the 4-state composition of the ground truths actually used.

    Args:
        root: Dataset root.
        split: Split name (used when ``sample_files`` is None).
        sample_files: Explicit manifest paths, or None to scan the split.
        status: Audited status map.
        native: Native-fraction map.
        native_threshold: Threshold used to classify unaudited-kept.

    Returns:
        A dict of PASS/REVIEW/REJECT/UNAUDITED counts among the used samples.
    """
    used = {"PASS": 0, "REVIEW": 0, "REJECT": 0, "UNAUDITED": 0}
    if sample_files is not None:
        keys = []
        for f in sample_files:
            spec = json.loads(Path(f).read_text(encoding="utf-8"))
            k = parse_patch_key(spec.get("ground_truth", "") or "")
            if k:
                keys.append(f"{k[0]}_{k[1]}")
    else:
        keys = [f"{r.gt_key[0]}_{r.gt_key[1]}" for r in scan_split(root, split) if r.gt_key]
    for pid in keys:
        used[status.get(pid, "UNAUDITED")] += 1
    return used
