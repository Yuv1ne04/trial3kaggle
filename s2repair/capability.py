"""Part 4 - reference-input capability report.

Inspects the *existing* manifests, registry and shared libraries (no dataset
regeneration, no field invention) to record which reference-related inputs are
actually available, derivable, recoverable, or missing. Reference local cloud
fractions are recoverable from ``_gt_patches.jsonl`` by (date, cell) without
duplicating any image data, so a helper to attach them is exposed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from s2audit.manifest import parse_patch_key, scan_split


def _native_fraction_map(root: Path) -> dict[str, float]:
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
            out[f"{r.get('date')}_{r.get('cell_index')}"] = r.get("native_cloud_fraction", 1.0)
    return out


def reference_cloud_fractions(root: str | Path, reference_keys: list[tuple[str, int]],
                              native: dict[str, float] | None = None) -> list[float]:
    """Recover per-reference native cloud fractions from the registry.

    Args:
        root: Dataset root.
        reference_keys: List of ``(date, cell)`` reference keys.
        native: Optional pre-loaded native-fraction map.

    Returns:
        A list of cloud fractions (``nan`` where unknown), one per reference.
    """
    native = native if native is not None else _native_fraction_map(Path(root))
    return [native.get(f"{d}_{c}", float("nan")) for d, c in reference_keys]


def build_capability_report(root: str | Path, output_dir: str | Path, *,
                            sample_manifests: int = 200) -> dict[str, Any]:
    """Inspect the dataset and write ``reference_input_capability_report.json``.

    Args:
        root: Dataset root.
        output_dir: Output directory.
        sample_manifests: Number of manifests to inspect for field presence.

    Returns:
        The capability report dict.
    """
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inspect a sample of manifests for reference-related fields.
    has_refs = has_dates = has_nref = False
    n = 0
    split = "test" if (root / "samples" / "test").is_dir() else "train"
    for rec in scan_split(root, split, max_samples=sample_manifests):
        n += 1
        raw = json.loads(Path(rec.path).read_text(encoding="utf-8"))
        meta = raw.get("metadata", {})
        has_refs = has_refs or bool(raw.get("references"))
        has_dates = has_dates or bool(meta.get("reference_dates"))
        has_nref = has_nref or (meta.get("n_references") is not None)

    native = _native_fraction_map(root)
    mask_library = (root / "cloud_tile_library").is_dir()

    capability = {
        "reference_images": {
            "status": "AVAILABLE" if has_refs else "MISSING",
            "source": "manifest 'references' -> patch_library npz"},
        "reference_validity_mask": {
            "status": "AVAILABLE",
            "source": "constructed by the loader (1 per real reference slot)"},
        "reference_dates": {
            "status": "AVAILABLE" if has_dates else "MISSING",
            "source": "manifest metadata 'reference_dates'"},
        "time_differences": {
            "status": "DERIVABLE",
            "source": "target_date - reference_dates (both in manifest)"},
        "reference_selection_scores": {
            "status": "MISSING",
            "source": "not stored in manifests; would require re-running s2refselect"},
        "reference_cloud_masks_perpixel": {
            "status": "MISSING",
            "source": ("per-pixel reference cloud masks are not stored; recovering "
                       "them would require regenerating from the shared cloud library "
                       "(disallowed).")},
        "reference_local_cloud_fractions": {
            "status": "RECOVERABLE" if native else "MISSING",
            "source": ("_gt_patches.jsonl native_cloud_fraction keyed by (date, cell); "
                       "exposed via s2repair.capability.reference_cloud_fractions() "
                       "without duplicating image data.")},
    }

    report = {
        "dataset_root": str(root),
        "manifests_inspected": n,
        "registry_patches_with_native_fraction": len(native),
        "cloud_tile_library_present": mask_library,
        "capability": capability,
        "recommended_reference_weighting": (
            "Keep fusion simple: validity-weighted mean (current). Optionally down-weight "
            "references by recovered native cloud fraction or temporal distance - both are "
            "available/derivable without new image data. No attention/transformers."),
    }
    (output_dir / "reference_input_capability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
