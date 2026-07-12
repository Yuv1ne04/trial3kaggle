"""Ground-truth cloud-contamination audit (Part 1).

The synthetic supervision assumes each ``ground_truth`` patch is cloud-free. The
validation panels suggest some patches still carry thin/natural cloud. This
module independently re-estimates cloud risk for every *unique* ground-truth
patch (deduplicated across the manifests - not once per synthetic sample) using
spectral tests that do not depend on the original generation threshold, then
grades each patch PASS / REVIEW / REJECT and emits a filter manifest the
PyTorch dataset can optionally honour.

Independent cloud-risk indicators (over valid, non-water pixels):
    * brightness   - bright visible reflectance (clouds are bright).
    * whiteness    - bright *and* spectrally flat across RGB (clouds are white).
    * cirrus (B10) - high B10 reflectance (thin cirrus; negligible surface term).
    * low NDVI     - bright vegetation-free response (clouds suppress NDVI).
An optional s2cloudless second opinion is used when the package is importable.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import BAND_INDEX

_B = BAND_INDEX
_BLUE, _GREEN, _RED, _NIR, _CIRRUS, _SWIR1 = (
    _B["B02"], _B["B03"], _B["B04"], _B["B08"], _B["B10"], _B["B11"])


@dataclass
class QualityThresholds:
    """Documented, tunable thresholds for the cloud-risk tests (reflectance)."""

    bright: float = 0.30          # visible mean above this = bright
    white_std: float = 0.030      # RGB std below this (and bright) = white
    cirrus: float = 0.015         # B10 above this = cirrus-flagged
    ndvi_low: float = 0.20        # NDVI below this (and bright) supports cloud
    background: float = 1.5e-3    # all-band reflectance below this = NoData
    review_at: float = 0.02       # suspected fraction >= this -> REVIEW
    reject_at: float = 0.10       # suspected fraction >= this -> REJECT


@dataclass
class _Reservoir:
    """Reproducible reservoir sample of patch ids per quality class."""

    k: int
    rng: random.Random
    seen: int = 0
    items: list = field(default_factory=list)

    def offer(self, item: Any) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.k:
                self.items[j] = item


def _patch_path(root: Path, key: tuple[str, int]) -> Path:
    date, cell = key
    return root / "patch_library" / "256" / date / f"patch_{cell:06d}.npz"


def _load_patch(path: Path, scale: float) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            return data["image"].astype(np.float32) / scale
    except (OSError, KeyError, ValueError):
        return None


def _cloud_indicators(img: np.ndarray, thr: QualityThresholds) -> dict[str, float]:
    """Compute per-patch cloud-risk fractions over valid pixels.

    Args:
        img: Reflectance ``(13, H, W)`` in ``[0, 1]``.
        thr: Thresholds.

    Returns:
        A dict of fractions and the combined suspected-cloud fraction.
    """
    visible = img[[_BLUE, _GREEN, _RED]]
    vis_mean = visible.mean(axis=0)
    vis_std = visible.std(axis=0)
    background = (img < thr.background).all(axis=0)
    valid = ~background
    n_valid = max(int(valid.sum()), 1)

    ndwi_denom = img[_GREEN] + img[_NIR]
    ndwi = np.where(np.abs(ndwi_denom) > 2e-3, (img[_GREEN] - img[_NIR]) / (ndwi_denom + 1e-6), 0.0)
    water = ndwi > 0.0
    ndvi_denom = img[_NIR] + img[_RED]
    ndvi = np.where(np.abs(ndvi_denom) > 2e-3, (img[_NIR] - img[_RED]) / (ndvi_denom + 1e-6), 0.0)

    bright = (vis_mean > thr.bright) & valid & (~water)
    white = bright & (vis_std < thr.white_std)
    cirrus = (img[_CIRRUS] > thr.cirrus) & valid
    suspected = (white & (ndvi < thr.ndvi_low)) | cirrus

    return {
        "brightness_fraction": float(bright.sum()) / n_valid,
        "whiteness_fraction": float(white.sum()) / n_valid,
        "cirrus_fraction": float(cirrus.sum()) / n_valid,
        "suspected_cloud_fraction": float(suspected.sum()) / n_valid,
        "valid_fraction": n_valid / img[0].size,
    }


def _status(suspected: float, existing: float, thr: QualityThresholds) -> str:
    worst = max(suspected, existing if existing is not None else 0.0)
    if worst >= thr.reject_at:
        return "REJECT"
    if worst >= thr.review_at:
        return "REVIEW"
    return "PASS"


def _s2cloudless_prob(img: np.ndarray) -> float | None:
    """Optional s2cloudless second opinion (mean cloud probability)."""
    try:
        from s2cloudless import S2PixelCloudDetector
    except Exception:
        return None
    try:
        # s2cloudless expects (H, W, 10) TOA in [0,1] for its 10 bands:
        # B01 B02 B04 B05 B08 B8A B09 B10 B11 B12.
        order = [_B["B01"], _B["B02"], _B["B04"], _B["B05"], _B["B08"],
                 _B["B8A"], _B["B09"], _B["B10"], _B["B11"], _B["B12"]]
        stack = np.transpose(img[order], (1, 2, 0))[None]
        det = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
        return float(det.get_cloud_probability_maps(stack)[0].mean())
    except Exception:
        return None


def audit_ground_truth(root: Path | str, output_dir: Path | str, *,
                       max_patches: int = 0,
                       reflectance_scale: float = 10000.0,
                       thresholds: QualityThresholds | None = None,
                       use_s2cloudless: bool = False, seed: int = 1234,
                       visual_per_class: int = 100) -> dict[str, Any]:
    """Audit unique ground-truth patches and write the quality artefacts.

    The unique ground-truth patches are taken from the ``_gt_patches.jsonl``
    registry (one record per (date, cell) with its recorded native cloud
    fraction) - not by scanning the 89k synthetic manifests. Each patch's image
    is streamed once for the independent spectral test.

    Args:
        root: Dataset root.
        output_dir: Audit output directory.
        max_patches: Cap on unique patches (0 = all). When set, a reproducible
            random sample across the whole registry is taken (representative of
            every date), streamed one patch at a time.
        reflectance_scale: DN -> reflectance divisor.
        thresholds: Cloud-risk thresholds.
        use_s2cloudless: Compute the optional second opinion when available.
        seed: Reproducible sampling seed.
        visual_per_class: Patches per class in the visual audit panel.

    Returns:
        A summary dict (counts, fractions, artefact paths).
    """
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thr = thresholds or QualityThresholds()
    rng = random.Random(seed)

    # 1) Draw unique ground-truth patches from the registry (representative of
    #    all dates); reservoir-sample when capped so memory stays bounded.
    records = _sample_registry(root, max_patches, seed)

    reservoirs = {c: _Reservoir(visual_per_class, rng) for c in ("PASS", "REVIEW", "REJECT")}
    counts = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
    sum_suspected = 0.0
    n_written = 0
    csv_path = output_dir / "ground_truth_quality.csv"
    filter_exclude: list[str] = []
    status_by_patch: dict[str, str] = {}

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patch_id", "target_date", "existing_cloud_fraction",
                         "secondary_cloud_fraction", "brightness_fraction",
                         "cirrus_fraction", "suspected_cloud_fraction",
                         "quality_score", "quality_status"])
        for rec in records:
            key = (rec["date"], rec["cell_index"])
            img = _load_patch(_patch_path(root, key), reflectance_scale)
            if img is None:
                continue
            ind = _cloud_indicators(img, thr)
            secondary = _s2cloudless_prob(img) if use_s2cloudless else None
            existing = rec.get("native_cloud_fraction")
            suspected = ind["suspected_cloud_fraction"]
            status = _status(suspected, existing, thr)
            score = round(1.0 - max(suspected, existing or 0.0), 6)
            patch_id = f"{key[0]}_{key[1]}"

            writer.writerow([
                patch_id, key[0],
                "" if existing is None else round(existing, 6),
                "" if secondary is None else round(secondary, 6),
                round(ind["brightness_fraction"], 6), round(ind["cirrus_fraction"], 6),
                round(suspected, 6), score, status])

            counts[status] += 1
            sum_suspected += suspected
            n_written += 1
            status_by_patch[patch_id] = status
            reservoirs[status].offer((patch_id, key))
            if status in ("REVIEW", "REJECT"):
                filter_exclude.append(patch_id)

    # 2) Filter manifest for optional dataset exclusion.
    manifest = {
        "version": 1,
        "source": "_gt_patches.jsonl unique ground-truth registry",
        "reflectance_scale": reflectance_scale,
        "thresholds": thr.__dict__,
        "policy": "Exclude REVIEW and REJECT ground-truth patches from training.",
        "counts": counts,
        "n_unique_patches_audited": n_written,
        "exclude_patch_ids": sorted(set(filter_exclude)),
        "status_by_patch": status_by_patch,
    }
    (output_dir / "ground_truth_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # 3) Visual audit panels.
    visual_dir = output_dir / "visual_ground_truth_audit"
    made = _render_visual_audit(root, reservoirs, visual_dir, reflectance_scale)

    summary = {
        "n_unique_patches_audited": n_written,
        "counts": counts,
        "fraction": {k: round(v / max(1, n_written), 4) for k, v in counts.items()},
        "mean_suspected_cloud_fraction": round(sum_suspected / max(1, n_written), 6),
        "s2cloudless_used": bool(use_s2cloudless and _s2cloudless_available()),
        "artefacts": {
            "csv": str(csv_path),
            "filter_manifest": str(output_dir / "ground_truth_filter_manifest.json"),
            "visual_dir": str(visual_dir),
            "visual_panels": made,
        },
    }
    (output_dir / "ground_truth_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _s2cloudless_available() -> bool:
    try:
        import s2cloudless  # noqa: F401
        return True
    except Exception:
        return False


def _iter_registry(root: Path):
    """Yield unique GT records from ``_gt_patches.jsonl`` (date, cell, fraction)."""
    reg = root / "_gt_patches.jsonl"
    if not reg.is_file():
        return
    with reg.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _sample_registry(root: Path, max_patches: int, seed: int) -> list[dict]:
    """Return registry records, reservoir-sampled to ``max_patches`` when set.

    Args:
        root: Dataset root.
        max_patches: Cap (0 = all records, streamed order).
        seed: Reproducible sampling seed.

    Returns:
        A list of GT registry records.
    """
    if not max_patches:
        return list(_iter_registry(root))
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, rec in enumerate(_iter_registry(root)):
        if len(reservoir) < max_patches:
            reservoir.append(rec)
        else:
            j = rng.randint(0, i)
            if j < max_patches:
                reservoir[j] = rec
    return reservoir


def _rgb(img: np.ndarray) -> np.ndarray:
    """Percentile-stretched RGB (B04, B03, B02) for a thumbnail."""
    rgb = img[[_RED, _GREEN, _BLUE]].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _render_visual_audit(root: Path, reservoirs: dict, visual_dir: Path,
                         scale: float) -> dict[str, str]:
    """Render one RGB grid per quality class (PASS/REVIEW/REJECT)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    visual_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    for status, res in reservoirs.items():
        items = res.items
        if not items:
            continue
        cols = 10
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (patch_id, key) in zip(axes, items):
            img = _load_patch(_patch_path(root, key), scale)
            if img is None:
                continue
            ax.imshow(_rgb(img))
            ax.set_title(patch_id, fontsize=4)
        fig.suptitle(f"Ground-truth audit - {status} (n={len(items)})", fontsize=10)
        fig.tight_layout()
        out = visual_dir / f"gt_audit_{status.lower()}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        made[status] = str(out)
    return made
