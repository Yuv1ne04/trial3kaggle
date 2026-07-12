# s2audit — Baseline Scientific Audit & Evaluation

A pre-TCR-Net audit that determines whether the current baseline metrics are
**scientifically trustworthy for operational sugar-cane monitoring in Mauritius**.
It never retrains, never mutates the checkpoint, and never regenerates the
dataset — it streams the existing manifests/libraries and writes a self-contained
`audit/` report tree.

## Run

```bash
# Full run (Kaggle GPU, whole test split, all unique GT patches, full leakage scan)
python -m s2audit --dataset /kaggle/input/.../synthetic_dataset \
    --checkpoint /kaggle/input/.../best.pt --output /kaggle/working/audit --full

# Quick local smoke (CPU, small caps)
python -m s2audit --dataset DATA --checkpoint best.pt --output audit \
    --max-samples 240 --gt-max-patches 800 --leakage-max-samples 5000 --device cpu
```

Select parts with `--parts gt,leakage,eval`. Memory is O(1) in the split size
(scalar accumulators + per-sample macro lists), so the full 89k test set streams.

## What it produces (`audit/`)

| Artefact | Part | Contents |
|---|---|---|
| `ground_truth_quality.csv` | 1 | per unique GT patch: brightness/cirrus/suspected cloud fractions, quality score, PASS/REVIEW/REJECT |
| `ground_truth_filter_manifest.json` | 1 | excluded patch ids for optional clean-training |
| `visual_ground_truth_audit/` | 1 | ≥100 PASS/REVIEW/REJECT RGB thumbnails each |
| `test_metrics.csv` | 3 | region metrics, **micro (pixel-weighted) and macro (per-sample)** |
| `per_band_metrics.csv` | 5 | per band: MAE/RMSE/bias/PSNR/correlation (cloud + whole) |
| `vegetation_metrics.csv` | 4 | NDVI/NDRE/EVI/NDWI/NDMI: MAE/RMSE/bias/Pearson (cloud + land) |
| `baseline_comparison.csv` | 6 | U-Net vs cloudy-input / weighted-mean / nearest-temporal / best-single-oracle |
| `stratified_metrics.csv` | 7 | by difficulty, coverage bin, reference count, surface category |
| `data_leakage_audit.json` | 9 | split-crossing checks (PASS/WARNING/FAIL) |
| `test_evaluation_report.json` | 8 | full report + provenance (checkpoint hash, git commit, config, seed, timestamp) |
| `visual_reconstruction/` | 2 | enhanced panels: fixed + 2–98% RGB, signed/abs/cloud-only error, NDVI GT/pred/error |
| `scientific_audit_summary.json` | 10 | overall PASS / PASS_WITH_WARNINGS / FAIL + answers to the 8 questions |

## Key methodology (fixes over the training metrics)

- **Micro vs macro** — the primary dataset score is pixel-weighted (micro).
  Averaging per-batch/per-sample PSNR (macro) is reported alongside but is *not*
  the headline; the two diverge sharply here.
- **Cloud-region first** — cloud-region metrics are primary; whole-image
  secondary; **clear-region metrics are flagged non-informative** under hard
  compositing (clear pixels are copied verbatim).
- **SAM** — computed in double precision with the cosine clamped to `1.0` (no
  `1-1e-6` floor), so identical spectra give ~0 rad; background pixels excluded.
- **ERGAS** — excludes bands whose regional mean reflectance is near zero
  (otherwise they explode the ratio), reports per-band components, an
  operational-surface-band variant, and warnings — never a silent huge value.
- **Surface classification** — provisional land/water via NDWI on the clean
  ground truth (no external land mask needed); extensible to an MSIRI field
  polygon layer.
- **Prediction sanity** — reports the fraction of predicted reflectance outside
  `[0, 1]` (unphysical output).

## Module map

```
indices.py         vegetation indices + spectral land/water/background masks
metrics.py         PairAccumulator, RegionAccumulator, ErgasAccumulator, SSIM/MS-SSIM
baselines.py       non-learned reference baselines (composited identically)
manifest.py        streaming sample-manifest reader (metadata only)
gt_quality.py      Part 1 ground-truth cloud audit (registry-driven)
leakage.py         Part 9 cross-split leakage audit
stratify.py        Part 7 stratification keys
visualize.py       Part 2 enhanced panels
evaluate.py        Parts 3–8 streaming orchestrator
report.py          Part 10 synthesis
cli.py / __main__  command-line entry
```

## Optional clean-training

```python
from s2train.datasets import SyntheticDataset
ds = SyntheticDataset(root, split="train",
                      gt_filter="audit/ground_truth_filter_manifest.json")  # drops REVIEW/REJECT
```

The `gt_filter` hook is opt-in and fully backward compatible (default `None`).
