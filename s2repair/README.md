# s2repair — Physically-constrained baseline repair

Repairs the audited v1 baseline (which emitted 31% negative reflectance and lost
to the weighted-reference mean) into a **physically-bounded** model, and
validates it through three training gates before any scaled run. Never modifies
`best.pt` or the dataset.

## The fix in one line

```
pred = sigmoid( logit(clamp(B, eps, 1-eps)) + residual_scale * tanh(delta) )
```

where `B` = weighted-reference composite (the physical prior) and `delta` is the
network output. `sigmoid` makes every reflectance lie in (0, 1) by construction;
the residual head is zero-initialised so training **starts at the
weighted-reference mean** (`ReferenceResidualUNetV2` in `s2train/models`).

## Why (the audit + Part-1 diagnostic)

Post-hoc bounding of the *unchanged* checkpoint recovered ~11 dB, proving the
failure was dominated by unbounded outputs:

| best.pt output | cloud PSNR | cloud RMSE | NDVI MAE | neg frac |
|---|---|---|---|---|
| raw | 16.7 dB | 0.146 | 0.469 | 30.7% |
| clamp [0,1] | 26.0 dB | 0.050 | 0.327 | 0% |
| per-band bounds | 28.3 dB | 0.039 | 0.327 | 0% |

Clamped, best.pt already beats the weighted-reference mean (25.3 dB) — so a
properly bounded, trainable model is well-motivated.

## Commands

```bash
# Part 1 - bounded-output diagnostic (does not modify best.pt)
python -m s2repair diagnose   --checkpoint best.pt --dataset DATA --output repair

# Part 2 - worst-case failure analysis + panels
python -m s2repair worstcase  --checkpoint best.pt --dataset DATA --output repair

# Part 4 - reference-input capability report
python -m s2repair capability --dataset DATA --output repair

# Part 8 - gates (train the bounded v2; each refuses to auto-continue on FAIL)
python -m s2repair gate1 --config configs/reference_unet_v2_gate1.yaml \
    --dataset DATA --output repair --audit-manifest audit/ground_truth_filter_manifest.json
python -m s2repair gate2 --config configs/reference_unet_v2_gate2.yaml --dataset DATA --output repair
python -m s2repair gate3 --config configs/reference_unet_v2_gate3.yaml --dataset DATA --output repair
```

## Modules

| Module | Part | Role |
|---|---|---|
| `bounding_diagnostic.py` | 1 | raw / clamp / per-band-bounds comparison + per-band quantiles |
| `worst_case.py` | 2 | 50 worst samples, panels, failure correlations |
| `capability.py` | 4 | reference-input capability report |
| `gt_filter.py` | 5 | 4-state (PASS/REVIEW/REJECT/UNAUDITED) manifest filter |
| `gate_trainer.py` | 7 | micro-metric trainer (AMP, accum, clip, resume, best/latest) |
| `gates.py` | 8, 11 | three gates + acceptance criteria |
| `cli.py` | — | command-line entry |

Model + loss live in the training package for registry/config reuse:
`s2train/models/reference_residual_unet_v2.py`, `s2train/losses` (`repair_composite`).

## Checkpoint selection (Part 7)

Primary monitor is **`val/cloud_land_rmse_micro` (min)** — pixel-weighted over the
whole validation subset, never per-batch macro PSNR. A secondary checkpoint
tracks the best NDVI MAE. Clear-region metrics are never used as model
performance (clear pixels are copied by construction).

## Ground-truth filtering (Part 5)

`gt_filter` distinguishes PASS / REVIEW / REJECT / **UNAUDITED** and never treats
unaudited patches as PASS. Default `conservative` policy keeps audited PASS plus
UNAUDITED patches with recorded native cloud fraction ≤ `native_threshold`; every
gate reports the 4-state composition it used.

## Acceptance criteria (Part 11)

A repaired baseline is accepted only when: negative & over-one fractions are 0;
cloud-land micro PSNR/RMSE and NDVI MAE beat the weighted-reference mean;
catastrophic outliers are reduced; clear pixels are unchanged; and the run is
reproducible from YAML + checkpoint. `gates.acceptance_criteria` evaluates these
from the final micro metrics. Negative/over-one = 0 and clear-pixel preservation
are guaranteed by construction.
