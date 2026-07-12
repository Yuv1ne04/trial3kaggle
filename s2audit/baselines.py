"""Non-learned reference baselines for the U-Net comparison (Part 6).

Every baseline produces a ``(B, 13, H, W)`` reconstruction and is composited
with the *same* cloud mask as the model (observed clear pixels preserved), so
the cloud-region comparison is apples-to-apples. The U-Net must beat these to
justify a learned model at all.

Baselines:
    * ``cloudy_input``            - no correction (floor).
    * ``weighted_reference_mean`` - validity-weighted mean of the references.
    * ``nearest_temporal_ref``    - the reference closest in time to the target.
    * ``best_single_ref_oracle``  - per-sample the reference with the lowest
      cloud-region error vs GT (an *oracle* upper bound on single-reference
      copying; labelled as such, never presented as achievable operationally).
    * ``score_weighted_mean``     - requires stored per-reference selection
      scores; skipped with a note when they are absent (they are not in the
      current manifests).
"""

from __future__ import annotations

from datetime import datetime

import torch


def _composite(pred: torch.Tensor, cloudy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep the observed clear pixels; use ``pred`` only under cloud."""
    cloud = (mask > 0.5).float()
    return cloud * pred + (1.0 - cloud) * cloudy


def _weighted_mean(references: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    """Validity-weighted mean over reference slots ``(B, R, 13, H, W)``."""
    w = validity.view(validity.shape[0], validity.shape[1], 1, 1, 1)
    denom = w.sum(dim=1).clamp_min(1e-6)
    return (references * w).sum(dim=1) / denom


def cloudy_input(batch: dict) -> torch.Tensor:
    """The synthetic cloudy input, unchanged (do-nothing floor)."""
    return batch["cloudy"].clone()


def weighted_reference_mean(batch: dict) -> torch.Tensor:
    """Validity-weighted mean of the references, composited."""
    pred = _weighted_mean(batch["references"], batch["reference_validity_mask"])
    return _composite(pred, batch["cloudy"], batch["mask"])


def _target_dates(batch: dict) -> list[str | None]:
    meta = batch.get("metadata") or [{}] * batch["cloudy"].shape[0]
    return [m.get("target_date") for m in meta]


def _reference_dates(batch: dict, b: int) -> list | None:
    meta = batch.get("metadata")
    if not meta or b >= len(meta):
        return None
    return meta[b].get("reference_dates")


def _parse(date: str | None):
    if not date:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date, fmt)
        except (ValueError, TypeError):
            continue
    return None


def nearest_temporal_ref(batch: dict) -> torch.Tensor:
    """The valid reference closest in acquisition time to the target date.

    Falls back to the first valid slot when dates are unavailable (references
    are stored most-recent-first, so slot 0 is the nearest by construction).
    """
    refs = batch["references"]
    validity = batch["reference_validity_mask"]
    b, r = refs.shape[0], refs.shape[1]
    chosen = torch.zeros((b, refs.shape[2], refs.shape[3], refs.shape[4]), dtype=refs.dtype)
    target_dates = _target_dates(batch)
    for i in range(b):
        rdates = _reference_dates(batch, i)
        tdate = _parse(target_dates[i])
        pick = 0
        if rdates and tdate is not None:
            best, best_gap = 0, None
            for j in range(min(r, len(rdates))):
                if validity[i, j] <= 0:
                    continue
                rd = _parse(rdates[j])
                if rd is None:
                    continue
                gap = abs((tdate - rd).days)
                if best_gap is None or gap < best_gap:
                    best_gap, best = gap, j
            pick = best
        else:
            valid_idx = (validity[i] > 0).nonzero(as_tuple=True)[0]
            pick = int(valid_idx[0]) if len(valid_idx) else 0
        chosen[i] = refs[i, pick]
    return _composite(chosen, batch["cloudy"], batch["mask"])


def best_single_ref_oracle(batch: dict) -> torch.Tensor:
    """Oracle: per-sample reference with the lowest cloud-region MAE vs GT.

    Uses the ground truth to pick, so it is an *upper bound* on single-reference
    copying - a diagnostic ceiling, not an operational method.
    """
    refs = batch["references"]
    validity = batch["reference_validity_mask"]
    gt = batch["ground_truth"]
    cloud = (batch["mask"] > 0.5).float()
    b, r = refs.shape[0], refs.shape[1]
    chosen = torch.zeros_like(gt)
    denom = cloud.sum(dim=(1, 2, 3)).clamp_min(1.0)
    for i in range(b):
        best, best_err = 0, None
        for j in range(r):
            if validity[i, j] <= 0:
                continue
            err = ((refs[i, j] - gt[i]).abs() * cloud[i]).sum() / denom[i]
            if best_err is None or err < best_err:
                best_err, best = err, j
        chosen[i] = refs[i, best]
    return _composite(chosen, batch["cloudy"], batch["mask"])


#: Operational (non-oracle) baselines, always available.
OPERATIONAL_BASELINES = {
    "cloudy_input": cloudy_input,
    "weighted_reference_mean": weighted_reference_mean,
    "nearest_temporal_ref": nearest_temporal_ref,
}

#: Oracle diagnostic baselines (use GT; ceilings only).
ORACLE_BASELINES = {
    "best_single_ref_oracle": best_single_ref_oracle,
}


def available_baselines(batch: dict) -> tuple[dict, list[str]]:
    """Return the callable baselines and a list of skipped ones with reasons.

    Args:
        batch: A sample batch (used to detect optional metadata like scores).

    Returns:
        ``(baselines, skipped_notes)``.
    """
    baselines = {**OPERATIONAL_BASELINES, **ORACLE_BASELINES}
    skipped = []
    meta = batch.get("metadata") or [{}]
    if not any("reference_scores" in (m or {}) for m in meta):
        skipped.append("score_weighted_mean: per-reference selection scores are "
                       "not stored in the sample manifests.")
    return baselines, skipped
