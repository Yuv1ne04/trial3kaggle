"""Corrected, accumulator-based reconstruction metrics.

Why this exists: the training-time evaluator averages *per-batch* PSNR values,
which is statistically wrong (PSNR is non-linear, and batches carry different
numbers of cloud pixels). Here every metric is accumulated as global numerators
and denominators over the whole split and reported two ways:

* ``micro`` - pixel-weighted over the entire split (the scientific dataset
  score);
* ``macro`` - the mean of per-sample scores (comparable to per-image reporting
  in the literature).

Fixes applied versus the training metrics:
    * SAM no longer has a ``1 - 1e-6`` cosine ceiling, so identical spectra give
      exactly ``0`` rad instead of a ~1.4 mrad floor; zero-norm (background)
      pixels are skipped rather than contributing a spurious 90 deg.
    * ERGAS excludes bands whose regional mean reflectance is near zero (which
      otherwise explode the ratio), reports every per-band component, an
      operational-surface-band variant, and warnings - never a silent huge
      number.
    * Clear-region pixel metrics are flagged non-informative when hard
      compositing is on (those pixels are copied from the input verbatim).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import BAND_ORDER
from .indices import land_mask, water_mask, background_mask

#: Bands excluded from the "operational surface" ERGAS (aerosol/vapour/cirrus).
_NON_SURFACE_BANDS = frozenset({"B01", "B09", "B10"})


# --------------------------------------------------------------------------- #
# Generic paired-sample accumulator (bands, indices)                          #
# --------------------------------------------------------------------------- #
@dataclass
class PairAccumulator:
    """Streaming accumulator for a scalar field (MAE/RMSE/bias/Pearson/PSNR).

    Update with matched, already-masked 1-D tensors of predicted and target
    values. Only Python float sums are retained, so memory is O(1).
    """

    n: float = 0.0
    sae: float = 0.0
    sse: float = 0.0
    sum_p: float = 0.0
    sum_t: float = 0.0
    sum_pp: float = 0.0
    sum_tt: float = 0.0
    sum_pt: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate a batch of matched values (1-D tensors)."""
        if pred.numel() == 0:
            return
        pred = pred.double()
        target = target.double()
        diff = pred - target
        self.n += pred.numel()
        self.sae += diff.abs().sum().item()
        self.sse += (diff * diff).sum().item()
        self.sum_p += pred.sum().item()
        self.sum_t += target.sum().item()
        self.sum_pp += (pred * pred).sum().item()
        self.sum_tt += (target * target).sum().item()
        self.sum_pt += (pred * target).sum().item()

    @property
    def mae(self) -> float:
        return self.sae / self.n if self.n else float("nan")

    @property
    def mse(self) -> float:
        return self.sse / self.n if self.n else float("nan")

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mse) if self.n else float("nan")

    @property
    def bias(self) -> float:
        """Mean signed error (prediction - target)."""
        return (self.sum_p - self.sum_t) / self.n if self.n else float("nan")

    def psnr(self, data_range: float = 1.0) -> float:
        mse = self.mse
        if not self.n or math.isnan(mse):
            return float("nan")
        if mse <= 0:
            return 100.0
        return 20 * math.log10(data_range) - 10 * math.log10(mse)

    @property
    def pearson(self) -> float:
        """Pearson correlation between prediction and target."""
        if self.n < 2:
            return float("nan")
        cov = self.sum_pt - self.sum_p * self.sum_t / self.n
        vp = self.sum_pp - self.sum_p ** 2 / self.n
        vt = self.sum_tt - self.sum_t ** 2 / self.n
        denom = math.sqrt(max(vp, 0.0) * max(vt, 0.0))
        return cov / denom if denom > 1e-12 else float("nan")

    def summary(self, data_range: float = 1.0) -> dict[str, float]:
        return {"n": self.n, "mae": self.mae, "rmse": self.rmse, "bias": self.bias,
                "psnr": self.psnr(data_range), "pearson": self.pearson}


# --------------------------------------------------------------------------- #
# Region reconstruction accumulator (PSNR/RMSE/MAE/SAM/SSIM, micro + macro)    #
# --------------------------------------------------------------------------- #
@dataclass
class RegionAccumulator:
    """Accumulates band-averaged pixel metrics + SAM + SSIM for one region.

    Micro numerators are global sums; macro lists hold one value per sample so
    the mean-of-samples can be reported alongside the pixel-weighted score.
    """

    # Micro (band x pixel) numerators.
    n_bandpix: float = 0.0
    sse: float = 0.0
    sae: float = 0.0
    # SAM / SSIM per-pixel numerators.
    sam_sum: float = 0.0
    sam_n: float = 0.0
    ssim_sum: float = 0.0
    ssim_n: float = 0.0
    # Macro per-sample values.
    psnr_samples: list[float] = field(default_factory=list)
    rmse_samples: list[float] = field(default_factory=list)
    mae_samples: list[float] = field(default_factory=list)
    sam_samples: list[float] = field(default_factory=list)
    ssim_samples: list[float] = field(default_factory=list)
    pixels: float = 0.0        # region pixel count (per-pixel, not x band)

    def update(self, pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor,
               ssim_map_full: torch.Tensor, *, data_range: float = 1.0) -> None:
        """Accumulate one batch over a boolean region.

        Args:
            pred: Prediction ``(B, 13, H, W)``.
            target: Target ``(B, 13, H, W)``.
            region: Boolean region ``(B, 1, H, W)``.
            ssim_map_full: Precomputed per-pixel channel-averaged SSIM map
                ``(B, 1, H, W)`` (so it is only computed once per batch).
            data_range: Dynamic range.
        """
        reg = region.expand_as(pred)
        diff = (pred - target)
        # ----- micro -----
        self.n_bandpix += reg.sum().item()
        self.sse += ((diff * diff) * reg).sum().item()
        self.sae += (diff.abs() * reg).sum().item()
        self.pixels += region.sum().item()
        angle, ang_valid = _sam_angle(pred, target, region)
        self.sam_sum += angle.sum().item()
        self.sam_n += ang_valid.sum().item()
        s_sel = region
        self.ssim_sum += (ssim_map_full * s_sel).sum().item()
        self.ssim_n += s_sel.sum().item()
        # ----- macro (per sample) -----
        per_bandpix = reg.sum(dim=(1, 2, 3))
        per_pix = region.sum(dim=(1, 2, 3))
        sse_i = ((diff * diff) * reg).sum(dim=(1, 2, 3))
        sae_i = (diff.abs() * reg).sum(dim=(1, 2, 3))
        ang_i = angle.sum(dim=(1, 2, 3))
        angn_i = ang_valid.sum(dim=(1, 2, 3))
        ssim_i = (ssim_map_full * region).sum(dim=(1, 2, 3))
        for b in range(pred.shape[0]):
            if per_bandpix[b] > 0:
                mse = (sse_i[b] / per_bandpix[b]).item()
                self.rmse_samples.append(math.sqrt(mse))
                self.mae_samples.append((sae_i[b] / per_bandpix[b]).item())
                self.psnr_samples.append(100.0 if mse <= 0 else
                                         -10 * math.log10(mse) + 20 * math.log10(data_range))
            if angn_i[b] > 0:
                self.sam_samples.append((ang_i[b] / angn_i[b]).item())
            if per_pix[b] > 0:
                self.ssim_samples.append((ssim_i[b] / per_pix[b]).item())

    def result(self, data_range: float = 1.0) -> dict[str, float]:
        """Return the micro and macro metric summary for this region."""
        micro_mse = self.sse / self.n_bandpix if self.n_bandpix else float("nan")
        return {
            "n_pixels": self.pixels,
            "psnr_micro": (100.0 if micro_mse == 0 else
                           (-10 * math.log10(micro_mse) + 20 * math.log10(data_range)))
            if self.n_bandpix else float("nan"),
            "rmse_micro": math.sqrt(micro_mse) if self.n_bandpix else float("nan"),
            "mae_micro": self.sae / self.n_bandpix if self.n_bandpix else float("nan"),
            "sam_micro": self.sam_sum / self.sam_n if self.sam_n else float("nan"),
            "ssim_micro": self.ssim_sum / self.ssim_n if self.ssim_n else float("nan"),
            "psnr_macro": _mean(self.psnr_samples),
            "rmse_macro": _mean(self.rmse_samples),
            "mae_macro": _mean(self.mae_samples),
            "sam_macro": _mean(self.sam_samples),
            "ssim_macro": _mean(self.ssim_samples),
        }


def _mean(values: list[float]) -> float:
    """NaN-safe mean of a list."""
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _sam_angle(pred: torch.Tensor, target: torch.Tensor, region: torch.Tensor,
               eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectral angle map (radians) with a correct zero floor.

    Identical spectra yield exactly 0 rad (cosine clamped to ``1.0``, not
    ``1 - 1e-6``). Pixels whose predicted or target spectrum has near-zero norm
    (background) are excluded from the region.

    Args:
        pred: Prediction ``(B, 13, H, W)``.
        target: Target ``(B, 13, H, W)``.
        region: Boolean region ``(B, 1, H, W)``.
        eps: Norm floor below which a pixel is treated as background.

    Returns:
        ``(angle, valid)`` each ``(B, 1, H, W)``; ``angle`` is 0 outside valid.
    """
    # Double precision so identical spectra give a numerically-zero angle
    # (float32 dot/norm rounding otherwise leaves a ~5e-4 rad floor).
    pred = pred.double()
    target = target.double()
    dot = (pred * target).sum(dim=1, keepdim=True)
    pn = pred.norm(dim=1, keepdim=True)
    tn = target.norm(dim=1, keepdim=True)
    denom = pn * tn
    valid = region & (denom > eps)
    cos = torch.where(valid, dot / denom.clamp_min(eps), torch.ones_like(dot))
    cos = cos.clamp(-1.0, 1.0)
    angle = torch.arccos(cos)
    angle = torch.where(valid, angle, torch.zeros_like(angle))
    return angle, valid


# --------------------------------------------------------------------------- #
# SSIM map (channel-averaged, per pixel) - self-contained (no training import) #
# --------------------------------------------------------------------------- #
def ssim_map(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0,
             window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel, channel-averaged SSIM map ``(B, 1, H, W)`` (Gaussian window)."""
    channels = pred.shape[1]
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - (window - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).expand(channels, 1, window, window).contiguous()
    pad = window // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, [pad] * 4, mode="reflect")
        return torch.nn.functional.conv2d(x, kernel, groups=channels)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_p, mu_t = blur(pred), blur(target)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sig_p = blur(pred * pred) - mu_p2
    sig_t = blur(target * target) - mu_t2
    sig_pt = blur(pred * target) - mu_pt
    ssim = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / \
           ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
    return ssim.mean(dim=1, keepdim=True)


def ms_ssim_per_sample(pred: torch.Tensor, target: torch.Tensor, *,
                       data_range: float = 1.0, scales: int = 4) -> list[float]:
    """Whole-image multi-scale SSIM, one value per sample (macro reporting).

    MS-SSIM is not well defined over an arbitrary masked region, so it is
    reported whole-image only; region structure is captured by region SSIM.

    Args:
        pred: Prediction ``(B, 13, H, W)``.
        target: Target ``(B, 13, H, W)``.
        data_range: Dynamic range.
        scales: Number of octaves (downsample by 2 each).

    Returns:
        A list of per-sample MS-SSIM values.
    """
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                           device=pred.device)[:scales]
    weights = weights / weights.sum()
    p, t = pred, target
    per_scale = []
    for s in range(scales):
        m = ssim_map(p, t, data_range=data_range).mean(dim=(1, 2, 3)).clamp_min(1e-6)
        per_scale.append(m)
        if s < scales - 1:
            p = torch.nn.functional.avg_pool2d(p, 2)
            t = torch.nn.functional.avg_pool2d(t, 2)
    stacked = torch.stack(per_scale, dim=0)          # (scales, B)
    ms = (stacked ** weights[:, None]).prod(dim=0)    # (B,)
    return ms.tolist()


# --------------------------------------------------------------------------- #
# ERGAS (fixed / scoped)                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class ErgasAccumulator:
    """Accumulates per-band RMSE and mean reflectance for a scoped ERGAS."""

    sse: list[float] = field(default_factory=lambda: [0.0] * 13)
    sum_t: list[float] = field(default_factory=lambda: [0.0] * 13)
    n: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               region: torch.Tensor) -> None:
        """Accumulate per-band squared error and target sum over a region."""
        reg = region  # (B,1,H,W)
        count = reg.sum().item()
        if count == 0:
            return
        self.n += count
        for b in range(target.shape[1]):
            d = (pred[:, b:b + 1] - target[:, b:b + 1])
            self.sse[b] += ((d * d) * reg).sum().item()
            self.sum_t[b] += (target[:, b:b + 1] * reg).sum().item()

    def result(self, *, min_mean: float = 0.01, ratio: float = 1.0) -> dict:
        """Compute ERGAS with per-band components and near-zero-mean warnings.

        Returns:
            A dict with ``ergas_all`` (bands with a valid mean),
            ``ergas_operational`` (surface bands only), per-band components and
            a list of bands excluded for near-zero mean.
        """
        components: dict[str, float] = {}
        excluded: list[str] = []
        valid_all: list[float] = []
        valid_ops: list[float] = []
        for b, name in enumerate(BAND_ORDER):
            if self.n <= 0:
                continue
            rmse_b = math.sqrt(self.sse[b] / self.n)
            mu_b = self.sum_t[b] / self.n
            if mu_b <= min_mean:
                excluded.append(name)
                components[name] = float("nan")
                continue
            comp = (rmse_b / mu_b) ** 2
            components[name] = comp
            valid_all.append(comp)
            if name not in _NON_SURFACE_BANDS:
                valid_ops.append(comp)

        def _ergas(vals: list[float]) -> float:
            return 100.0 * ratio * math.sqrt(sum(vals) / len(vals)) if vals else float("nan")

        return {
            "ergas_all": _ergas(valid_all),
            "ergas_operational": _ergas(valid_ops),
            "per_band_components": components,
            "excluded_near_zero_mean": excluded,
            "note": ("ERGAS excludes bands with regional mean reflectance <= "
                     f"{min_mean}; 'operational' also drops {sorted(_NON_SURFACE_BANDS)}."),
        }


# --------------------------------------------------------------------------- #
# Region construction                                                         #
# --------------------------------------------------------------------------- #
def build_regions(target: torch.Tensor, cloud_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the standard boolean regions from the clean target + cloud mask.

    Args:
        target: Clean ground-truth reflectance ``(B, 13, H, W)``.
        cloud_mask: Cloud mask ``(B, 1, H, W)`` (1 = cloud).

    Returns:
        A dict of boolean regions ``(B, 1, H, W)``.
    """
    cloud = cloud_mask > 0.5
    clear = ~cloud
    bg = background_mask(target)
    land = land_mask(target)
    ocean = water_mask(target)
    whole = torch.ones_like(cloud)
    return {
        "whole": whole & (~bg),
        "cloud": cloud & (~bg),
        "clear": clear & (~bg),
        "land": land,
        "ocean": ocean,
        "cloud_land": cloud & land,
        "cloud_ocean": cloud & ocean,
        "clear_land": clear & land,
    }
