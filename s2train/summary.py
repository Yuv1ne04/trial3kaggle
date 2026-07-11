"""Model summary: parameters, VRAM, and measured throughput / epoch-time.

Throughput and peak VRAM are *measured* on the current device with a dummy batch
of the configured shape. On Kaggle this means the numbers reported are the real
figures for whatever accelerator the kernel is on (T4 / P100 / L4). On CPU a
parameter-based VRAM estimate is used instead.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from .config import ExperimentConfig
from .reproducibility import count_parameters


def _dummy_batch(config: ExperimentConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a random batch matching the configured shapes.

    Args:
        config: The experiment configuration.
        device: Target device.

    Returns:
        A batch dict on ``device``.
    """
    b = config.data.batch_size
    s = config.data.patch_size
    r = config.data.max_references
    return {
        "cloudy": torch.rand(b, 13, s, s, device=device),
        "mask": (torch.rand(b, 1, s, s, device=device) > 0.6).float(),
        "references": torch.rand(b, r, 13, s, s, device=device),
        "reference_validity_mask": torch.ones(b, r, device=device),
        "ground_truth": torch.rand(b, 13, s, s, device=device),
    }


def _estimate_vram_mb(params: int, config: ExperimentConfig) -> float:
    """Estimate training VRAM from parameter and activation sizes (CPU fallback).

    Args:
        params: Total parameter count.
        config: The experiment configuration.

    Returns:
        A rough VRAM estimate in MiB (weights + grads + Adam states + activations).
    """
    bytes_per = 4
    optimizer_states = 4  # weights + grad + Adam m + v
    weights = params * bytes_per * optimizer_states
    # Crude activation estimate: batch * bands * H * W * depth_factor.
    b, s = config.data.batch_size, config.data.patch_size
    activations = b * 64 * s * s * 8 * bytes_per
    return (weights + activations) / 1024 ** 2


def model_summary(model: nn.Module, config: ExperimentConfig, device: torch.device,
                  *, num_train_samples: int | None = None, warmup: int = 2,
                  iters: int = 5) -> dict[str, Any]:
    """Produce a model + performance summary.

    Args:
        model: The model to profile.
        config: The experiment configuration.
        device: The device to profile on.
        num_train_samples: Training-set size, for an epoch-time estimate.
        warmup: Warmup iterations before timing.
        iters: Timed forward+backward iterations.

    Returns:
        A JSON-serialisable summary dict.
    """
    params = count_parameters(model)
    summary: dict[str, Any] = {
        "model": type(model).__name__,
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "parameters_millions": round(params["total"] / 1e6, 3),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else "cpu"),
        "batch_size": config.data.batch_size,
        "patch_size": config.data.patch_size,
    }

    model = model.to(device)
    batch = _dummy_batch(config, device)
    loss_fn = torch.nn.functional.l1_loss
    try:
        model.train()
        for _ in range(warmup):
            out = model(batch)
            loss_fn(out, batch["ground_truth"]).backward()
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iters):
            out = model(batch)
            loss_fn(out, batch["ground_truth"]).backward()
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_step = elapsed / iters
        throughput = config.data.batch_size / per_step
        summary["measured_step_seconds"] = round(per_step, 4)
        summary["throughput_samples_per_sec"] = round(throughput, 2)
        if device.type == "cuda":
            summary["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
        else:
            summary["estimated_vram_mb"] = round(
                _estimate_vram_mb(params["total"], config), 1)
        if num_train_samples:
            summary["estimated_epoch_seconds"] = round(num_train_samples / throughput, 1)
            summary["estimated_epoch_minutes"] = round(
                num_train_samples / throughput / 60, 2)
            summary["num_train_samples"] = num_train_samples
    except Exception as exc:  # noqa: BLE001 - profiling must never break a run
        summary["profiling_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        model.zero_grad(set_to_none=True)
    return summary
