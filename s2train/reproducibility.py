"""Reproducibility utilities: seeding and environment/run provenance capture."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The master seed.
        deterministic: When ``True`` request deterministic cuDNN/algorithms
            (slower, but bit-reproducible where supported).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - not all ops support it
            pass


def _git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_info() -> dict[str, Any]:
    """Return a description of the available compute device(s).

    Returns:
        A dict describing CUDA availability and device names/memory.
    """
    if not torch.cuda.is_available():
        return {"cuda": False, "device": "cpu",
                "cpu_count": os.cpu_count()}
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
            "capability": f"{props.major}.{props.minor}",
        })
    return {"cuda": True, "device_count": len(devices), "devices": devices,
            "cuda_version": torch.version.cuda}


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for a model.

    Args:
        model: The model to inspect.

    Returns:
        A dict with ``total`` and ``trainable`` parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def provenance(config: dict[str, Any], dataset_version: str | None) -> dict[str, Any]:
    """Assemble a reproducibility/provenance record for an experiment.

    Args:
        config: The experiment configuration as a dict.
        dataset_version: An identifier for the dataset (e.g. a stats hash/path).

    Returns:
        A JSON-serialisable provenance dict (git, timestamp, seed, GPU, env).
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "dataset_version": dataset_version,
        "seed": config.get("seed"),
        "gpu": gpu_info(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "config": config,
    }
