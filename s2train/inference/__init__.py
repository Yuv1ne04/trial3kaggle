"""Inference workflow: load a trained model and reconstruct patches or tiles.

Full-tile reconstruction uses overlapping windows with cosine blending and
composites the observed clear pixels back in, so real observations are never
altered and tile seams do not appear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..builder import build_model, resolve_device
from ..config import ExperimentConfig, ComponentSpec, DataConfig, TrainerConfig
from ..config import CheckpointConfig, EarlyStoppingConfig, VisualizationConfig


def _config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Reconstruct an :class:`ExperimentConfig` from a stored config dict.

    Args:
        data: The ``config`` dict saved in a checkpoint.

    Returns:
        A populated :class:`ExperimentConfig`.
    """
    def spec(d: Any, default: str) -> ComponentSpec:
        return ComponentSpec(d["name"], d.get("params", {})) if isinstance(d, dict) \
            else ComponentSpec(default)

    def sub(cls, key):
        value = data.get(key, {})
        valid = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in valid})

    return ExperimentConfig(
        name=data.get("name", "experiment"), seed=data.get("seed", 0),
        model=spec(data.get("model", {}), "unet"),
        data=sub(DataConfig, "data"), optimizer=spec(data.get("optimizer", {}), "adamw"),
        scheduler=spec(data.get("scheduler", {}), "cosine"),
        loss=spec(data.get("loss", {}), "composite"),
        metrics=data.get("metrics", []), trainer=sub(TrainerConfig, "trainer"),
        checkpoint=sub(CheckpointConfig, "checkpoint"),
        early_stopping=sub(EarlyStoppingConfig, "early_stopping"),
        visualization=sub(VisualizationConfig, "visualization"))


class Predictor:
    """Loads a trained checkpoint and reconstructs images."""

    def __init__(self, model: torch.nn.Module, config: ExperimentConfig,
                 device: torch.device) -> None:
        """Initialise the predictor.

        Args:
            model: The loaded model (in eval mode).
            config: The experiment configuration.
            device: The inference device.
        """
        self.model = model.eval()
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint: Path | str, *, device: str = "auto") -> "Predictor":
        """Build a predictor from a saved checkpoint.

        Args:
            checkpoint: Path to a ``.pt`` checkpoint.
            device: Device string.

        Returns:
            A ready :class:`Predictor`.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
        """
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        dev = resolve_device(device)
        state = torch.load(checkpoint, map_location=dev)
        config = _config_from_dict(state["config"])
        model = build_model(config).to(dev)
        model.load_state_dict(state["model"])
        return cls(model, config, dev)

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct a batch.

        Args:
            batch: A standard batch (tensors).

        Returns:
            The prediction ``(B, 13, H, W)``.
        """
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        return self.model(batch).cpu()

    @torch.no_grad()
    def reconstruct_tile(self, cloudy: np.ndarray, mask: np.ndarray,
                         references: np.ndarray, *, patch: int = 256,
                         overlap: int = 64, reflectance_scale: float = 10000.0) -> np.ndarray:
        """Reconstruct a full tile with overlapping windows and cosine blending.

        Args:
            cloudy: Cloudy tile ``(13, H, W)`` in DN.
            mask: Cloud mask ``(1, H, W)`` (1 = cloud).
            references: References ``(R, 13, H, W)`` in DN.
            patch: Window size.
            overlap: Window overlap in pixels.
            reflectance_scale: DN -> reflectance divisor.

        Returns:
            The reconstructed tile ``(13, H, W)`` in DN (clear pixels preserved).
        """
        _, h, w = cloudy.shape
        r = references.shape[0]
        acc = np.zeros((13, h, w), dtype=np.float32)
        weight = np.zeros((1, h, w), dtype=np.float32)
        window = _cosine_window(patch)
        step = patch - overlap
        scale = reflectance_scale

        for row in _starts(h, patch, step):
            for col in _starts(w, patch, step):
                sl = (slice(row, row + patch), slice(col, col + patch))
                c = cloudy[:, sl[0], sl[1]].astype(np.float32) / scale
                m = mask[:, sl[0], sl[1]].astype(np.float32)
                refs = references[:, :, sl[0], sl[1]].astype(np.float32) / scale
                validity = np.array([1.0 if refs[i].any() else 0.0 for i in range(r)],
                                    dtype=np.float32)
                batch = {
                    "cloudy": torch.from_numpy(c)[None],
                    "mask": torch.from_numpy(m)[None],
                    "references": torch.from_numpy(refs)[None],
                    "reference_validity_mask": torch.from_numpy(validity)[None],
                }
                pred = self.predict_batch(batch)[0].numpy() * scale
                acc[:, sl[0], sl[1]] += pred * window
                weight[:, sl[0], sl[1]] += window
        weight = np.clip(weight, 1e-6, None)
        return (acc / weight).astype(np.float32)


def _starts(extent: int, patch: int, step: int) -> list[int]:
    """Return window start positions covering ``extent`` (last shifted inward)."""
    if patch >= extent:
        return [0]
    starts = list(range(0, extent - patch + 1, step))
    if starts[-1] != extent - patch:
        starts.append(extent - patch)
    return starts


def _cosine_window(size: int) -> np.ndarray:
    """Return a 2-D separable cosine (Hann) blending window ``(1, size, size)``."""
    w = np.hanning(size)
    w = np.clip(w, 1e-3, None)
    window = np.outer(w, w).astype(np.float32)
    return window[None, :, :]
