"""Model registry.

Importing this package registers all built-in models. To add a new architecture
(e.g. TCR-Net), create a module here that subclasses
:class:`~s2train.models.base.BaseReconstructionModel` and decorates the class
with ``@MODELS.register("name")`` — then select it via ``model.name`` in YAML.
No other framework code changes.
"""

from __future__ import annotations

from .base import BaseReconstructionModel
from .reference_unet import ReferenceUNetBaseline
from .unet import ReferenceUNet, UNet

__all__ = ["BaseReconstructionModel", "UNet", "ReferenceUNet", "ReferenceUNetBaseline"]
