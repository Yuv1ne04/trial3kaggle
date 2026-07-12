"""Baseline repair: diagnose, bound, and re-validate the reconstruction model.

This package repairs the audited v1 baseline without retraining it: it quantifies
how much of the failure is unbounded output (Part 1), analyses worst-case samples
(Part 2), reports reference-input capability (Part 4), provides a 4-state
ground-truth filter (Part 5), and runs the three training gates for the new
physically-bounded ``ReferenceResidualUNetV2`` (Parts 7-8). It never modifies the
original ``best.pt`` or the dataset.
"""

from __future__ import annotations

__all__: list[str] = []
