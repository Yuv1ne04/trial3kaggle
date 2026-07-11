#!/usr/bin/env python3
"""Convenience entry point for evaluation.

Example:
    python run_eval.py --checkpoint experiments/unet_baseline_.../checkpoints/best.pt \\
        --split test --out evaluation_report.json
"""

from __future__ import annotations

import sys

from s2train.cli import run

if __name__ == "__main__":
    raise SystemExit(run(["evaluate", *sys.argv[1:]]))
