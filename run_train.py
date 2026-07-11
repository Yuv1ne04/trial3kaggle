#!/usr/bin/env python3
"""Convenience entry point for training.

Examples:
    python run_train.py --config configs/experiment_001.yaml
    python run_train.py --config configs/experiment_002.yaml --resume auto
"""

from __future__ import annotations

import sys

from s2train.cli import run

if __name__ == "__main__":
    raise SystemExit(run(["train", *sys.argv[1:]]))
