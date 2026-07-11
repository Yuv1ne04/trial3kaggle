#!/usr/bin/env python3
"""Convenience entry point for full-tile inference.

Example:
    python run_infer.py --checkpoint .../best.pt \\
        --stack cloudy.tif --mask mask.tif --references r1.tif r2.tif --out recon.tif
"""

from __future__ import annotations

import sys

from s2train.cli import run

if __name__ == "__main__":
    raise SystemExit(run(["infer", *sys.argv[1:]]))
