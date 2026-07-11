#!/usr/bin/env python3
"""Convenience entry point for the synthetic supervision pipeline.

Examples:
    python run_synthetic.py --config synthetic_config.yaml
    python run_synthetic.py --config synthetic_config.yaml --variants 4 --workers 4
"""

from __future__ import annotations

from s2dataset.synthetic.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
