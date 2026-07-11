#!/usr/bin/env python3
"""Convenience entry point for the reference-selection engine.

Examples:
    python run_refselect.py
    python run_refselect.py --config reference_config.yaml
    python run_refselect.py -n 6 --direction both
"""

from __future__ import annotations

from s2refselect.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
