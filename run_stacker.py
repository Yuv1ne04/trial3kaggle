#!/usr/bin/env python3
"""Convenience entry point for the stacking pipeline.

Examples:
    python run_stacker.py "F:/Sen 2"
    python run_stacker.py "F:/Sen 2" -o processed/stacks_10m --workers 3
"""

from __future__ import annotations

from s2stacker.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
