#!/usr/bin/env python3
"""Convenience entry point for the temporal database builder.

Examples:
    python run_temporal.py "processed/stacks_10m"
    python run_temporal.py "F:/processed/stacks_10m" -n 15
"""

from __future__ import annotations

from s2temporal.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
