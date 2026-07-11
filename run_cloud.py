#!/usr/bin/env python3
"""Convenience entry point for the cloud-mask pipeline.

Examples:
    python run_cloud.py "processed/stacks_10m"
    python run_cloud.py "F:/processed/stacks_10m" --config cloud_config.json
    python run_cloud.py "F:/processed/stacks_10m" -t 50 --workers 2
"""

from __future__ import annotations

from s2cloud.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
