#!/usr/bin/env python3
"""Convenience entry point for the AI training-dataset builder.

Examples:
    python run_dataset.py --config dataset_config.yaml
    python run_dataset.py --config dataset_config.yaml --workers 6 --no-resume
"""

from __future__ import annotations

from s2dataset.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
