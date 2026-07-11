#!/usr/bin/env python3
"""Convenience entry point for the shared-reference dataset builder.

Examples:
    python run_shared_dataset.py --config dataset_config.yaml
    python run_shared_dataset.py --config dataset_config.yaml --workers 4
    # migrate an existing duplicated dataset:
    python run_shared_dataset.py --output-dir F:/dataset_shared \\
        --migrate-from F:/dataset
"""

from __future__ import annotations

from s2dataset.shared.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
