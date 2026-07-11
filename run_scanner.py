#!/usr/bin/env python3
"""Convenience entry point so the scanner can be run without ``-m``.

Examples:
    python run_scanner.py "D:/MSIRI/sentinel2"
    python run_scanner.py "D:/MSIRI/sentinel2" -o reports --expected-crs EPSG:32740
"""

from __future__ import annotations

from s2scanner.cli import run

if __name__ == "__main__":
    raise SystemExit(run())
