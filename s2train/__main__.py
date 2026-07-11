"""Enable ``python -m s2train``."""

from __future__ import annotations

from .cli import run

if __name__ == "__main__":
    raise SystemExit(run())
