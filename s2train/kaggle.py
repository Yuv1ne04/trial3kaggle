"""Kaggle-environment awareness: paths, defaults and cross-session resume.

Kaggle mounts datasets read-only under ``/kaggle/input`` and gives a single
writable, output-persisted directory ``/kaggle/working``. GPU sessions are time
limited and can be interrupted, so checkpoints must live in ``/kaggle/working``
and resume must be able to find a checkpoint saved by a *previous* session
(typically re-attached under ``/kaggle/input``).
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def on_kaggle() -> bool:
    """Return whether the code is running inside a Kaggle kernel.

    Returns:
        ``True`` on Kaggle (detected via the environment or ``/kaggle`` paths).
    """
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or KAGGLE_WORKING.exists()


def default_output_root(configured: str) -> str:
    """Return the output root, redirecting the default onto ``/kaggle/working``.

    The framework default (``"experiments"``) is redirected to
    ``/kaggle/working``. As a safety net, any output root that points at the
    **read-only** ``/kaggle/input`` mount (e.g. from a mis-edited config) is also
    redirected there, so a run can never fail trying to write to the dataset.

    Args:
        configured: The ``output_root`` from the config.

    Returns:
        The effective output root.
    """
    if on_kaggle():
        try:
            under_input = Path(configured).resolve().is_relative_to(KAGGLE_INPUT)
        except (ValueError, OSError):
            under_input = str(configured).startswith(str(KAGGLE_INPUT))
        if configured == "experiments" or under_input:
            return str(KAGGLE_WORKING / "experiments")
    return configured


def default_resume_search_dirs(configured: list[str]) -> list[str]:
    """Return checkpoint search directories, adding ``/kaggle/input`` on Kaggle.

    Args:
        configured: Search dirs from the config.

    Returns:
        The effective search-dir list.
    """
    dirs = list(configured)
    if on_kaggle() and str(KAGGLE_INPUT) not in dirs and KAGGLE_INPUT.exists():
        dirs.append(str(KAGGLE_INPUT))
    return dirs


def find_resume_checkpoint(run_dir: Path, experiment_name: str,
                           search_dirs: list[str]) -> Path | None:
    """Locate a checkpoint to resume from.

    Preference order: the run's own ``checkpoints/latest.pt`` (same session or a
    persisted ``/kaggle/working``); otherwise the newest ``latest.pt`` (then
    ``best.pt``) found under the search dirs — preferring a path that mentions
    the experiment name (e.g. a previous session re-attached under
    ``/kaggle/input``).

    Args:
        run_dir: This run's output directory.
        experiment_name: The experiment name (for preferring matching paths).
        search_dirs: Extra directories to search recursively.

    Returns:
        The chosen checkpoint path, or ``None`` if none is found.
    """
    local = run_dir / "checkpoints" / "latest.pt"
    if local.exists():
        return local

    candidates: list[Path] = []
    for base in search_dirs:
        root = Path(base)
        if not root.exists():
            continue
        for pattern in ("**/checkpoints/latest.pt", "**/latest.pt", "**/best.pt"):
            candidates.extend(root.glob(pattern))

    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, float]:
        matches_name = int(experiment_name.lower() in str(path).lower())
        is_latest = int(path.name == "latest.pt")
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (matches_name, is_latest, mtime)

    return max(candidates, key=score)
