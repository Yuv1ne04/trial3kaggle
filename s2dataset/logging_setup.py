"""Logging configuration for the dataset builder."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "s2dataset"


def configure_logging(log_file: Path, *, verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger (console + file).

    Re-invoking replaces existing handlers to avoid duplicate log lines.

    Args:
        log_file: Destination path for the log file (parents created).
        verbose: When ``True`` the console also shows DEBUG messages.

    Returns:
        The configured :class:`logging.Logger`.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(processName)s | %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return logging.getLogger(LOGGER_NAME)
