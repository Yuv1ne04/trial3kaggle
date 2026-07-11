"""Leakage-free train/validation/test splitting by acquisition date.

Splitting is performed on whole acquisition *dates* in chronological order, so
every patch from a given date lands in exactly one split. This prevents
temporal leakage: the model is validated/tested on dates it never saw in
training, mirroring the operational task of reconstructing *future* imagery.
"""

from __future__ import annotations

from .config import DatasetConfig
from .logging_setup import get_logger
from .models import SampleSpec

logger = get_logger()


def assign_splits(specs: list[SampleSpec], config: DatasetConfig) -> list[SampleSpec]:
    """Assign a split to each sample spec, chronologically by date, in place.

    The earliest dates form the training set, the middle the validation set and
    the most recent the test set, matching the deployment scenario of training
    on the past and reconstructing the present/future.

    Args:
        specs: The sample specs to split (any order).
        config: Active dataset configuration (split fractions).

    Returns:
        The same specs, sorted by date, with ``split`` populated.
    """
    specs.sort(key=lambda s: s.target_date)
    unique_dates = sorted({s.target_date for s in specs})
    n = len(unique_dates)

    n_train = int(round(n * config.split.train))
    n_val = int(round(n * config.split.val))
    # Guarantee every non-empty split gets at least one date when feasible.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])

    for spec in specs:
        if spec.target_date in train_dates:
            spec.split = "train"
        elif spec.target_date in val_dates:
            spec.split = "val"
        else:
            spec.split = "test"

    logger.info(
        "Temporal split by date: %d train / %d val / %d test (of %d dates)",
        n_train, n_val, n_test, n,
    )
    if train_dates and val_dates:
        logger.info(
            "Split boundaries: train<=%s | val<=%s | test>=%s",
            max(train_dates),
            max(val_dates) if val_dates else "-",
            min(d for d in unique_dates if d not in train_dates and d not in val_dates)
            if n_test else "-",
        )
    return specs
