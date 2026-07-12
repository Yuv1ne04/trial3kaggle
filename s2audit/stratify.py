"""Stratification keys for the evaluation (Part 7).

Assigns each sample to buckets along four axes so the evaluator can maintain a
:class:`~s2audit.metrics.RegionAccumulator` per stratum. Surface category is
computed from the ground truth at evaluation time (see
:func:`s2audit.indices.surface_category`); the other three come from manifest
metadata. The design is intentionally table-driven so an MSIRI sugar-cane field
polygon layer can be added later as a fifth axis without touching the evaluator.
"""

from __future__ import annotations

#: Coverage bin edges as percentages (lo inclusive, hi exclusive) + label.
_COVERAGE_BINS = ((10, 20, "cov_10_20"), (20, 40, "cov_20_40"), (40, 70.0001, "cov_40_70"))


def coverage_bin(coverage: float | None) -> str:
    """Map an applied cloud coverage (fraction or percent) to a bin label."""
    if coverage is None:
        return "cov_other"
    pct = coverage * 100.0 if coverage <= 1.0 else coverage
    for lo, hi, label in _COVERAGE_BINS:
        if lo <= pct < hi:
            return label
    return "cov_other"


def difficulty_bin(difficulty: str | None) -> str:
    """Return the curriculum difficulty band (easy/medium/hard) or 'unknown'."""
    return difficulty if difficulty in ("easy", "medium", "hard") else "unknown"


def reference_count_bin(n_references: int | None) -> str:
    """Bucket the reference count (2/3/4 refs) or 'other'."""
    return f"{n_references}_refs" if n_references in (2, 3, 4) else "refs_other"


#: The stratification axes; each maps a label name -> how it is derived.
AXES = ("difficulty", "coverage", "reference_count", "surface")


def sample_strata(*, difficulty: str | None, coverage: float | None,
                  n_references: int | None, surface: str) -> dict[str, str]:
    """Return the stratum label along every axis for one sample.

    Args:
        difficulty: Manifest difficulty band.
        coverage: Applied cloud coverage.
        n_references: Number of valid references.
        surface: Surface category from the ground truth.

    Returns:
        A dict ``axis -> "axis=label"`` used as accumulator keys.
    """
    return {
        "difficulty": f"difficulty={difficulty_bin(difficulty)}",
        "coverage": f"coverage={coverage_bin(coverage)}",
        "reference_count": f"reference_count={reference_count_bin(n_references)}",
        "surface": f"surface={surface}",
    }
