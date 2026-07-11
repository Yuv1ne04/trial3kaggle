"""Loading inputs and assembling per-target sample specifications."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import DatasetConfig
from .logging_setup import get_logger
from .models import SampleSpec

logger = get_logger()

#: Extracts the 8-digit acquisition date from any stack/mask filename.
_DATE_IN_NAME = re.compile(r"(?P<date>\d{8})")


def _compact(iso_or_compact: str) -> str:
    """Return a date string as compact ``YYYYMMDD``.

    Args:
        iso_or_compact: A date as ``YYYY-MM-DD`` or ``YYYYMMDD``.

    Returns:
        The date as ``YYYYMMDD``.
    """
    return iso_or_compact.replace("-", "").strip()


def _index_by_date(
    directory: Path, suffix: str, *, tile: str | None = None
) -> dict[str, Path]:
    """Index files in a directory by the 8-digit date in their name.

    Args:
        directory: Directory to scan.
        suffix: Required filename suffix (e.g. ``"_stack.tif"``).
        tile: Optional MGRS tile id (e.g. ``"T40KEC"``); when set, files whose
            names do not contain it are skipped. This prevents acquisitions from
            neighbouring tiles (different grid origin) contaminating a same-tile
            dataset.

    Returns:
        Mapping of ``YYYYMMDD`` -> file path. Later matches overwrite earlier.
    """
    index: dict[str, Path] = {}
    if not directory.exists():
        return index
    for path in sorted(directory.glob(f"*{suffix}")):
        if tile is not None and tile.upper() not in path.name.upper():
            continue
        match = _DATE_IN_NAME.search(path.name)
        if match:
            index[match.group("date")] = path
    return index


class DatasetLoader:
    """Loads metadata and resolves it into processable sample specifications."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialise the loader.

        Args:
            config: Active dataset configuration.
        """
        self.config = config

    def _load_reference_database(self) -> dict[str, list[str]]:
        """Load target -> ranked reference dates from the reference database.

        Returns:
            Mapping of compact target date -> list of compact reference dates.

        Raises:
            FileNotFoundError: If the reference database is missing.
            ValueError: If the database JSON has an unexpected structure.
        """
        path = self.config.reference_database
        if not path.exists():
            raise FileNotFoundError(f"Reference database not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        targets = data.get("targets") if isinstance(data, dict) else data
        if not isinstance(targets, list):
            raise ValueError(f"Unexpected reference database structure: {path}")

        mapping: dict[str, list[str]] = {}
        for entry in targets:
            target = _compact(str(entry["target_date"]))
            refs = [_compact(str(r["date"])) for r in entry.get("references", [])]
            mapping[target] = refs
        logger.info("Loaded reference database for %d target(s)", len(mapping))
        return mapping

    def _load_temporal(self) -> dict[str, dict[str, object]]:
        """Load calendar attributes keyed by compact date.

        Returns:
            Mapping of ``YYYYMMDD`` -> ``{season, year, month, day_of_year}``.
            Empty if the temporal database is absent.
        """
        path = self.config.temporal_database
        if not path.exists():
            logger.warning("Temporal database not found: %s", path)
            return {}
        frame = pd.read_csv(path)
        result: dict[str, dict[str, object]] = {}
        for _, row in frame.iterrows():
            date = _compact(str(row["date"]))
            result[date] = {
                "season": str(row.get("season", "")),
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "month": int(row["month"]) if pd.notna(row.get("month")) else None,
                "day_of_year": int(row["day_of_year"])
                if pd.notna(row.get("day_of_year"))
                else None,
            }
        return result

    def _load_cloud(self) -> dict[str, float]:
        """Load cloud percentages keyed by compact date.

        Returns:
            Mapping of ``YYYYMMDD`` -> cloud percentage. Empty if absent.
        """
        path = self.config.cloud_statistics
        if not path.exists():
            logger.warning("Cloud statistics not found: %s", path)
            return {}
        frame = pd.read_csv(path)
        result: dict[str, float] = {}
        for _, row in frame.iterrows():
            date = _compact(str(row["date"]))
            value = row.get("cloud_percentage")
            if pd.notna(value):
                result[date] = float(value)
        return result

    def build_specs(self) -> list[SampleSpec]:
        """Assemble validated sample specs for every resolvable target.

        Targets missing a stack or mask are skipped. References without a stack
        on disk are dropped; references are padded cyclically to
        ``n_references`` so output tensors keep a fixed shape.

        Returns:
            A date-sorted list of :class:`SampleSpec`.

        Raises:
            FileNotFoundError: If the reference database is missing.
            ValueError: If no processable targets remain.
        """
        references = self._load_reference_database()
        temporal = self._load_temporal()
        cloud = self._load_cloud()
        # The tile filter applies to stacks (which carry the MGRS tile in their
        # filename); masks are named by date only, so no tile filter there.
        stack_index = _index_by_date(
            self.config.stacks_dir, "_stack.tif", tile=self.config.tile
        )
        mask_index = _index_by_date(self.config.masks_dir, "_cloudmask.tif")
        logger.info(
            "Indexed %d stack(s) and %d mask(s)", len(stack_index), len(mask_index)
        )

        specs: list[SampleSpec] = []
        for target_date in sorted(references):
            spec = self._build_one(
                target_date, references[target_date],
                stack_index, mask_index, temporal, cloud,
            )
            if spec is not None:
                specs.append(spec)

        if not specs:
            raise ValueError("No processable targets (missing stacks/masks/references)")
        logger.info("Assembled %d processable sample spec(s)", len(specs))
        return specs

    def _build_one(
        self,
        target_date: str,
        ref_dates: list[str],
        stack_index: dict[str, Path],
        mask_index: dict[str, Path],
        temporal: dict[str, dict[str, object]],
        cloud: dict[str, float],
    ) -> SampleSpec | None:
        """Resolve a single target into a :class:`SampleSpec`, or ``None``.

        Args:
            target_date: Compact target date.
            ref_dates: Ranked compact reference dates for this target.
            stack_index: Date -> stack path index.
            mask_index: Date -> mask path index.
            temporal: Calendar attribute lookup.
            cloud: Cloud-percentage lookup.

        Returns:
            A populated :class:`SampleSpec`, or ``None`` if unprocessable.
        """
        target_stack = stack_index.get(target_date)
        target_mask = mask_index.get(target_date)
        if target_stack is None or target_mask is None:
            logger.warning(
                "Skipping target %s: missing %s",
                target_date,
                "stack" if target_stack is None else "mask",
            )
            return None

        # Candidate references are the ranked list, capped at the maximum, with
        # NO padding: the shared builder keeps each real reference once and the
        # PyTorch loader pads to a fixed slot count at load time instead.
        max_refs = self.config.references.maximum
        resolved_dates: list[str] = []
        resolved_stacks: list[Path] = []
        resolved_masks: list[Path | None] = []
        for ref_date in ref_dates:
            ref_stack = stack_index.get(ref_date)
            if ref_stack is None:
                logger.debug("Reference %s for %s has no stack", ref_date, target_date)
                continue
            resolved_dates.append(ref_date)
            resolved_stacks.append(ref_stack)
            resolved_masks.append(mask_index.get(ref_date))
            if len(resolved_dates) >= max_refs:
                break

        if not resolved_stacks:
            logger.warning("Skipping target %s: no resolvable references", target_date)
            return None

        attrs = temporal.get(target_date, {})
        return SampleSpec(
            target_date=target_date,
            target_stack=target_stack,
            target_mask=target_mask,
            reference_dates=resolved_dates,
            reference_stacks=resolved_stacks,
            reference_masks=resolved_masks,
            cloud_percentage=cloud.get(target_date),
            season=attrs.get("season"),
            year=attrs.get("year"),
            month=attrs.get("month"),
            day_of_year=attrs.get("day_of_year"),
        )
