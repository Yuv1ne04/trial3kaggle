"""Command-line interface for the shared-reference dataset builder + migration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import DatasetConfig
from .builder import SharedDatasetBuilder


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="s2dataset.shared",
        description=(
            "Build the storage-efficient shared-reference Sentinel-2 dataset "
            "(each patch stored once; samples are JSON references). No ML."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("dataset_config.yaml"),
                        help="YAML configuration file (default: dataset_config.yaml).")
    parser.add_argument("--stacks-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("-n", "--n-references", type=int, default=None)
    parser.add_argument("-w", "--workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--no-resume", action="store_true",
                        help="Rebuild from scratch (ignore checkpoint).")
    parser.add_argument("--migrate-from", type=Path, default=None,
                        help="Convert an old duplicated-NPZ dataset at this path "
                             "into the shared layout (at --output-dir) and exit.")
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    """Build a :class:`DatasetConfig` from CLI args + YAML.

    Args:
        args: Parsed argument namespace.

    Returns:
        A configured :class:`DatasetConfig`.
    """
    overrides: dict[str, object] = {
        "stacks_dir": args.stacks_dir,
        "masks_dir": args.masks_dir,
        "output_dir": args.output_dir,
        "n_references": args.n_references,
        "num_workers": args.num_workers,
    }
    if args.no_resume:
        overrides["resume"] = False
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.config.exists():
        return DatasetConfig.from_yaml(args.config, **overrides)
    return DatasetConfig(**overrides)  # type: ignore[arg-type]


def run(argv: list[str] | None = None) -> int:
    """Entry point for the shared-dataset CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, ``2`` on a fatal error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        if args.migrate_from is not None:
            from ..logging_setup import configure_logging
            from .migrate import migrate_dataset
            configure_logging(config.log_path)
            counts = migrate_dataset(args.migrate_from, config.output_dir)
            print(f"\nMigration complete: {counts}")
            return 0
        statistics = SharedDatasetBuilder(config).run()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nBuild interrupted (resume to continue).", file=sys.stderr)
        return 2

    _print_summary(statistics)
    return 0


def _print_summary(statistics: dict[str, object]) -> None:
    """Print a concise terminal summary of the build.

    Args:
        statistics: The dataset statistics dictionary.
    """
    storage = statistics.get("storage", {})
    rule = "=" * 64
    print()
    print(rule)
    print("  SHARED-REFERENCE DATASET SUMMARY")
    print(rule)
    print(f"  Patch sizes            : {statistics.get('patch_sizes')}")
    print(f"  Total samples          : {statistics.get('total_samples')}")
    print(f"  Reference distribution : {statistics.get('reference_count_distribution')}")
    print(f"  Target patches         : {storage.get('target_patches')}")
    print(f"  Reference patches      : {storage.get('reference_patches')}")
    print(f"  Mask patches           : {storage.get('mask_patches')}")
    print(f"  Deduplication factor   : {storage.get('deduplication_factor')}x")
    print(f"  Estimated (pre-gen)    : {storage.get('pre_generation_estimate_human')}")
    print(f"  New size (shared)      : {storage.get('new_actual_human')}")
    print(f"  Old size (duplicated)  : ~{storage.get('old_estimated_human')}")
    print(f"  Saved                  : ~{storage.get('savings_human')}")
    print(rule)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
