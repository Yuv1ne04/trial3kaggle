"""Command-line interface for the synthetic supervision pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import SyntheticSupervisionBuilder
from .config import SyntheticConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="s2dataset.synthetic",
        description=(
            "Generate scientifically-valid supervised training pairs by "
            "transplanting real Sentinel-2 cloud masks onto clear ground-truth "
            "patches (with curriculum difficulty). No neural network."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("synthetic_config.yaml"),
                        help="YAML configuration file (default: synthetic_config.yaml).")
    parser.add_argument("--stacks-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--mask-library-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variants", type=int, default=None, dest="variants_per_patch")
    parser.add_argument("-w", "--workers", type=int, default=None, dest="num_workers")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    """Build a :class:`SyntheticConfig` from CLI args + YAML.

    Args:
        args: Parsed argument namespace.

    Returns:
        A configured :class:`SyntheticConfig`.
    """
    overrides: dict[str, object] = {
        "stacks_dir": args.stacks_dir,
        "masks_dir": args.masks_dir,
        "mask_library_dir": args.mask_library_dir,
        "output_dir": args.output_dir,
        "variants_per_patch": args.variants_per_patch,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    if args.no_resume:
        overrides["resume"] = False
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.config.exists():
        return SyntheticConfig.from_yaml(args.config, **overrides)
    return SyntheticConfig(**overrides)  # type: ignore[arg-type]


def run(argv: list[str] | None = None) -> int:
    """Entry point for the synthetic-supervision CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, ``2`` on a fatal error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        statistics = SyntheticSupervisionBuilder(config).run()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted (resume to continue).", file=sys.stderr)
        return 2
    _print_summary(statistics)
    return 0


def _print_summary(statistics: dict[str, object]) -> None:
    """Print a concise terminal summary of the run.

    Args:
        statistics: The statistics dictionary.
    """
    rule = "=" * 64
    print()
    print(rule)
    print("  SYNTHETIC SUPERVISION SUMMARY")
    print(rule)
    print(f"  Clear GT patches      : {statistics.get('clear_ground_truth_patches')}")
    print(f"  Written samples       : {statistics.get('written_samples')}")
    print(f"  Rejected / failed     : {statistics.get('rejected_samples')} / "
          f"{statistics.get('failed_samples')}")
    print(f"  Split distribution    : {statistics.get('split_distribution')}")
    print(f"  Curriculum            : {statistics.get('curriculum_distribution')}")
    print(f"  Avg cloud coverage    : {statistics.get('average_cloud_coverage')}")
    print(f"  Reference counts      : {statistics.get('reference_count_distribution')}")
    print(f"  Mask reuse            : {statistics.get('mask_reuse')}")
    print(rule)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
