"""Command-line interface for the dataset builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import DatasetBuilder
from .config import DatasetConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="s2dataset",
        description=(
            "Build AI-ready Sentinel-2 cloud-reconstruction training samples "
            "(GeoTIFF + NPZ) with a leakage-free temporal split. No ML."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("dataset_config.yaml"),
        help="YAML configuration file (default: dataset_config.yaml).",
    )
    parser.add_argument("--stacks-dir", type=Path, default=None,
                        help="Override: directory of 13-band stacks.")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="Override: directory of cloud masks.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override: dataset output directory.")
    parser.add_argument("-n", "--n-references", type=int, default=None,
                        help="Override: references per sample.")
    parser.add_argument("-w", "--workers", type=int, default=None, dest="num_workers",
                        help="Override: parallel worker processes.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume; rebuild from scratch.")
    parser.add_argument("--no-geotiff", action="store_true",
                        help="Disable the GeoTIFF output format.")
    parser.add_argument("--no-npz", action="store_true",
                        help="Disable the NPZ output format.")
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetConfig:
    """Build a :class:`DatasetConfig` from CLI args (YAML + overrides).

    Args:
        args: Parsed argument namespace.

    Returns:
        A configured :class:`DatasetConfig`.

    Raises:
        FileNotFoundError: If a required override (stacks/masks) is absent and
            the YAML does not supply it.
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
    if args.no_geotiff:
        overrides["write_geotiff"] = False
    if args.no_npz:
        overrides["write_npz"] = False
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.config.exists():
        return DatasetConfig.from_yaml(args.config, **overrides)
    if "stacks_dir" not in overrides or "masks_dir" not in overrides:
        raise FileNotFoundError(
            f"Config file {args.config} not found and --stacks-dir/--masks-dir "
            "not both provided."
        )
    return DatasetConfig(**overrides)  # type: ignore[arg-type]


def run(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, ``2`` on a fatal error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        statistics = DatasetBuilder(config).run()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nBuild interrupted by user (resume to continue).", file=sys.stderr)
        return 2

    _print_summary(statistics)
    return 0


def _print_summary(statistics: dict[str, object]) -> None:
    """Print a concise terminal summary of the build.

    Args:
        statistics: The dataset statistics dictionary.
    """
    overall = statistics.get("overall", {})
    per_size = statistics.get("per_patch_size", {})
    rule = "=" * 64
    print()
    print(rule)
    print("  DATASET BUILD SUMMARY")
    print(rule)
    print(f"  Patch sizes         : {statistics.get('patch_sizes')}")
    print(f"  Total samples       : {overall.get('total_samples')}")
    print(f"  Train / Val / Test  : {overall.get('training_samples')} / "
          f"{overall.get('validation_samples')} / {overall.get('testing_samples')}")
    print(f"  References / sample : {statistics.get('average_references')}")
    print("  Per patch size:")
    for size, block in per_size.items():
        print(
            f"    {size:>4} (str {block.get('stride')}): "
            f"{block.get('total_samples')} samples "
            f"[{block.get('training_samples')}/{block.get('validation_samples')}/"
            f"{block.get('testing_samples')}] | "
            f"cloud {block.get('average_cloud_coverage_percent')}% | "
            f"nodata {block.get('average_nodata_percentage')}%"
        )
    print(rule)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
