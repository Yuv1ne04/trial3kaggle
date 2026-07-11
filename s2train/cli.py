"""Command-line interface: train / evaluate / infer, all config-driven."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="s2train",
        description="Config-driven experiment framework for Sentinel-2 cloud "
                    "reconstruction (train / evaluate / infer).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from a YAML config.")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--name", type=str, default=None, help="Override experiment name.")
    p_train.add_argument("--resume", type=str, default=None, help="'auto' or checkpoint path.")
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--sanity", action="store_true",
                         help="Run sanity/overfit mode (tiny subset, few epochs).")
    p_train.add_argument("--limit-batches", type=int, default=None, dest="limit_batches",
                         help="Fast-dev: cap train/val batches per epoch.")

    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint over a split.")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--split", type=str, default=None)
    p_eval.add_argument("--root", type=str, default=None, help="Dataset root override.")
    p_eval.add_argument("--device", type=str, default="auto")
    p_eval.add_argument("--out", type=Path, default=Path("evaluation_report.json"))

    p_infer = sub.add_parser("infer", help="Reconstruct a tile from a checkpoint.")
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--stack", type=Path, required=True, help="Cloudy stack GeoTIFF.")
    p_infer.add_argument("--mask", type=Path, required=True, help="Cloud mask GeoTIFF.")
    p_infer.add_argument("--references", type=Path, nargs="+", required=True)
    p_infer.add_argument("--out", type=Path, required=True, help="Output GeoTIFF.")
    p_infer.add_argument("--device", type=str, default="auto")
    return parser


def _train(args: argparse.Namespace) -> int:
    """Run the training command."""
    from .config import load_config
    from .trainers import Trainer

    overrides = {}
    if args.name:
        overrides["name"] = args.name
    if args.resume:
        overrides["resume"] = args.resume
    config = load_config(args.config, **overrides)
    if args.device:
        config.trainer.device = args.device
    if args.epochs:
        config.trainer.epochs = args.epochs
    if args.limit_batches is not None:
        config.trainer.limit_batches = args.limit_batches

    if args.sanity or config.sanity.enabled:
        from .sanity import run_sanity

        result = run_sanity(config)
        print(f"\nSANITY {'PASSED' if result['passed'] else 'FAILED'}: {result}")
        return 0 if result["passed"] else 1

    Trainer(config).fit()
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation command."""
    from .evaluation import Evaluator

    evaluator = Evaluator.from_checkpoint(args.checkpoint, device=args.device)
    report = evaluator.evaluate_split(args.split, root=args.root)
    evaluator.write_report(report, args.out)
    print(f"Wrote evaluation report: {args.out}")
    return 0


def _infer(args: argparse.Namespace) -> int:
    """Run the inference command (full-tile reconstruction)."""
    import numpy as np
    import rasterio

    from .inference import Predictor

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    with rasterio.open(args.stack) as ds:
        cloudy = ds.read()
        profile = ds.profile
    with rasterio.open(args.mask) as ds:
        mask = ds.read(1)[None, :, :]
    refs = np.stack([rasterio.open(p).read() for p in args.references], axis=0)

    out = predictor.reconstruct_tile(cloudy, mask, refs)
    profile.update(count=out.shape[0], dtype="uint16", compress="DEFLATE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(np.clip(out, 0, 65535).astype("uint16"))
    print(f"Wrote reconstruction: {args.out}")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "infer":
            return _infer(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
