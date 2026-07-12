"""Command-line entry point for the baseline scientific audit.

Examples:
    # Quick local smoke (small caps, CPU):
    python -m s2audit --dataset DATA --checkpoint best.pt --output audit \\
        --max-samples 200 --gt-max-patches 500 --leakage-max-samples 4000

    # Full Kaggle run (all parts, GPU, whole test split):
    python -m s2audit --dataset /kaggle/input/.../synthetic_dataset \\
        --checkpoint /kaggle/input/.../best.pt --output /kaggle/working/audit --full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("s2audit", description="Baseline scientific audit & evaluation")
    p.add_argument("--dataset", required=True, help="Synthetic dataset root")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--output", default="audit", help="Output directory")
    p.add_argument("--parts", default="all",
                   help="Comma list of gt,leakage,eval (default all)")
    p.add_argument("--full", action="store_true",
                   help="Full evaluation (all patches/samples, no caps)")
    # Caps (ignored when --full).
    p.add_argument("--max-samples", type=int, default=200, help="Test samples to evaluate")
    p.add_argument("--gt-max-patches", type=int, default=1000, help="Unique GT patches to audit")
    p.add_argument("--leakage-max-samples", type=int, default=0,
                   help="Per-split manifests scanned for leakage (0 = all)")
    # Runtime.
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--reflectance-scale", type=float, default=10000.0)
    p.add_argument("--visualize-n", type=int, default=8)
    p.add_argument("--s2cloudless", action="store_true",
                   help="Use s2cloudless second opinion in the GT audit when installed")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parts = {"gt", "leakage", "eval"} if args.parts == "all" else set(args.parts.split(","))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    max_samples = 0 if args.full else args.max_samples
    gt_max = 0 if args.full else args.gt_max_patches
    leak_max = 0 if args.full else args.leakage_max_samples

    gt_summary = leakage_report = eval_report = None

    if "gt" in parts:
        from .gt_quality import audit_ground_truth
        print("[1/3] Ground-truth quality audit ...", flush=True)
        gt_summary = audit_ground_truth(
            args.dataset, out, max_patches=gt_max, reflectance_scale=args.reflectance_scale,
            use_s2cloudless=args.s2cloudless, seed=args.seed)
        print("      ", gt_summary["counts"], flush=True)

    if "leakage" in parts:
        from .leakage import audit_leakage
        print("[2/3] Data-leakage audit ...", flush=True)
        leakage_report = audit_leakage(args.dataset, max_samples=leak_max)
        (out / "data_leakage_audit.json").write_text(
            json.dumps(leakage_report, indent=2), encoding="utf-8")
        print("      ", leakage_report["overall_status"], flush=True)

    if "eval" in parts:
        from .evaluate import evaluate_test_split
        print("[3/3] Test-split evaluation ...", flush=True)
        eval_report = evaluate_test_split(
            args.checkpoint, args.dataset, out, max_samples=max_samples,
            batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
            seed=args.seed, reflectance_scale=args.reflectance_scale,
            visualize_n=args.visualize_n)
        cloud = eval_report["region_metrics"]["cloud"]
        print(f"       cloud PSNR micro={cloud['psnr_micro']:.2f} macro={cloud['psnr_macro']:.2f}",
              flush=True)

    from .report import build_summary, write_summary
    summary = build_summary(gt_summary=gt_summary, leakage=leakage_report,
                            evaluation=eval_report)
    path = write_summary(summary, out)
    print(f"\nOVERALL: {summary['overall_status']}")
    print(f"Wrote {path}")
    for k, v in summary["scientific_answers"].items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
