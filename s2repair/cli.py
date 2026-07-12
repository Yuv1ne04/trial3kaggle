"""Command-line entry for the baseline repair (diagnostics + gates).

Examples:
    python -m s2repair diagnose   --checkpoint best.pt --dataset DATA --output repair
    python -m s2repair worstcase  --checkpoint best.pt --dataset DATA --output repair
    python -m s2repair capability --dataset DATA --output repair
    python -m s2repair gate1 --config configs/reference_unet_v2_gate1.yaml \\
        --dataset DATA --output repair --audit-manifest audit/ground_truth_filter_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _gate_params(config_path: str) -> dict:
    """Read the raw ``gate:`` section from a YAML config (dropped by load_config)."""
    import yaml
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return data.get("gate", {})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("s2repair", description="Baseline repair diagnostics & gates")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--dataset", required=True)
        sp.add_argument("--output", default="repair")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--max-samples", type=int, default=240)
        sp.add_argument("--batch-size", type=int, default=8)
        sp.add_argument("--seed", type=int, default=1234)

    d = sub.add_parser("diagnose"); common(d); d.add_argument("--checkpoint", required=True)
    w = sub.add_parser("worstcase"); common(w); w.add_argument("--checkpoint", required=True)
    w.add_argument("--top-k", type=int, default=50); w.add_argument("--render", type=int, default=20)
    c = sub.add_parser("capability"); c.add_argument("--dataset", required=True)
    c.add_argument("--output", default="repair")

    for name in ("gate1", "gate2", "gate3"):
        g = sub.add_parser(name)
        g.add_argument("--config", required=True)
        g.add_argument("--dataset", required=True)
        g.add_argument("--output", default="repair")
        g.add_argument("--audit-manifest", default=None)
        g.add_argument("--device", default="auto")
        g.add_argument("--original-checkpoint", default=None,
                       help="v1 best.pt for the gate3 four-way comparison")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "diagnose":
        from .bounding_diagnostic import run_bounding_diagnostic
        s = run_bounding_diagnostic(args.checkpoint, args.dataset, out,
                                    max_samples=args.max_samples, batch_size=args.batch_size,
                                    device=args.device, seed=args.seed)
        print(json.dumps(s["variants"], indent=2))
        print("train_bounds_available:", s["train_bounds_available"])
        return 0

    if args.command == "worstcase":
        from .worst_case import run_worst_case
        s = run_worst_case(args.checkpoint, args.dataset, out, max_samples=args.max_samples,
                           top_k=args.top_k, batch_size=args.batch_size, device=args.device,
                           seed=args.seed, render=args.render)
        print("worst_by_surface:", s["worst_by_surface"])
        print("worst_by_difficulty:", s["worst_by_difficulty"])
        print("worst_mean_negative_fraction:", round(s["worst_mean_negative_fraction"], 4))
        return 0

    if args.command == "capability":
        from .capability import build_capability_report
        r = build_capability_report(args.dataset, out)
        print(json.dumps({k: v["status"] for k, v in r["capability"].items()}, indent=2))
        return 0

    # Gates.
    from s2train.config import load_config
    from .gates import run_gate1, run_gate2
    config = load_config(args.config)
    config.data.root = args.dataset
    gp = _gate_params(args.config)
    if args.command == "gate1":
        rep = run_gate1(config, args.dataset, out, audit_manifest=args.audit_manifest,
                        n_samples=gp.get("n_samples", 96), epochs=config.trainer.epochs,
                        batch_size=config.data.batch_size, grad_accum=config.trainer.grad_accum_steps,
                        grad_clip=config.trainer.grad_clip_norm, device=args.device,
                        policy=gp.get("policy", "conservative"),
                        native_threshold=gp.get("native_threshold", 0.01),
                        scan_cap=gp.get("scan_cap", 4000))
    elif args.command == "gate2":
        rep = run_gate2(config, args.dataset, out, audit_manifest=args.audit_manifest,
                        n_train=gp.get("n_train", 3000), n_val=gp.get("n_val", 600),
                        epochs=config.trainer.epochs, batch_size=config.data.batch_size,
                        grad_accum=config.trainer.grad_accum_steps,
                        grad_clip=config.trainer.grad_clip_norm, device=args.device,
                        policy=gp.get("policy", "conservative"),
                        native_threshold=gp.get("native_threshold", 0.01),
                        scan_cap=gp.get("scan_cap", 0))
    else:  # gate3: same as gate2 at 30k scale + four-way comparison.
        rep = run_gate2(config, args.dataset, out, audit_manifest=args.audit_manifest,
                        n_train=gp.get("n_train", 30000), n_val=gp.get("n_val", 3000),
                        epochs=config.trainer.epochs, batch_size=config.data.batch_size,
                        grad_accum=config.trainer.grad_accum_steps,
                        grad_clip=config.trainer.grad_clip_norm, device=args.device,
                        policy=gp.get("policy", "conservative"),
                        native_threshold=gp.get("native_threshold", 0.01),
                        scan_cap=gp.get("scan_cap", 0))
        rep["gate"] = 3
    print(f"\n{args.command.upper()} status: {rep['status']}")
    print(json.dumps(rep.get("checks", {}), indent=2))
    print("final:", json.dumps({k: round(v, 4) for k, v in rep.get("final_metrics", {}).items()
                                if isinstance(v, (int, float))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
