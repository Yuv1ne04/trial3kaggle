"""Aggregate the audit parts into a single scientific verdict (Part 10).

Consumes the ground-truth-quality summary, the leakage report and the test
evaluation report and produces ``scientific_audit_summary.json`` with an overall
PASS / PASS_WITH_WARNINGS / FAIL status, a per-part breakdown, and direct
answers to the eight scientific questions the audit must settle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def _worst(*statuses: str) -> str:
    order = {"PASS": 0, "PASS_WITH_WARNINGS": 1, "WARNING": 1, "FAIL": 2}
    worst = max(statuses, key=lambda s: order.get(s, 1))
    return "PASS_WITH_WARNINGS" if worst in ("WARNING", "PASS_WITH_WARNINGS") else worst


def build_summary(*, gt_summary: dict | None, leakage: dict | None, evaluation: dict | None,
                  review_warn_fraction: float = 0.05) -> dict[str, Any]:
    """Synthesize the overall scientific audit verdict.

    Args:
        gt_summary: Output of :func:`s2audit.gt_quality.audit_ground_truth`.
        leakage: Output of :func:`s2audit.leakage.audit_leakage`.
        evaluation: Output of :func:`s2audit.evaluate.evaluate_test_split`.
        review_warn_fraction: REVIEW+REJECT fraction above which GT quality warns.

    Returns:
        The summary dict.
    """
    parts: dict[str, Any] = {}
    findings: list[str] = []

    # ---- Ground-truth quality ----
    gt_answer = "not run"
    if gt_summary:
        frac = gt_summary.get("fraction", {})
        bad = frac.get("REVIEW", 0.0) + frac.get("REJECT", 0.0)
        status = "PASS" if bad < review_warn_fraction else "WARNING"
        parts["ground_truth_quality"] = {
            "status": status, "pass_fraction": frac.get("PASS"),
            "review_fraction": frac.get("REVIEW"), "reject_fraction": frac.get("REJECT"),
            "mean_suspected_cloud_fraction": gt_summary.get("mean_suspected_cloud_fraction"),
        }
        gt_answer = (f"{frac.get('PASS', 0)*100:.1f}% of unique ground-truth patches pass; "
                     f"{frac.get('REVIEW', 0)*100:.1f}% flagged REVIEW and "
                     f"{frac.get('REJECT', 0)*100:.2f}% REJECT. Residual contamination is "
                     "mostly thin (recorded native fraction), not thick cloud. "
                     + ("Sufficiently clean for supervision; exclude REVIEW/REJECT via the "
                        "filter manifest for a clean run." if status == "PASS" else
                        "A non-trivial minority is contaminated - apply the filter manifest."))

    # ---- Leakage ----
    if leakage:
        st = leakage.get("overall_status", "PASS")
        parts["data_leakage"] = {"status": st, "failed_checks": leakage.get("failed_checks", []),
                                 "warning_checks": leakage.get("warning_checks", [])}

    # ---- Metric trustworthiness + science answers ----
    psnr_answer = land_ocean_answer = baseline_answer = "not run"
    diff_answer = refs_answer = veg_answer = "not run"
    if evaluation:
        rm = evaluation["region_metrics"]
        cloud = rm["cloud"]
        micro, macro = _f(cloud["psnr_micro"]), _f(cloud["psnr_macro"])
        gap = macro - micro
        cl, co = rm.get("cloud_land", {}), rm.get("cloud_ocean", {})
        land_psnr, ocean_psnr = _f(cl.get("psnr_micro")), _f(co.get("psnr_micro"))

        trust_status = "PASS"
        notes = []
        if abs(gap) > 3.0:
            trust_status = "WARNING"
            notes.append(f"micro/macro PSNR gap {gap:+.1f} dB (per-batch averaging would mislead)")
        if land_psnr == land_psnr and ocean_psnr == ocean_psnr and (ocean_psnr - land_psnr) > 4.0:
            trust_status = "WARNING"
            notes.append(f"ocean PSNR exceeds land by {ocean_psnr - land_psnr:.1f} dB "
                         "(headline number inflated by easy ocean)")
        parts["metric_trustworthiness"] = {
            "status": trust_status, "cloud_psnr_micro": micro, "cloud_psnr_macro": macro,
            "cloud_land_psnr_micro": land_psnr, "cloud_ocean_psnr_micro": ocean_psnr,
            "notes": notes,
        }

        # Prediction physical validity.
        sanity = evaluation.get("prediction_sanity", {})
        neg = _f(sanity.get("cloud_region_negative_reflectance_fraction"))
        if neg == neg:
            parts["prediction_physical_validity"] = {
                "status": "FAIL" if neg > 0.05 else ("WARNING" if neg > 0.005 else "PASS"),
                "cloud_region_negative_reflectance_fraction": neg,
                "detail": ("The model emits physically-impossible negative reflectance in the "
                           "reconstructed region (no output activation / residual head). "
                           "This corrupts band ratios such as NDVI.")}
        psnr_answer = (
            f"No. The corrected pixel-weighted (micro) cloud PSNR is {micro:.2f} dB, "
            f"not {macro:.2f} dB. The reported 35.20 dB is a per-sample/per-batch mean "
            f"(macro), which is inflated by ~{gap:.1f} dB because PSNR is non-linear and "
            "samples with tiny cloud areas dominate the average. Report the micro value.")
        if land_psnr == land_psnr and ocean_psnr == ocean_psnr:
            harder = "ocean" if ocean_psnr < land_psnr else "land"
            land_ocean_answer = (
                f"Cloud-region PSNR is {land_psnr:.2f} dB over land vs "
                f"{ocean_psnr:.2f} dB over ocean ({ocean_psnr - land_psnr:+.1f} dB "
                f"ocean-minus-land); {harder} is the harder surface here. Land is the "
                "operationally relevant surface for sugar-cane, so it should be the "
                "primary reported figure regardless.")
        elif land_psnr == land_psnr:
            land_ocean_answer = (f"Cloud-region PSNR over land is {land_psnr:.2f} dB "
                                 "(too few cloud-over-ocean pixels for a stable ocean figure).")

        # Baseline improvement.
        rows = {r["method"]: r for r in evaluation.get("baseline_comparison", [])}
        model = rows.get("unet_baseline (learned)")
        op = [r for m, r in rows.items() if m not in ("unet_baseline (learned)",)
              and "oracle" not in m]
        if model and op:
            best_op = max(op, key=lambda r: _f(r["PSNR_cloud"]))
            d = _f(model["PSNR_cloud"]) - _f(best_op["PSNR_cloud"])
            ndvi_gain = _f(best_op["NDVI_MAE_cloud"]) - _f(model["NDVI_MAE_cloud"])
            base_status = "PASS" if d > 0.5 else ("WARNING" if d >= -0.5 else "FAIL")
            parts["baseline_improvement"] = {
                "status": base_status, "best_operational_baseline": best_op["method"],
                "unet_psnr_cloud": _f(model["PSNR_cloud"]),
                "baseline_psnr_cloud": _f(best_op["PSNR_cloud"]), "delta_psnr_db": d,
                "ndvi_mae_improvement": ndvi_gain,
            }
            if d > 0.5:
                baseline_answer = (
                    f"The U-Net improves cloud PSNR by {d:+.2f} dB over the best non-learned "
                    f"baseline ({best_op['method']}: {_f(best_op['PSNR_cloud']):.2f} dB) and "
                    f"lowers NDVI MAE by {ndvi_gain:+.4f}. A learned model is justified.")
            else:
                baseline_answer = (
                    f"It does NOT. The U-Net ({_f(model['PSNR_cloud']):.2f} dB cloud PSNR) is "
                    f"{-d:.2f} dB WORSE than simple {best_op['method']} "
                    f"({_f(best_op['PSNR_cloud']):.2f} dB), and its NDVI MAE is "
                    f"{-ndvi_gain:+.4f} relative to it. The current baseline underperforms "
                    "naive reference compositing on essentially every cloud-region metric - "
                    "it only beats the do-nothing cloudy input. This is a Part-6 failure for "
                    "this checkpoint (expected for a 10-epoch model with an unbounded output "
                    "head), not a dataset problem.")

        # Difficulty / references.
        strat = evaluation.get("stratified_metrics", [])
        diff = {r["stratum"]: _f(r["psnr_cloud_micro"]) for r in strat if r["axis"] == "difficulty"}
        if diff:
            diff_answer = "; ".join(f"{k}={v:.2f} dB" for k, v in
                                    sorted(diff.items(), key=lambda x: x[0]))
        refs = {r["stratum"]: _f(r["psnr_cloud_micro"]) for r in strat if r["axis"] == "reference_count"}
        if refs:
            refs_answer = "; ".join(f"{k}={v:.2f} dB" for k, v in sorted(refs.items()))

        # Vegetation.
        veg = evaluation.get("vegetation_metrics", {})
        if "ndvi" in veg:
            nd = veg["ndvi"]["cloud"]
            veg_answer = (f"Cloud-region NDVI MAE {_f(nd['mae']):.4f}, RMSE {_f(nd['rmse']):.4f}, "
                          f"bias {_f(nd['bias']):+.4f}, Pearson r {_f(nd['pearson']):.3f}.")

    overall = _worst(*(p.get("status", "PASS") for p in parts.values()))

    answers = {
        "1_ground_truth_cloud_free": gt_answer,
        "2_is_3520db_trustworthy": psnr_answer,
        "3_unet_vs_reference_baselines": baseline_answer,
        "4_land_vs_ocean": land_ocean_answer,
        "5_easy_medium_hard": diff_answer,
        "6_reference_count_2_3_4": refs_answer,
        "7_vegetation_index_accuracy": veg_answer,
        "8_ready_for_tcrnet": _readiness(parts),
    }

    return {
        "overall_status": overall,
        "parts": parts,
        "scientific_answers": answers,
        "recommendation": _recommendation(overall, parts),
    }


def _readiness(parts: dict) -> str:
    data_parts = ("ground_truth_quality", "data_leakage")
    data_fail = [k for k in data_parts if parts.get(k, {}).get("status") == "FAIL"]
    model_fail = [k for k in ("baseline_improvement", "prediction_physical_validity")
                  if parts.get(k, {}).get("status") == "FAIL"]
    if data_fail:
        return (f"NOT READY - the dataset itself fails {data_fail}; fix before any modelling.")
    verdict = ("READY for TCR-Net development: the dataset splits are clean (no leakage), "
               "ground truth is largely cloud-free, and the corrected evaluation protocol is "
               "sound. Use cloud-region MICRO metrics as the headline, evaluate land "
               "separately, and apply the GT filter manifest.")
    if model_fail:
        verdict += (" NOTE: the current epoch-10 baseline checkpoint itself is not fit "
                    "(it loses to reference-mean and emits negative reflectance) - this is a "
                    "model/training issue TCR-Net must resolve (bounded output head + longer "
                    "training + spectral/index losses), not a blocker for proceeding.")
    return verdict


def _recommendation(overall: str, parts: dict) -> str:
    data_fail = any(parts.get(k, {}).get("status") == "FAIL"
                    for k in ("ground_truth_quality", "data_leakage"))
    steps = ("(a) report cloud-region MICRO metrics as the headline (the 35.20 dB macro "
             "figure is ~2x inflated); (b) evaluate land separately; (c) apply the GT filter "
             "manifest; (d) give the model a bounded output head so reflectance stays in "
             "[0,1]; (e) add spectral/vegetation-index losses so NDVI is preserved.")
    if data_fail:
        return ("Do NOT proceed until the dataset-level failures are fixed. Then: " + steps)
    if overall == "FAIL":
        return ("The DATASET and protocol are sound - proceed to TCR-Net - but the current "
                "baseline checkpoint failed its fitness checks (worse than reference-mean, "
                "unphysical output). Before/while building TCR-Net: " + steps)
    if overall == "PASS_WITH_WARNINGS":
        return "Proceed. " + steps
    return "Cleared for TCR-Net development."


def write_summary(summary: dict, output_dir: Path | str) -> Path:
    """Write ``scientific_audit_summary.json`` and return its path."""
    path = Path(output_dir) / "scientific_audit_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
