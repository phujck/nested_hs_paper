import argparse
import os

import numpy as np

from suite_common_v2 import ensure_dir, write_json


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def slope_fit_log(x, y):
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), 1e-15)
    lx = np.log(x)
    ly = np.log(y)
    a = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(a, ly, rcond=None)
    return float(coef[0]), float(coef[1])


def cutoff_improvement_fraction(cg_benchmark):
    checks = []
    couplings = np.unique(cg_benchmark["coupling"])
    betas = np.unique(cg_benchmark["beta"])
    for g in couplings:
        for beta in betas:
            sub = cg_benchmark[np.isclose(cg_benchmark["coupling"], g) & np.isclose(cg_benchmark["beta"], beta)]
            if len(sub) < 2:
                continue
            sub = np.sort(sub, order="cutoff")
            td = np.asarray(sub["td_ed_analytic"], dtype=float)
            checks.append(float(td[-1] <= td[0]))
    return float(np.mean(checks)) if checks else 0.0


def lambda_collapse_deviation(cg_benchmark):
    max_cut = np.max(cg_benchmark["cutoff"])
    sub = cg_benchmark[cg_benchmark["cutoff"] == max_cut]
    vals = []
    for g in np.unique(sub["coupling"]):
        sg = sub[np.isclose(sub["coupling"], g)]
        scale = max(float(g) ** 2, 1e-15)
        lam_scaled = np.asarray(sg["lambda_est_ed"], dtype=float) / scale
        vals.append(float(np.max(lam_scaled) - np.min(lam_scaled)))
    return float(max(vals)) if vals else 0.0


def validate(data_dir):
    cg = read_csv(os.path.join(data_dir, "cg_nhs_cg_1_v2.csv"))
    cng = read_csv(os.path.join(data_dir, "cng_nhs_cng_1_v2.csv"))
    synth = read_csv(os.path.join(data_dir, "cng_nhs_synth_nonpsd_v2.csv"))
    phase = read_csv(os.path.join(data_dir, "cng_nhs_phase_clock_v2.csv"))

    cg_bench = cg[cg["record_type"] == "benchmark"]
    cg_scaling = cg[cg["record_type"] == "mc_scaling"]

    scalar_scaling = cg_scaling[cg_scaling["mc_method"] == "hs_scalar"]
    path_scaling = cg_scaling[cg_scaling["mc_method"] == "hs_path"]
    scalar_slope, scalar_inter = slope_fit_log(scalar_scaling["mc_M"], scalar_scaling["mc_error_to_analytic"])
    path_slope, path_inter = slope_fit_log(path_scaling["mc_M"], path_scaling["mc_error_to_analytic"])

    frac_cutoff = cutoff_improvement_fraction(cg_bench)
    collapse_dev = lambda_collapse_deviation(cg_bench)

    frac_improve_24 = float(np.mean(cng["td_k24"] < cng["td_k2"]))
    frac_nonworse_246 = float(np.mean(cng["td_k246"] <= (cng["td_k24"] + 1e-12)))
    resolved_mask = cng["k6_resolved"] > 0.5
    if np.any(resolved_mask):
        frac_improve_246_resolved = float(np.mean(cng["td_k246"][resolved_mask] < cng["td_k24"][resolved_mask]))
        median_improve_246 = float(np.median(cng["improvement_246"][resolved_mask]))
        max_stability_resolved = float(np.max(cng["stability_rel_6"][resolved_mask]))
    else:
        frac_improve_246_resolved = 1.0
        median_improve_246 = 1.0
        max_stability_resolved = 0.0
    median_improve_24 = float(np.median(cng["improvement_24"]))
    synth_td_max = float(np.max(synth["trace_distance"]))
    synth_leak_max = float(np.max(synth["imag_leakage"]))
    phase_err_max = float(np.max(phase["max_abs_error"]))
    phase_td_max = float(np.max(phase["trace_distance"]))
    hmf_exact_td_max = float(max(np.nanmax(cg_bench["td_ed_hmf_exact"]), np.nanmax(cng["td_ed_hmf_exact"])))

    metrics = {
        "trace_distance": float(max(np.max(cg_bench["td_ed_analytic"]), np.max(cng["td_k24"]))),
        "fro_error": float(np.max(cng["fro_k246"])),
        "hmf_basis_residual": hmf_exact_td_max,
        "collapse_max_deviation": collapse_dev,
        "slope_fit": {
            "hs_scalar_log_slope": {"slope": scalar_slope, "intercept": scalar_inter},
            "hs_path_log_slope": {"slope": path_slope, "intercept": path_inter},
        },
        "improvement_ratio": median_improve_24,
        "auxiliary": {
            "cutoff_improvement_fraction": frac_cutoff,
            "frac_improve_k24_over_k2": frac_improve_24,
            "frac_nonworse_k246_over_k24": frac_nonworse_246,
            "frac_improve_k246_over_k24_resolved": frac_improve_246_resolved,
            "k6_resolved_fraction": float(np.mean(resolved_mask.astype(float))),
            "max_stability_rel_resolved": max_stability_resolved,
            "improvement_ratio_k246_resolved": median_improve_246,
            "synth_trace_distance_max": synth_td_max,
            "synth_imag_leakage_max": synth_leak_max,
            "phase_max_abs_error": phase_err_max,
            "phase_trace_distance_max": phase_td_max,
            "hmf_exact_rebuild_max_td": hmf_exact_td_max,
            "max_trace_error": float(
                max(
                    np.max(cg_bench["trace_err_ed"]),
                    np.max(cg_bench["trace_err_hs_scalar"]),
                    np.max(cg_bench["trace_err_hs_path"]),
                    np.max(cng["trace_err_ed"]),
                    np.max(cng["trace_err_k246"]),
                )
            ),
        },
        "pass_fail": False,
    }

    checks = [
        metrics["trace_distance"] < 3.0e-1,
        metrics["collapse_max_deviation"] < 8.0e-2,
        metrics["improvement_ratio"] > 1.10,
        frac_cutoff > 0.75,
        frac_improve_24 > 0.60,
        frac_nonworse_246 > 0.95,
        (-0.8 < scalar_slope < -0.2),
        (-0.8 < path_slope < -0.2),
        synth_td_max < 8.0e-2,
        synth_leak_max < 5.0,
        phase_err_max < 8.0e-2,
        phase_td_max < 8.0e-2,
        hmf_exact_td_max < 1.0e-10,
        metrics["auxiliary"]["max_stability_rel_resolved"] < 0.50,
        metrics["auxiliary"]["max_trace_error"] < 5.0e-9,
    ]
    metrics["pass_fail"] = bool(all(checks))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Validate commuting-only nested HS v2 suite")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base, "..", "results_v2"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)
    metrics = validate(data_dir)
    write_json(os.path.join(out_dir, "claim_metrics_nested_hs_v2.json"), metrics)
    print("nested_hs_v2 validation:", "PASS" if metrics["pass_fail"] else "FAIL")
    print(metrics)


if __name__ == "__main__":
    main()
