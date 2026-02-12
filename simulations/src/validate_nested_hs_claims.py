import argparse
import os

import numpy as np

from suite_common import ensure_dir, write_json


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def slope_fit_log(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.maximum(np.asarray(y, dtype=float), 1e-15))
    a = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(a, ly, rcond=None)
    return float(coef[0]), float(coef[1])


def validate(data_dir):
    cg = read_csv(os.path.join(data_dir, "cg_nhs_cg_1.csv"))
    cng = read_csv(os.path.join(data_dir, "cng_nhs_cng_1.csv"))
    ncg = read_csv(os.path.join(data_dir, "ncg_nhs_ncg_1.csv"))
    ncng = read_csv(os.path.join(data_dir, "ncng_nhs_ncng_1.csv"))

    metrics = {
        "trace_distance": float(np.max(np.r_[cg["trace_distance"], cng["trace_distance"], ncg["trace_distance_pred"], ncng["trace_distance_k2k4"]])),
        "fro_error": float(np.max(ncng["trace_distance_k2k4"])),
        "hmf_basis_residual": float(np.max(ncg["offdiag_norm"])),
        "collapse_max_deviation": 0.0,
        "slope_fit": {},
        "improvement_ratio": float(np.median(ncng["improvement_ratio"])),
        "pass_fail": True,
    }

    collapse_devs = []
    for bl in np.unique(np.round(cg["beta_lambda"], 8)):
        sub = cg[np.abs(cg["beta_lambda"] - bl) < 1e-9]
        max_cut = np.max(sub["cutoff"])
        sub = sub[sub["cutoff"] == max_cut]
        if len(sub) > 1:
            collapse_devs.append(float(np.max(sub["p0_exact"] - sub["p0_pred"]) - np.min(sub["p0_exact"] - sub["p0_pred"])))
    metrics["collapse_max_deviation"] = float(max(collapse_devs) if collapse_devs else 0.0)

    n_max = np.max(cng["N"])
    sub = cng[(cng["N"] == n_max) & (cng["branch"] == "PSD")]
    slope, intercept = slope_fit_log(sub["M"], sub["trace_distance"])
    metrics["slope_fit"] = {"cng_log_slope": {"slope": slope, "intercept": intercept}}

    checks = []
    checks.append(metrics["trace_distance"] < 4.0e-1)
    checks.append(metrics["collapse_max_deviation"] < 6.0e-2)
    checks.append(metrics["improvement_ratio"] > 1.05)
    checks.append(slope < -0.35)

    metrics["pass_fail"] = bool(all(checks))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Validate nested HS claims")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)

    metrics = validate(data_dir)
    write_json(os.path.join(out_dir, "claim_metrics_nested_hs.json"), metrics)
    print("nested_hs validation:", "PASS" if metrics["pass_fail"] else "FAIL")
    print(metrics)


if __name__ == "__main__":
    main()
