import argparse
import csv
import math
import os
import time

import numpy as np

from suite_common import (
    SuiteProfile,
    commuting_gaussian_prediction,
    edgeworth_k2_k4_expectation,
    ensure_dir,
    exact_reduced_density_bosonic,
    exact_static_mixture,
    gauss_hermite_expectation_matrix,
    hmf_from_rho,
    moments_from_distribution,
    qutrit_models,
    static_spin_distribution,
    trace_distance,
    write_json,
)


PROFILES = {
    "quick": {
        "core": SuiteProfile(
            name="quick",
            cutoff_list=[4, 6],
            beta_list=[0.8, 1.2],
            coupling_list=[0.08, 0.14],
            mc_samples=20000,
            quad_order=24,
            edgeworth_order=4,
        ),
        "n_list": [4, 8, 12],
        "m_list": [5000, 20000, 60000],
    },
    "full": {
        "core": SuiteProfile(
            name="full",
            cutoff_list=[4, 6, 8, 10],
            beta_list=[0.7, 1.0, 1.4, 1.8],
            coupling_list=[0.05, 0.08, 0.12, 0.16, 0.2],
            mc_samples=60000,
            quad_order=40,
            edgeworth_order=4,
        ),
        "n_list": [4, 8, 12, 16],
        "m_list": [10000, 40000, 120000],
    },
    "publish": {
        "core": SuiteProfile(
            name="publish",
            cutoff_list=[4, 6, 8, 10, 12],
            beta_list=[0.6, 0.8, 1.0, 1.3, 1.6, 2.0],
            coupling_list=[0.04, 0.06, 0.09, 0.12, 0.16, 0.2, 0.24],
            mc_samples=120000,
            quad_order=64,
            edgeworth_order=4,
        ),
        "n_list": [4, 8, 12, 16],
        "m_list": [20000, 80000, 240000],
    },
}

REGIMES = ("cg", "cng", "ncg", "ncng")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def magnetization_levels(n):
    ms = np.arange(-n, n + 1, 2, dtype=int)
    gs = np.array([math.comb(n, (n + m) // 2) for m in ms], dtype=float)
    return ms, gs


def analytic_magnetization_probs(n, k2, k4):
    ms, gs = magnetization_levels(n)
    ex = k2 * ms**2 + k4 * ms**4
    ex = ex - np.max(ex)
    w = gs * np.exp(ex)
    p = w / np.sum(w)
    return ms, p


def complex_scale(var):
    if abs(var) < 1e-15:
        return 0.0 + 0.0j
    return np.sqrt(abs(var)) * np.exp(0.5j * np.angle(var))


def mc_magnetization_probs(n, k2, k4, m_samples, seed):
    rng = np.random.default_rng(seed)
    ms, gs = magnetization_levels(n)

    sx = complex_scale(2.0 * k2)
    sy = complex_scale(8.0 * k4)

    x = sx * rng.standard_normal(m_samples)
    y = sy * rng.standard_normal(m_samples)

    wm = np.zeros_like(ms, dtype=np.complex128)
    for i, (m, g) in enumerate(zip(ms, gs)):
        wm[i] = g * np.mean(np.exp(m * x + 0.5 * (m**2) * y))

    wr = np.maximum(np.real(wm), 0.0)
    p = wr / np.sum(wr)
    leakage = float(np.max(np.abs(np.imag(wm)) / (np.abs(np.real(wm)) + 1e-15)))
    return ms, p, leakage


def run_cg(profile, data_dir):
    models = qutrit_models()
    hs = models["hs_comm"]
    f = models["f_comm"]

    lambda_target = 0.08
    mode_sets = {
        "A": np.array([0.8, 1.4], dtype=float),
        "B": np.array([0.6, 1.2, 2.0], dtype=float),
    }
    base_weights = {
        "A": np.array([1.0, 0.8], dtype=float),
        "B": np.array([1.0, 0.7, 0.5], dtype=float),
    }

    rows = []
    for label, omegas in mode_sets.items():
        w = base_weights[label]
        scale = np.sqrt(2.0 * lambda_target / np.sum((w * w) / (omegas * omegas)))
        cs = scale * w

        for beta in profile.beta_list:
            rho_pred, lam = commuting_gaussian_prediction(hs, f, omegas, cs, beta)
            for cutoff in profile.cutoff_list:
                rho_ed = exact_reduced_density_bosonic(hs, f, omegas, cs, cutoff, beta)
                rows.append(
                    {
                        "test_id": "NHS-CG-1",
                        "mode_set": label,
                        "beta": beta,
                        "cutoff": cutoff,
                        "lambda": lam,
                        "beta_lambda": beta * lam,
                        "trace_distance": trace_distance(rho_ed, rho_pred),
                        "p0_exact": float(np.real_if_close(rho_ed[0, 0])),
                        "p1_exact": float(np.real_if_close(rho_ed[1, 1])),
                        "p2_exact": float(np.real_if_close(rho_ed[2, 2])),
                        "p0_pred": float(np.real_if_close(rho_pred[0, 0])),
                        "p1_pred": float(np.real_if_close(rho_pred[1, 1])),
                        "p2_pred": float(np.real_if_close(rho_pred[2, 2])),
                    }
                )

    write_csv(os.path.join(data_dir, "cg_nhs_cg_1.csv"), rows)


def run_cng(profile, cfg, data_dir, seed):
    rows = []
    curve_rows = []
    a = 0.28
    b = 0.035
    for n in cfg["n_list"]:
        for sign_label, sign in (("PSD", 1.0), ("nonPSD", -1.0)):
            k2 = a / float(n)
            k4 = sign * b / float(n * n)
            _, p_exact = analytic_magnetization_probs(n, k2, k4)

            for m_samples in cfg["m_list"]:
                _, p_mc, leak = mc_magnetization_probs(n, k2, k4, m_samples, seed + n + m_samples)
                td = 0.5 * np.sum(np.abs(p_exact - p_mc))
                rows.append(
                    {
                        "test_id": "NHS-CNG-1",
                        "N": n,
                        "branch": sign_label,
                        "k2": k2,
                        "k4": k4,
                        "M": m_samples,
                        "trace_distance": td,
                        "imag_leakage": leak,
                        "collapse_x": k2 * n,
                        "collapse_y": k4 * n * n,
                    }
                )
                if m_samples == max(cfg["m_list"]):
                    ms, _ = magnetization_levels(n)
                    for m_val, p_e, p_m in zip(ms, p_exact, p_mc):
                        curve_rows.append(
                            {
                                "test_id": "NHS-CNG-1",
                                "N": n,
                                "branch": sign_label,
                                "k2": k2,
                                "k4": k4,
                                "M": m_samples,
                                "m": int(m_val),
                                "p_exact": float(p_e),
                                "p_mc": float(p_m),
                            }
                        )

    write_csv(os.path.join(data_dir, "cng_nhs_cng_1.csv"), rows)
    write_csv(os.path.join(data_dir, "cng_nhs_cng_1_curves.csv"), curve_rows)


def run_ncg(profile, data_dir, seed):
    models = qutrit_models()
    hs = models["hs_noncomm"]
    f = models["f_noncomm"]

    omega = np.array([1.05], dtype=float)
    cutoff = max(profile.cutoff_list)
    rows = []
    for beta in profile.beta_list:
        for g in profile.coupling_list:
            cs = np.array([g], dtype=float)
            rho_ed = exact_reduced_density_bosonic(hs, f, omega, cs, cutoff, beta)

            # Match variance with bosonic reorganization proxy.
            lam = float(np.sum((cs**2) / (2.0 * omega * omega)))
            sigma = np.sqrt(max(2.0 * lam / max(beta, 1e-12), 1e-12))

            rho_ref = gauss_hermite_expectation_matrix(hs, f, beta, sigma, profile.quad_order)
            td_ref = trace_distance(rho_ed, rho_ref)

            offdiag_ed = np.linalg.norm(hmf_from_rho(rho_ed, beta) - np.diag(np.diag(hmf_from_rho(rho_ed, beta))), ord="fro")
            rows.append(
                {
                    "test_id": "NHS-NCG-1",
                    "beta": beta,
                    "coupling": g,
                    "g2": g * g,
                    "trace_distance_pred": td_ref,
                    "offdiag_norm": float(offdiag_ed),
                    "p0_exact": float(np.real_if_close(rho_ed[0, 0])),
                    "p1_exact": float(np.real_if_close(rho_ed[1, 1])),
                    "p2_exact": float(np.real_if_close(rho_ed[2, 2])),
                    "p0_pred": float(np.real_if_close(rho_ref[0, 0])),
                    "p1_pred": float(np.real_if_close(rho_ref[1, 1])),
                    "p2_pred": float(np.real_if_close(rho_ref[2, 2])),
                }
            )

    write_csv(os.path.join(data_dir, "ncg_nhs_ncg_1.csv"), rows)


def run_ncng(profile, data_dir):
    models = qutrit_models()
    hs = models["hs_noncomm"]
    f = models["f_noncomm"]

    templates = [
        np.array([0.08, 0.08, 0.16, 0.16], dtype=float),
        np.array([0.05, 0.10, 0.15, 0.20], dtype=float),
        np.array([0.06, 0.12, 0.12, 0.20], dtype=float),
    ]

    rows = []
    for beta in profile.beta_list:
        for g in profile.coupling_list:
            for ti, t in enumerate(templates):
                couplings = g * t
                values, probs = static_spin_distribution(couplings)
                rho_exact = exact_static_mixture(hs, f, beta, values, probs)

                _, k2, _, k4 = moments_from_distribution(values, probs)
                rho_k2 = gauss_hermite_expectation_matrix(hs, f, beta, np.sqrt(max(k2, 1e-12)), profile.quad_order)
                rho_k2k4 = edgeworth_k2_k4_expectation(hs, f, beta, k2, k4, profile.quad_order)

                err_k2 = trace_distance(rho_exact, rho_k2)
                err_k2k4 = trace_distance(rho_exact, rho_k2k4)

                rows.append(
                    {
                        "test_id": "NHS-NCNG-1",
                        "template": ti,
                        "beta": beta,
                        "coupling": g,
                        "kappa2": k2,
                        "kappa4": k4,
                        "chi4": float(k4 / (k2 * k2 + 1e-12)),
                        "trace_distance_k2": err_k2,
                        "trace_distance_k2k4": err_k2k4,
                        "improvement_ratio": err_k2 / max(err_k2k4, 1e-12),
                        "p0_exact": float(np.real_if_close(rho_exact[0, 0])),
                        "p1_exact": float(np.real_if_close(rho_exact[1, 1])),
                        "p2_exact": float(np.real_if_close(rho_exact[2, 2])),
                        "p0_k2": float(np.real_if_close(rho_k2[0, 0])),
                        "p1_k2": float(np.real_if_close(rho_k2[1, 1])),
                        "p2_k2": float(np.real_if_close(rho_k2[2, 2])),
                        "p0_k2k4": float(np.real_if_close(rho_k2k4[0, 0])),
                        "p1_k2k4": float(np.real_if_close(rho_k2k4[1, 1])),
                        "p2_k2k4": float(np.real_if_close(rho_k2k4[2, 2])),
                    }
                )

    write_csv(os.path.join(data_dir, "ncng_nhs_ncng_1.csv"), rows)


def write_manifest(out_dir, args, profile_name, duration):
    payload = {
        "suite": "nested_hs",
        "profile": profile_name,
        "regime": args.regime,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "duration_sec": duration,
    }
    write_json(os.path.join(out_dir, "manifest.json"), payload)


def parse_args():
    p = argparse.ArgumentParser(description="Run nested HS suite")
    p.add_argument("--regime", choices=["cg", "cng", "ncg", "ncng", "all"], default="all")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="quick")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)
    ensure_dir(data_dir)

    cfg = PROFILES[args.profile]
    profile = cfg["core"]

    t0 = time.time()
    regimes = REGIMES if args.regime == "all" else (args.regime,)

    if "cg" in regimes:
        run_cg(profile, data_dir)
    if "cng" in regimes:
        run_cng(profile, cfg, data_dir, args.seed)
    if "ncg" in regimes:
        run_ncg(profile, data_dir, args.seed)
    if "ncng" in regimes:
        run_ncng(profile, data_dir)

    elapsed = time.time() - t0
    write_manifest(out_dir, args, profile.name, elapsed)
    print(f"nested_hs suite complete in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
