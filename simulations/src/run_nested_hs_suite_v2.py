import argparse
import csv
import math
import os
import time

import numpy as np
from scipy.linalg import expm, logm

from suite_common_v2 import (
    EPS,
    SuiteProfileV2,
    commuting_gaussian_prediction,
    commuting_truncated_prediction,
    density_diagnostics,
    diag_probs,
    discretize_ohmic_bath,
    ensure_dir,
    exact_reduced_density,
    exact_reduced_density_bosonic,
    fro_error,
    integrated_even_cumulants,
    lambda_from_discrete,
    lambda_ohmic_continuum,
    level_shift,
    normalize_density,
    path_hs_density,
    qutrit_commuting_model,
    scalar_hs_density,
    spin_bath_ops,
    trace_distance,
    write_json,
)


PROFILES = {
    "quick": SuiteProfileV2(
        name="quick",
        cutoff_list=[4, 6, 8],
        beta_list=[0.7, 1.1, 1.7],
        coupling_list=[0.25, 0.40, 0.55],
        mode_count=2,
        eta=0.28,
        omega_c=1.3,
        omega_max_factor=6.0,
        path_tau_points=18,
        path_samples=30000,
        scalar_samples=30000,
        mc_scaling_samples=[10000, 40000, 160000],
        fd_step=8.0e-3,
        spin_omegas=[0.7, 0.9, 1.1, 1.3, 1.5],
        spin_c_template=[0.55, 0.8, 1.0, 1.2, 1.35],
    ),
    "full": SuiteProfileV2(
        name="full",
        cutoff_list=[4, 6, 8, 10],
        beta_list=[0.6, 0.9, 1.2, 1.6, 2.1],
        coupling_list=[0.25, 0.35, 0.45, 0.60, 0.72],
        mode_count=2,
        eta=0.28,
        omega_c=1.3,
        omega_max_factor=7.0,
        path_tau_points=22,
        path_samples=90000,
        scalar_samples=90000,
        mc_scaling_samples=[20000, 80000, 320000],
        fd_step=7.0e-3,
        spin_omegas=[0.65, 0.85, 1.05, 1.25, 1.45],
        spin_c_template=[0.55, 0.8, 1.0, 1.2, 1.35],
    ),
    "publish": SuiteProfileV2(
        name="publish",
        cutoff_list=[4, 6, 8, 10, 12],
        beta_list=[0.5, 0.75, 1.0, 1.3, 1.7, 2.2, 2.8],
        coupling_list=[0.20, 0.30, 0.40, 0.50, 0.62, 0.75],
        mode_count=2,
        eta=0.28,
        omega_c=1.3,
        omega_max_factor=8.0,
        path_tau_points=26,
        path_samples=180000,
        scalar_samples=180000,
        mc_scaling_samples=[40000, 160000, 640000],
        fd_step=6.0e-3,
        spin_omegas=[0.60, 0.80, 1.00, 1.20, 1.40],
        spin_c_template=[0.55, 0.8, 1.0, 1.2, 1.35],
    ),
}

REGIMES = ("cg", "cng")
K6_STABILITY_MAX = 0.35
PHASE_TAU_CS = [0.04, 0.12, 0.35]
PHASE_BETA = 1.0
PHASE_J = 0.6
PHASE_R = 0.45
PHASE_N_TAU = 18
PHASE_POINTS_BY_PROFILE = {"quick": 25, "full": 61, "publish": 81}
PHASE_REPLICAS_BY_PROFILE = {"quick": 5, "full": 8, "publish": 10}
SYNTH_N = 10
SYNTH_K2 = 0.02
SYNTH_K4_ABS = 0.00015
MC_SCALING_REPS_BY_PROFILE = {"quick": 2, "full": 2, "publish": 3}

CG_COLUMNS = [
    "test_id",
    "record_type",
    "coupling",
    "beta",
    "cutoff",
    "mode_count",
    "lambda_disc",
    "lambda_cont",
    "p0_ed",
    "p1_ed",
    "p2_ed",
    "p0_analytic",
    "p1_analytic",
    "p2_analytic",
    "p0_hs_scalar",
    "p1_hs_scalar",
    "p2_hs_scalar",
    "p0_hs_path",
    "p1_hs_path",
    "p2_hs_path",
    "td_ed_analytic",
    "td_hs_scalar_analytic",
    "td_hs_path_analytic",
    "td_hs_scalar_ed",
    "td_hs_path_ed",
    "td_ed_hmf_exact",
    "lambda_est_ed",
    "lambda_est_analytic",
    "lambda_est_hs_scalar",
    "lambda_est_hs_path",
    "trace_err_ed",
    "trace_err_analytic",
    "trace_err_hs_scalar",
    "trace_err_hs_path",
    "herm_err_ed",
    "herm_err_analytic",
    "herm_err_hs_scalar",
    "herm_err_hs_path",
    "mc_method",
    "mc_M",
    "mc_error_to_analytic",
    "mc_error_to_ed",
    "mc_error_std_to_analytic",
    "mc_error_std_to_ed",
    "mc_scaling_reps",
]

CNG_COLUMNS = [
    "test_id",
    "beta",
    "coupling",
    "num_spins",
    "alpha2",
    "alpha4",
    "alpha6",
    "alpha6_used",
    "chi4",
    "chi6",
    "stability_rel_2",
    "stability_rel_4",
    "stability_rel_6",
    "k6_resolved",
    "p0_ed",
    "p1_ed",
    "p2_ed",
    "p0_k2",
    "p1_k2",
    "p2_k2",
    "p0_k24",
    "p1_k24",
    "p2_k24",
    "p0_k246",
    "p1_k246",
    "p2_k246",
    "td_k2",
    "td_k24",
    "td_k246",
    "td_ed_hmf_exact",
    "fro_k2",
    "fro_k24",
    "fro_k246",
    "improvement_24",
    "improvement_246",
    "shift_lvl1_ed",
    "shift_lvl1_k2",
    "shift_lvl1_k24",
    "shift_lvl1_k246",
    "shift_lvl2_ed",
    "shift_lvl2_k2",
    "shift_lvl2_k24",
    "shift_lvl2_k246",
    "trace_err_ed",
    "trace_err_k2",
    "trace_err_k24",
    "trace_err_k246",
    "herm_err_ed",
    "herm_err_k2",
    "herm_err_k24",
    "herm_err_k246",
]

PHASE_COLUMNS = [
    "test_id",
    "beta",
    "j",
    "r",
    "tau_c",
    "n_tau",
    "phi_over_pi",
    "phi",
    "k2_real",
    "k2_imag",
    "k4_real",
    "k4_imag",
    "p0_analytic",
    "p1_analytic",
    "p2_analytic",
    "p0_mc",
    "p1_mc",
    "p2_mc",
    "p0_mc_std",
    "p1_mc_std",
    "p2_mc_std",
    "trace_distance",
    "trace_distance_std",
    "max_abs_error",
    "max_abs_error_std",
    "imag_leakage",
    "imag_leakage_std",
    "phase_replicas",
    "mc_M_per_rep",
    "mc_M_total",
]

SYNTH_COLUMNS = [
    "test_id",
    "branch",
    "N",
    "k2",
    "k4",
    "mc_M",
    "trace_distance",
    "max_abs_error",
    "imag_leakage",
]

SYNTH_CURVE_COLUMNS = [
    "test_id",
    "branch",
    "N",
    "k2",
    "k4",
    "mc_M",
    "m",
    "p_analytic",
    "p_mc",
]


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _blank_row(columns):
    row = {k: np.nan for k in columns}
    for key in ("test_id", "record_type", "mc_method"):
        if key in row:
            row[key] = ""
    return row


def _lambda_estimate(energies, fvals, probs, beta, idx=2):
    denom = (fvals[idx] ** 2) - (fvals[0] ** 2)
    if abs(denom) < EPS:
        return float("nan")
    return level_shift(energies, probs, beta, idx=idx, ref=0) / denom


def _hmf_exact_rebuild_distance(rho, beta):
    hmf_exact = -(1.0 / beta) * logm(rho)
    rho_rebuild = normalize_density(expm(-beta * hmf_exact))
    return trace_distance(rho, rho_rebuild)


def _phase_scale_from_var(var):
    if abs(var) < 1e-15:
        return 0.0 + 0.0j
    return np.sqrt(abs(var)) * np.exp(0.5j * np.angle(var))


def _phase_build_k2_base(beta, n_tau, tau_c):
    dt = beta / n_tau
    tau = (np.arange(n_tau) + 0.5) * dt
    d = np.abs(tau[:, None] - tau[None, :])
    return np.exp(-d / tau_c), dt


def _phase_cholesky_psd(k, jitter=1e-12):
    ksym = 0.5 * (k + k.T)
    for p in range(7):
        try:
            return np.linalg.cholesky(ksym + (10**p) * jitter * np.eye(ksym.shape[0]))
        except np.linalg.LinAlgError:
            continue
    vals, vecs = np.linalg.eigh(ksym)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals))


def _phase_analytic_probs(phi, beta=PHASE_BETA, j=PHASE_J, r=PHASE_R):
    omega = np.exp(2j * np.pi / 3.0)
    eig_f = np.array([1.0 + 0.0j, omega, omega**2], dtype=complex)
    eig_f2 = eig_f * eig_f
    k4 = r * np.exp(1j * phi)
    k2 = np.conj(k4)
    w = np.exp(-beta * j * (eig_f + np.conj(eig_f)) + k2 * eig_f2 + k4 * eig_f)
    wr = np.maximum(np.real(w), 0.0)
    return wr / np.sum(wr)


def _phase_mc_probs(phi, tau_c, m_samples, rng, n_tau=PHASE_N_TAU, beta=PHASE_BETA, j=PHASE_J, r=PHASE_R):
    omega = np.exp(2j * np.pi / 3.0)
    eig_f = np.array([1.0 + 0.0j, omega, omega**2], dtype=complex)
    eig_f2 = eig_f * eig_f
    k4 = r * np.exp(1j * phi)
    k2 = np.conj(k4)
    kbase, dt = _phase_build_k2_base(beta, n_tau, tau_c)
    ones = np.ones(n_tau, dtype=float)
    s = (dt**2) * (ones @ kbase @ ones)
    a = (2.0 * k2) / s
    sqrt_a = _phase_scale_from_var(a)
    l = _phase_cholesky_psd(kbase)
    z = rng.normal(0.0, 1.0, size=(m_samples, n_tau))
    x = dt * np.sum(sqrt_a * (z @ l.T), axis=1)

    var_s = (8.0 * k4) / ((n_tau**4) * (dt**4))
    scale_s = _phase_scale_from_var(var_s)
    zs = rng.normal(0.0, 1.0, size=m_samples)
    y = (scale_s * zs) * ((n_tau**2) * (dt**2))

    base = np.exp(-beta * j * (eig_f + np.conj(eig_f))).real
    wbar = np.zeros(3, dtype=complex)
    for k in range(3):
        wbar[k] = base[k] * np.mean(np.exp(eig_f[k] * x + 0.5 * eig_f2[k] * y))
    wr = np.maximum(np.real(wbar), 0.0)
    probs = wr / np.sum(wr)
    imag_leakage = float(np.max(np.abs(np.imag(wbar)) / (np.abs(np.real(wbar)) + 1e-15)))
    return probs, imag_leakage


def _magnetization_levels(n):
    ms = np.arange(-n, n + 1, 2, dtype=int)
    gs = np.array([math.comb(n, (n + int(m)) // 2) for m in ms], dtype=float)
    return ms, gs


def _synth_analytic_probs(n, k2, k4):
    ms, gs = _magnetization_levels(n)
    ex = k2 * ms**2 + k4 * ms**4
    ex = ex - np.max(ex)
    w = gs * np.exp(ex)
    probs = w / np.sum(w)
    return ms, probs


def _synth_complex_scale(var):
    if abs(var) < 1e-15:
        return 0.0 + 0.0j
    return np.sqrt(abs(var)) * np.exp(0.5j * np.angle(var))


def _synth_mc_probs(n, k2, k4, m_samples, rng):
    ms, gs = _magnetization_levels(n)
    sx = _synth_complex_scale(2.0 * k2)
    sy = _synth_complex_scale(8.0 * k4)
    x = sx * rng.standard_normal(int(m_samples))
    y = sy * rng.standard_normal(int(m_samples))
    wm = np.zeros_like(ms, dtype=np.complex128)
    for i, (m, g) in enumerate(zip(ms, gs)):
        wm[i] = g * np.mean(np.exp(m * x + 0.5 * (m**2) * y))
    wr = np.maximum(np.real(wm), 0.0)
    probs = wr / np.sum(wr)
    imag_leakage = float(np.sum(np.abs(np.imag(wm))) / max(np.sum(np.abs(np.real(wm))), 1e-15))
    return ms, probs, imag_leakage


def run_cg(profile, data_dir, seed):
    energies, hs, fvals, f = qutrit_commuting_model()
    rows = []

    for gi, g in enumerate(profile.coupling_list):
        omegas, cs, _ = discretize_ohmic_bath(
            num_modes=profile.mode_count,
            omega_c=profile.omega_c,
            eta=profile.eta,
            g=g,
            omega_max_factor=profile.omega_max_factor,
        )
        lam_disc = lambda_from_discrete(omegas, cs)
        lam_cont = lambda_ohmic_continuum(profile.eta, profile.omega_c, g)

        for bi, beta in enumerate(profile.beta_list):
            rho_analytic = commuting_gaussian_prediction(hs, f, beta, lam_disc)
            p_analytic = diag_probs(rho_analytic)

            rng_scalar = np.random.default_rng(seed + 100000 + 1000 * gi + 37 * bi)
            rho_scalar, p_scalar, _ = scalar_hs_density(
                energies=energies,
                fvals=fvals,
                beta=beta,
                lam=lam_disc,
                n_samples=profile.scalar_samples,
                rng=rng_scalar,
            )
            rng_path = np.random.default_rng(seed + 200000 + 1000 * gi + 37 * bi)
            rho_path, p_path, _ = path_hs_density(
                energies=energies,
                fvals=fvals,
                beta=beta,
                omegas=omegas,
                cs=cs,
                n_tau=profile.path_tau_points,
                n_samples=profile.path_samples,
                rng=rng_path,
            )

            for cutoff in profile.cutoff_list:
                rho_ed = exact_reduced_density_bosonic(hs, f, omegas, cs, cutoff, beta)
                p_ed = diag_probs(rho_ed)
                td_ed_hmf_exact = _hmf_exact_rebuild_distance(rho_ed, beta)
                trace_err_ed, herm_err_ed = density_diagnostics(rho_ed)
                trace_err_an, herm_err_an = density_diagnostics(rho_analytic)
                trace_err_scalar, herm_err_scalar = density_diagnostics(rho_scalar)
                trace_err_path, herm_err_path = density_diagnostics(rho_path)

                row = _blank_row(CG_COLUMNS)
                row.update(
                    {
                        "test_id": "NHS-CG-1-v2",
                        "record_type": "benchmark",
                        "coupling": float(g),
                        "beta": float(beta),
                        "cutoff": int(cutoff),
                        "mode_count": int(profile.mode_count),
                        "lambda_disc": lam_disc,
                        "lambda_cont": lam_cont,
                        "p0_ed": float(p_ed[0]),
                        "p1_ed": float(p_ed[1]),
                        "p2_ed": float(p_ed[2]),
                        "p0_analytic": float(p_analytic[0]),
                        "p1_analytic": float(p_analytic[1]),
                        "p2_analytic": float(p_analytic[2]),
                        "p0_hs_scalar": float(p_scalar[0]),
                        "p1_hs_scalar": float(p_scalar[1]),
                        "p2_hs_scalar": float(p_scalar[2]),
                        "p0_hs_path": float(p_path[0]),
                        "p1_hs_path": float(p_path[1]),
                        "p2_hs_path": float(p_path[2]),
                        "td_ed_analytic": trace_distance(rho_ed, rho_analytic),
                        "td_hs_scalar_analytic": trace_distance(rho_scalar, rho_analytic),
                        "td_hs_path_analytic": trace_distance(rho_path, rho_analytic),
                        "td_hs_scalar_ed": trace_distance(rho_scalar, rho_ed),
                        "td_hs_path_ed": trace_distance(rho_path, rho_ed),
                        "td_ed_hmf_exact": td_ed_hmf_exact,
                        "lambda_est_ed": _lambda_estimate(energies, fvals, p_ed, beta, idx=2),
                        "lambda_est_analytic": _lambda_estimate(energies, fvals, p_analytic, beta, idx=2),
                        "lambda_est_hs_scalar": _lambda_estimate(energies, fvals, p_scalar, beta, idx=2),
                        "lambda_est_hs_path": _lambda_estimate(energies, fvals, p_path, beta, idx=2),
                        "trace_err_ed": trace_err_ed,
                        "trace_err_analytic": trace_err_an,
                        "trace_err_hs_scalar": trace_err_scalar,
                        "trace_err_hs_path": trace_err_path,
                        "herm_err_ed": herm_err_ed,
                        "herm_err_analytic": herm_err_an,
                        "herm_err_hs_scalar": herm_err_scalar,
                        "herm_err_hs_path": herm_err_path,
                    }
                )
                rows.append(row)

    g_ref = float(profile.coupling_list[len(profile.coupling_list) // 2])
    beta_ref = float(profile.beta_list[len(profile.beta_list) // 2])
    cutoff_ref = int(max(profile.cutoff_list))
    omegas_ref, cs_ref, _ = discretize_ohmic_bath(
        num_modes=profile.mode_count,
        omega_c=profile.omega_c,
        eta=profile.eta,
        g=g_ref,
        omega_max_factor=profile.omega_max_factor,
    )
    lam_disc_ref = lambda_from_discrete(omegas_ref, cs_ref)
    rho_analytic_ref = commuting_gaussian_prediction(hs, f, beta_ref, lam_disc_ref)
    rho_ed_ref = exact_reduced_density_bosonic(hs, f, omegas_ref, cs_ref, cutoff_ref, beta_ref)

    for method in ("hs_scalar", "hs_path"):
        for m_samples in profile.mc_scaling_samples:
            n_reps = int(MC_SCALING_REPS_BY_PROFILE.get(profile.name, 2))
            err_to_an = []
            err_to_ed = []
            for ri in range(n_reps):
                if method == "hs_scalar":
                    rng = np.random.default_rng(seed + 300000 + int(m_samples) + 104729 * ri)
                    rho_mc, _, _ = scalar_hs_density(
                        energies=energies,
                        fvals=fvals,
                        beta=beta_ref,
                        lam=lam_disc_ref,
                        n_samples=int(m_samples),
                        rng=rng,
                    )
                else:
                    rng = np.random.default_rng(seed + 400000 + int(m_samples) + 130363 * ri)
                    rho_mc, _, _ = path_hs_density(
                        energies=energies,
                        fvals=fvals,
                        beta=beta_ref,
                        omegas=omegas_ref,
                        cs=cs_ref,
                        n_tau=profile.path_tau_points,
                        n_samples=int(m_samples),
                        rng=rng,
                    )
                err_to_an.append(trace_distance(rho_mc, rho_analytic_ref))
                err_to_ed.append(trace_distance(rho_mc, rho_ed_ref))

            row = _blank_row(CG_COLUMNS)
            row.update(
                {
                    "test_id": "NHS-CG-1-v2",
                    "record_type": "mc_scaling",
                    "coupling": g_ref,
                    "beta": beta_ref,
                    "cutoff": cutoff_ref,
                    "mode_count": int(profile.mode_count),
                    "lambda_disc": lam_disc_ref,
                    "lambda_cont": lambda_ohmic_continuum(profile.eta, profile.omega_c, g_ref),
                    "mc_method": method,
                    "mc_M": int(m_samples),
                    "mc_error_to_analytic": float(np.mean(err_to_an)),
                    "mc_error_to_ed": float(np.mean(err_to_ed)),
                    "mc_error_std_to_analytic": float(np.std(err_to_an, ddof=1) if n_reps > 1 else 0.0),
                    "mc_error_std_to_ed": float(np.std(err_to_ed, ddof=1) if n_reps > 1 else 0.0),
                    "mc_scaling_reps": int(n_reps),
                    "td_ed_hmf_exact": np.nan,
                }
            )
            rows.append(row)

    write_csv(os.path.join(data_dir, "cg_nhs_cg_1_v2.csv"), rows, CG_COLUMNS)


def run_cng(profile, data_dir):
    energies, hs, fvals, f = qutrit_commuting_model()
    rows = []
    spin_omegas = np.asarray(profile.spin_omegas, dtype=float)
    c_template = np.asarray(profile.spin_c_template, dtype=float)
    num_spins = int(spin_omegas.size)

    for beta in profile.beta_list:
        for g in profile.coupling_list:
            couplings = g * c_template
            hb, b_op = spin_bath_ops(num_spins, spin_omegas, couplings)
            hi = np.kron(f, b_op)
            rho_ed = exact_reduced_density(hs, hb, hi, beta)
            p_ed = diag_probs(rho_ed)
            td_ed_hmf_exact = _hmf_exact_rebuild_distance(rho_ed, beta)

            alpha_map, _, stability = integrated_even_cumulants(
                Hb=hb,
                B=b_op,
                beta=beta,
                orders=(2, 4, 6),
                h=profile.fd_step,
            )

            k6_resolved = bool(stability[6] <= K6_STABILITY_MAX)
            alpha6_used = float(alpha_map[6] if k6_resolved else 0.0)
            alpha_map_k246 = dict(alpha_map)
            alpha_map_k246[6] = alpha6_used

            rho_k2, _ = commuting_truncated_prediction(hs, f, beta, alpha_map, max_order=2)
            rho_k24, _ = commuting_truncated_prediction(hs, f, beta, alpha_map, max_order=4)
            rho_k246, _ = commuting_truncated_prediction(hs, f, beta, alpha_map_k246, max_order=6)
            p_k2 = diag_probs(rho_k2)
            p_k24 = diag_probs(rho_k24)
            p_k246 = diag_probs(rho_k246)

            trace_err_ed, herm_err_ed = density_diagnostics(rho_ed)
            trace_err_k2, herm_err_k2 = density_diagnostics(rho_k2)
            trace_err_k24, herm_err_k24 = density_diagnostics(rho_k24)
            trace_err_k246, herm_err_k246 = density_diagnostics(rho_k246)

            td_k2 = trace_distance(rho_ed, rho_k2)
            td_k24 = trace_distance(rho_ed, rho_k24)
            td_k246 = trace_distance(rho_ed, rho_k246)

            row = _blank_row(CNG_COLUMNS)
            row.update(
                {
                    "test_id": "NHS-CNG-1-v2",
                    "beta": float(beta),
                    "coupling": float(g),
                    "num_spins": num_spins,
                    "alpha2": float(alpha_map[2]),
                    "alpha4": float(alpha_map[4]),
                    "alpha6": float(alpha_map[6]),
                    "alpha6_used": alpha6_used,
                    "chi4": float(alpha_map[4] / (alpha_map[2] ** 2 + EPS)),
                    "chi6": float(alpha_map[6] / (abs(alpha_map[2]) ** 3 + EPS)),
                    "stability_rel_2": float(stability[2]),
                    "stability_rel_4": float(stability[4]),
                    "stability_rel_6": float(stability[6]),
                    "k6_resolved": int(k6_resolved),
                    "p0_ed": float(p_ed[0]),
                    "p1_ed": float(p_ed[1]),
                    "p2_ed": float(p_ed[2]),
                    "p0_k2": float(p_k2[0]),
                    "p1_k2": float(p_k2[1]),
                    "p2_k2": float(p_k2[2]),
                    "p0_k24": float(p_k24[0]),
                    "p1_k24": float(p_k24[1]),
                    "p2_k24": float(p_k24[2]),
                    "p0_k246": float(p_k246[0]),
                    "p1_k246": float(p_k246[1]),
                    "p2_k246": float(p_k246[2]),
                    "td_k2": td_k2,
                    "td_k24": td_k24,
                    "td_k246": td_k246,
                    "td_ed_hmf_exact": td_ed_hmf_exact,
                    "fro_k2": fro_error(rho_ed, rho_k2),
                    "fro_k24": fro_error(rho_ed, rho_k24),
                    "fro_k246": fro_error(rho_ed, rho_k246),
                    "improvement_24": float(td_k2 / max(td_k24, EPS)),
                    "improvement_246": float(td_k2 / max(td_k246, EPS)),
                    "shift_lvl1_ed": level_shift(energies, p_ed, beta, idx=1, ref=0),
                    "shift_lvl1_k2": level_shift(energies, p_k2, beta, idx=1, ref=0),
                    "shift_lvl1_k24": level_shift(energies, p_k24, beta, idx=1, ref=0),
                    "shift_lvl1_k246": level_shift(energies, p_k246, beta, idx=1, ref=0),
                    "shift_lvl2_ed": level_shift(energies, p_ed, beta, idx=2, ref=0),
                    "shift_lvl2_k2": level_shift(energies, p_k2, beta, idx=2, ref=0),
                    "shift_lvl2_k24": level_shift(energies, p_k24, beta, idx=2, ref=0),
                    "shift_lvl2_k246": level_shift(energies, p_k246, beta, idx=2, ref=0),
                    "trace_err_ed": trace_err_ed,
                    "trace_err_k2": trace_err_k2,
                    "trace_err_k24": trace_err_k24,
                    "trace_err_k246": trace_err_k246,
                    "herm_err_ed": herm_err_ed,
                    "herm_err_k2": herm_err_k2,
                    "herm_err_k24": herm_err_k24,
                    "herm_err_k246": herm_err_k246,
                }
            )
            rows.append(row)

    write_csv(os.path.join(data_dir, "cng_nhs_cng_1_v2.csv"), rows, CNG_COLUMNS)


def run_phase_clock(profile, data_dir, seed):
    rows = []
    n_phi = int(PHASE_POINTS_BY_PROFILE.get(profile.name, 41))
    phi_grid = np.linspace(-1.0, 1.0, n_phi)
    n_reps = int(PHASE_REPLICAS_BY_PROFILE.get(profile.name, 8))
    m_per_rep = max(2000, int(profile.path_samples // max(n_reps, 1)))
    m_total = int(m_per_rep * n_reps)
    for ti, tau_c in enumerate(PHASE_TAU_CS):
        for pi, phi_over_pi in enumerate(phi_grid):
            phi = float(np.pi * phi_over_pi)
            p_an = _phase_analytic_probs(phi)
            p_mc_reps = np.zeros((n_reps, 3), dtype=float)
            td_reps = np.zeros(n_reps, dtype=float)
            mx_reps = np.zeros(n_reps, dtype=float)
            leak_reps = np.zeros(n_reps, dtype=float)
            for ri in range(n_reps):
                rng = np.random.default_rng(seed + 700000 + 100000 * ti + 1000 * pi + 17 * ri)
                p_mc, imag_leak = _phase_mc_probs(phi, tau_c=tau_c, m_samples=m_per_rep, rng=rng)
                p_mc_reps[ri, :] = p_mc
                td_reps[ri] = 0.5 * np.sum(np.abs(p_an - p_mc))
                mx_reps[ri] = np.max(np.abs(p_an - p_mc))
                leak_reps[ri] = imag_leak
            p_mean = np.mean(p_mc_reps, axis=0)
            p_std = np.std(p_mc_reps, axis=0, ddof=1) if n_reps > 1 else np.zeros(3, dtype=float)
            row = {
                "test_id": "NHS-CNG-PHASE-v2",
                "beta": PHASE_BETA,
                "j": PHASE_J,
                "r": PHASE_R,
                "tau_c": float(tau_c),
                "n_tau": int(PHASE_N_TAU),
                "phi_over_pi": float(phi_over_pi),
                "phi": phi,
                "k2_real": float(np.real(PHASE_R * np.exp(-1j * phi))),
                "k2_imag": float(np.imag(PHASE_R * np.exp(-1j * phi))),
                "k4_real": float(np.real(PHASE_R * np.exp(1j * phi))),
                "k4_imag": float(np.imag(PHASE_R * np.exp(1j * phi))),
                "p0_analytic": float(p_an[0]),
                "p1_analytic": float(p_an[1]),
                "p2_analytic": float(p_an[2]),
                "p0_mc": float(p_mean[0]),
                "p1_mc": float(p_mean[1]),
                "p2_mc": float(p_mean[2]),
                "p0_mc_std": float(p_std[0]),
                "p1_mc_std": float(p_std[1]),
                "p2_mc_std": float(p_std[2]),
                "trace_distance": float(np.mean(td_reps)),
                "trace_distance_std": float(np.std(td_reps, ddof=1) if n_reps > 1 else 0.0),
                "max_abs_error": float(np.mean(mx_reps)),
                "max_abs_error_std": float(np.std(mx_reps, ddof=1) if n_reps > 1 else 0.0),
                "imag_leakage": float(np.mean(leak_reps)),
                "imag_leakage_std": float(np.std(leak_reps, ddof=1) if n_reps > 1 else 0.0),
                "phase_replicas": int(n_reps),
                "mc_M_per_rep": int(m_per_rep),
                "mc_M_total": int(m_total),
            }
            rows.append(row)
    write_csv(os.path.join(data_dir, "cng_nhs_phase_clock_v2.csv"), rows, PHASE_COLUMNS)


def run_synthetic_nonpsd(profile, data_dir, seed):
    rows = []
    curve_rows = []
    branch_defs = [("PSD", SYNTH_K4_ABS), ("nonPSD", -SYNTH_K4_ABS)]
    m_max = int(max(profile.mc_scaling_samples))
    for bi, (branch, k4) in enumerate(branch_defs):
        ms, p_an = _synth_analytic_probs(SYNTH_N, SYNTH_K2, k4)
        for m_samples in profile.mc_scaling_samples:
            rng = np.random.default_rng(seed + 600000 + 10000 * bi + int(m_samples))
            _, p_mc, imag_leak = _synth_mc_probs(SYNTH_N, SYNTH_K2, k4, m_samples, rng)
            rows.append(
                {
                    "test_id": "NHS-SYNTH-NONPSD-v2",
                    "branch": branch,
                    "N": int(SYNTH_N),
                    "k2": float(SYNTH_K2),
                    "k4": float(k4),
                    "mc_M": int(m_samples),
                    "trace_distance": float(0.5 * np.sum(np.abs(p_an - p_mc))),
                    "max_abs_error": float(np.max(np.abs(p_an - p_mc))),
                    "imag_leakage": imag_leak,
                }
            )
            if int(m_samples) == m_max:
                for m_val, p0, p1 in zip(ms, p_an, p_mc):
                    curve_rows.append(
                        {
                            "test_id": "NHS-SYNTH-NONPSD-v2",
                            "branch": branch,
                            "N": int(SYNTH_N),
                            "k2": float(SYNTH_K2),
                            "k4": float(k4),
                            "mc_M": int(m_samples),
                            "m": int(m_val),
                            "p_analytic": float(p0),
                            "p_mc": float(p1),
                        }
                    )

    write_csv(os.path.join(data_dir, "cng_nhs_synth_nonpsd_v2.csv"), rows, SYNTH_COLUMNS)
    write_csv(os.path.join(data_dir, "cng_nhs_synth_nonpsd_curves_v2.csv"), curve_rows, SYNTH_CURVE_COLUMNS)


def write_manifest(out_dir, args, profile_name, duration):
    payload = {
        "suite": "nested_hs_v2",
        "profile": profile_name,
        "regime": args.regime,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "duration_sec": float(duration),
    }
    write_json(os.path.join(out_dir, "manifest.json"), payload)


def parse_args():
    p = argparse.ArgumentParser(description="Run commuting-only nested HS v2 suite")
    p.add_argument("--regime", choices=["cg", "cng", "all"], default="all")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="quick")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    profile = PROFILES[args.profile]

    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results_v2"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)
    ensure_dir(data_dir)

    start = time.time()
    regimes = REGIMES if args.regime == "all" else (args.regime,)
    if "cg" in regimes:
        run_cg(profile, data_dir, args.seed)
    if "cng" in regimes:
        run_synthetic_nonpsd(profile, data_dir, args.seed)
        run_cng(profile, data_dir)
        run_phase_clock(profile, data_dir, args.seed)
    elapsed = time.time() - start

    write_manifest(out_dir, args, profile.name, elapsed)
    print(f"nested_hs_v2 suite complete in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
