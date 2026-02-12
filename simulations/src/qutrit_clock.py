"""
Qutrit Clock Model — Discretised Functional HS Sampling
=========================================================

Validates the commuting-sector theorem for a qutrit (Z_3 clock coupling)
with second and fourth cumulant contributions.

Produces:
  - Figure 1: Populations vs phase (analytic + MC at multiple tau_c)
  - Figure 2: CRN error curves + replica bands
  - Figure 3: sqrt(M) scaling collapse

Usage:
    python qutrit_clock.py [--M 120000] [--seed 42] [--outdir path]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ───────────────────────────────────────────────────────────────

OMEGA = np.exp(2j * np.pi / 3)
EIG_F = np.array([1.0, OMEGA, OMEGA**2], dtype=complex)
EIG_F2 = EIG_F**2

J = 0.6
BETA = 1.0
R = 0.45
N_TAU = 18
TAU_CS = [0.04, 0.12, 0.35]
MARKERS = ["o", "s", "^"]


# ── Utility ─────────────────────────────────────────────────────────────────

def scale_from_var(var):
    """Complex scale so that (scale*Z)^2 has expectation var when Z~N(0,1)."""
    if var == 0:
        return 0.0 + 0.0j
    return np.sqrt(np.abs(var)) * np.exp(1j * 0.5 * np.angle(var))


def analytic_probs(phi, beta=BETA, J_val=J, r=R):
    """Exact populations for the qutrit clock model at given phase phi."""
    k4 = r * np.exp(1j * phi)
    k2 = np.conj(k4)
    lam = EIG_F
    w = np.exp(-beta * J_val * (lam + np.conj(lam)) + k2 * lam**2 + k4 * lam)
    w = w.real
    w = np.maximum(w, 0)
    return w / np.sum(w)


def build_K2_base(beta, N, tau_c):
    """Build N×N exponential correlation kernel."""
    dt = beta / N
    tau = (np.arange(N) + 0.5) * dt
    D = np.abs(tau[:, None] - tau[None, :])
    Kbase = np.exp(-D / tau_c)
    return Kbase, dt


def cholesky_psd(K, jitter=1e-12):
    """Robust Cholesky with jitter fallback."""
    Ksym = 0.5 * (K + K.T)
    for k in range(7):
        try:
            return np.linalg.cholesky(Ksym + (10**k) * jitter * np.eye(Ksym.shape[0]))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh(Ksym)
    w = np.clip(w, 0, None)
    return V @ np.diag(np.sqrt(w))


# ── Vectorised MC ───────────────────────────────────────────────────────────

def mc_probs_functional(phi, N=N_TAU, tau_c=0.12, M=60_000, seed=0):
    """
    Full discretised functional MC for one phi value.
    Vectorised: no per-sample loop.
    """
    rng = np.random.default_rng(seed)
    k4 = R * np.exp(1j * phi)
    k2 = np.conj(k4)

    Kbase, dt = build_K2_base(BETA, N, tau_c)
    ones = np.ones(N)
    S = (dt**2) * (ones @ Kbase @ ones)
    A = (2.0 * k2) / S
    sqrtA = scale_from_var(A)

    L = cholesky_psd(Kbase)
    z = rng.normal(0.0, 1.0, size=(M, N))
    X = dt * np.sum(sqrtA * (z @ L.T), axis=1)  # (M,)

    var_s = (8.0 * k4) / ((N**4) * (dt**4))
    scale_s = scale_from_var(var_s)
    s = scale_s * rng.normal(0.0, 1.0, size=M)
    Y = s * ((N**2) * (dt**2))  # (M,)

    base = np.exp(-BETA * J * (EIG_F + np.conj(EIG_F))).real  # (3,)
    wbar = np.zeros(3, dtype=complex)
    for k in range(3):
        wbar[k] = base[k] * np.mean(np.exp(EIG_F[k] * X + 0.5 * EIG_F2[k] * Y))

    wbar = wbar.real
    wbar = np.maximum(wbar, 0)
    return wbar / np.sum(wbar)


def mc_err_curve_precomp(phis, L, dt, N, Z_field, Z_s):
    """
    Fast CRN error curve with precomputed random draws.
    Z_field: (M, N), Z_s: (M,)
    """
    ones = np.ones(N)
    v = L.T @ ones
    S = (dt**2) * np.dot(v, v)

    G = Z_field @ L.T               # (M, N)
    Svec = dt * np.sum(G, axis=1)   # (M,) real

    cY = (N**2) * (dt**2)
    denom = (N**4) * (dt**4)
    base = np.exp(-BETA * J * (EIG_F + np.conj(EIG_F))).real

    errs = np.zeros(len(phis))
    for i, phi in enumerate(phis):
        k4 = R * np.exp(1j * phi)
        k2 = np.conj(k4)
        pA = analytic_probs(phi)

        A = (2.0 * k2) / S
        sqrtA = scale_from_var(A)

        var_s = (8.0 * k4) / denom
        scale_s = scale_from_var(var_s)

        X = sqrtA * Svec
        Y_vals = (scale_s * Z_s) * cY

        W = np.exp(np.outer(X, EIG_F) + 0.5 * np.outer(Y_vals, EIG_F2))
        wbar = base * np.mean(W, axis=0)
        wbar = np.maximum(wbar.real, 0)
        pM = wbar / np.sum(wbar)

        errs[i] = np.max(np.abs(pM - pA))
    return errs


# ── Figures ─────────────────────────────────────────────────────────────────

def figure1_populations_collapse(outdir, M, seed):
    """Figure 1: Populations vs phase with correlation-length collapse."""
    phis_dense = np.linspace(-np.pi, np.pi, 400)
    phis_mc = np.linspace(-np.pi, np.pi, 9)

    # Analytic
    p_exact = np.array([analytic_probs(phi) for phi in phis_dense])

    # MC for each tau_c
    p_mc_all = {}
    for tci, tau_c in enumerate(TAU_CS):
        p_mc = np.zeros((len(phis_mc), 3))
        for i, phi in enumerate(phis_mc):
            p_mc[i] = mc_probs_functional(
                phi, N=N_TAU, tau_c=tau_c, M=M,
                seed=seed + 200 * tci + i
            )
        p_mc_all[tau_c] = p_mc

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#e63946", "#457b9d", "#2a9d8f"]
    labels_p = [r"$p_0$", r"$p_1$", r"$p_2$"]

    for k in range(3):
        ax.plot(phis_dense / np.pi, p_exact[:, k], color=colors[k],
                linewidth=2, label=labels_p[k])

    for tci, tau_c in enumerate(TAU_CS):
        p_mc = p_mc_all[tau_c]
        for k in range(3):
            label = rf"$\tau_c={tau_c}$" if k == 0 else None
            ax.plot(phis_mc / np.pi, p_mc[:, k], marker=MARKERS[tci],
                    linestyle="None", color=colors[k], markersize=6,
                    markeredgecolor="k", markeredgewidth=0.4, label=label)

    ax.set_xlabel(r"$\phi/\pi$ in $\kappa_4 = r\,e^{i\phi}$", fontsize=13)
    ax.set_ylabel("Population", fontsize=13)
    ax.set_title(
        rf"Correlation-length collapse: $J={J}$, $r={R}$, $N={N_TAU}$, $M={M:,}$/pt",
        fontsize=11
    )
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_populations_collapse.pdf"), dpi=150)
    fig.savefig(os.path.join(outdir, "fig1_populations_collapse.png"), dpi=150)
    plt.close(fig)
    print("[Figure 1] Populations + collapse saved.")

    # Print diagnostics
    for tau_c in TAU_CS:
        p_mc = p_mc_all[tau_c]
        errs = []
        for i, phi in enumerate(phis_mc):
            pA = analytic_probs(phi)
            errs.append(np.max(np.abs(p_mc[i] - pA)))
        errs = np.array(errs)
        print(f"  tau_c={tau_c:.2f}: median max-err={np.median(errs):.2e}, "
              f"max-err={np.max(errs):.2e}")


def figure2_crn_and_replicas(outdir, M_crn, M_rep, R_rep, seed):
    """Figure 2: CRN error curves + replica statistics."""
    phis = np.linspace(-np.pi, np.pi, 7)

    # Precompute L and dt for each tau_c
    L_map, dt_map = {}, {}
    for tau_c in TAU_CS:
        Kbase, dt = build_K2_base(BETA, N_TAU, tau_c)
        L_map[tau_c] = cholesky_psd(Kbase)
        dt_map[tau_c] = dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: CRN error curves
    rng = np.random.default_rng(seed)
    Z_field = rng.normal(0.0, 1.0, size=(M_crn, N_TAU))
    Z_s = rng.normal(0.0, 1.0, size=M_crn)

    for tci, tau_c in enumerate(TAU_CS):
        e = mc_err_curve_precomp(phis, L_map[tau_c], dt_map[tau_c],
                                  N_TAU, Z_field, Z_s)
        ax1.plot(phis / np.pi, e, marker=MARKERS[tci], linewidth=2,
                 label=rf"$\tau_c={tau_c}$")

    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\phi/\pi$", fontsize=12)
    ax1.set_ylabel(r"$\max_k |p_k^{\mathrm{MC}} - p_k^{\mathrm{exact}}|$",
                    fontsize=12)
    ax1.set_title(f"CRN error curves ($M={M_crn:,}$)", fontsize=11)
    ax1.legend(fontsize=10, frameon=False)

    # Panel 2: Replica bands
    rng0 = np.random.default_rng(seed + 7777)
    for tau_c in TAU_CS:
        E = np.zeros((R_rep, len(phis)))
        for rep in range(R_rep):
            Zf = rng0.normal(0.0, 1.0, size=(M_rep, N_TAU))
            Zs = rng0.normal(0.0, 1.0, size=M_rep)
            E[rep] = mc_err_curve_precomp(phis, L_map[tau_c], dt_map[tau_c],
                                           N_TAU, Zf, Zs)
        m = E.mean(axis=0)
        s = E.std(axis=0)
        line, = ax2.plot(phis / np.pi, m, linewidth=2)
        ax2.fill_between(phis / np.pi, np.maximum(m - s, 1e-18), m + s,
                          alpha=0.2, color=line.get_color())

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\phi/\pi$", fontsize=12)
    ax2.set_ylabel(r"max error (mean $\pm$ std)", fontsize=12)
    ax2.set_title(f"Replica bands ($R={R_rep}$, $M={M_rep:,}$/pt)", fontsize=11)
    ax2.legend([rf"$\tau_c={tc}$" for tc in TAU_CS], fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_crn_replicas.pdf"), dpi=150)
    fig.savefig(os.path.join(outdir, "fig2_crn_replicas.png"), dpi=150)
    plt.close(fig)
    print("[Figure 2] CRN + replicas saved.")


def figure3_M_scaling(outdir, seed):
    """Figure 3: sqrt(M) * error collapse."""
    phis = np.linspace(-np.pi, np.pi, 7)
    Ms = [20_000, 80_000, 320_000]
    tau_c0 = 0.12

    Kbase, dt = build_K2_base(BETA, N_TAU, tau_c0)
    L = cholesky_psd(Kbase)

    fig, ax = plt.subplots(figsize=(8, 5))
    for M_val in Ms:
        rng = np.random.default_rng(seed + M_val)
        Zf = rng.normal(0.0, 1.0, size=(M_val, N_TAU))
        Zs = rng.normal(0.0, 1.0, size=M_val)
        e = mc_err_curve_precomp(phis, L, dt, N_TAU, Zf, Zs)
        ax.plot(phis / np.pi, np.sqrt(M_val) * e, marker="o", linewidth=2,
                label=rf"$M={M_val:,}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\phi/\pi$", fontsize=12)
    ax.set_ylabel(r"$\sqrt{M}\;\max_k |p_k^{\mathrm{MC}} - p_k^{\mathrm{exact}}|$",
                   fontsize=12)
    ax.set_title(rf"$\sqrt{{M}}$ scaling collapse ($\tau_c={tau_c0}$, $N={N_TAU}$)",
                  fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_M_scaling.pdf"), dpi=150)
    fig.savefig(os.path.join(outdir, "fig3_M_scaling.png"), dpi=150)
    plt.close(fig)
    print("[Figure 3] M-scaling collapse saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qutrit clock HS sampling")
    parser.add_argument("--M", type=int, default=120_000,
                        help="MC samples per point (figures 1 & 2 CRN)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(__file__), "..", "results", "figures"
    )
    os.makedirs(outdir, exist_ok=True)
    M = args.M
    seed = args.seed

    print(f"=== Qutrit Clock Model ===")
    print(f"J={J}, beta={BETA}, |kappa_4|={R}, N_tau={N_TAU}, M={M}")
    print(f"Output: {outdir}\n")

    # Stress-point quick check
    phi_stress = 0.9 * np.pi
    p_exact = analytic_probs(phi_stress)
    p_mc = mc_probs_functional(phi_stress, M=M, tau_c=0.12, seed=seed)
    err = np.max(np.abs(p_mc - p_exact))
    print(f"Stress point phi=0.9*pi:")
    print(f"  Analytic: {p_exact}")
    print(f"  MC:       {p_mc}")
    print(f"  Max error: {err:.2e}\n")

    figure1_populations_collapse(outdir, M, seed)
    print()
    figure2_crn_and_replicas(outdir, M_crn=M, M_rep=50_000, R_rep=12, seed=seed)
    print()
    figure3_M_scaling(outdir, seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
