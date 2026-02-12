"""
Multi-Qubit Magnetisation Model — Nested HS Sampling
======================================================

Demonstrates that a quartic cumulant induces nonlocal diagonal interactions
in a multi-qubit system coupled through S_z. Validates HS sampling in
both PSD and non-PSD regimes.

Produces a 3-panel figure:
  - Panel 1: PSD distribution (k4 > 0)
  - Panel 2: Non-PSD distribution (k4 < 0, contour rotation)
  - Panel 3: Trace-distance convergence vs 1/sqrt(M)

Usage:
    python multiqubit_sz.py [--N 4] [--M 200000] [--seed 42]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import comb


# ── Parameters ──────────────────────────────────────────────────────────────

N_QUBITS = 4
K2 = 0.08
K4_PSD = 0.003     # PSD case
K4_NON_PSD = -0.003  # non-PSD case


# ── Utility ─────────────────────────────────────────────────────────────────

def scale_from_var(var):
    """Complex scale so that (scale*Z)^2 has expectation var when Z~N(0,1)."""
    if var == 0:
        return 0.0 + 0.0j
    return np.sqrt(np.abs(var)) * np.exp(1j * 0.5 * np.angle(var))


def magnetisation_sectors(N):
    """Return magnetisation eigenvalues m and degeneracies g(m)."""
    m_values = np.arange(-N, N + 1, 2)
    g_values = np.array([comb(N, (N + m) // 2, exact=True) for m in m_values],
                        dtype=float)
    return m_values, g_values


def exact_distribution(m_values, g_values, k2, k4):
    """Exact equilibrium distribution from the commuting-sector theorem."""
    log_w = k2 * m_values**2 + k4 * m_values**4
    log_w -= np.max(log_w)
    w = g_values * np.exp(log_w)
    return w / np.sum(w)


# ── Vectorised HS sampling ─────────────────────────────────────────────────

def hs_sample(m_values, g_values, k2, k4, M, seed):
    """
    Vectorised HS sampling.
    e^{k2*m^2 + k4*m^4} = E_X,Y[ e^{mX + m^2*Y/2} ]
    with X ~ N(0, 2*k2), Y ~ N(0, 8*k4) (or contour-rotated for k4 < 0).
    """
    rng = np.random.default_rng(seed)
    n_sec = len(m_values)

    # X field
    scale_X = scale_from_var(2.0 * k2)
    X = scale_X * rng.standard_normal(M)  # (M,)

    # Y field
    scale_Y = scale_from_var(8.0 * k4)
    Y = scale_Y * rng.standard_normal(M)  # (M,)

    # Vectorised: W[s, k] = exp(m_k * X_s + 0.5 * m_k^2 * Y_s)
    m = m_values.astype(float)
    m2 = m**2
    W = np.exp(np.outer(X, m) + 0.5 * np.outer(Y, m2))  # (M, n_sec)

    wbar = g_values * np.mean(W, axis=0)
    wbar_real = np.maximum(wbar.real, 0)
    imag_leakage = np.max(np.abs(wbar.imag) / np.maximum(np.abs(wbar.real), 1e-30))

    p = wbar_real / np.sum(wbar_real)
    return p, imag_leakage


def trace_distance(p, q):
    """Trace distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p - q))


# ── Figure ──────────────────────────────────────────────────────────────────

def make_figure(outdir, N, M, seed):
    m_values, g_values = magnetisation_sectors(N)
    n_sec = len(m_values)

    # Exact
    p_exact_psd = exact_distribution(m_values, g_values, K2, K4_PSD)
    p_exact_npsd = exact_distribution(m_values, g_values, K2, K4_NON_PSD)

    # MC
    p_mc_psd, leak_psd = hs_sample(m_values, g_values, K2, K4_PSD, M, seed)
    p_mc_npsd, leak_npsd = hs_sample(m_values, g_values, K2, K4_NON_PSD, M, seed + 1)

    print(f"PSD:      TD={trace_distance(p_mc_psd, p_exact_psd):.2e}, "
          f"imag leak={leak_psd:.2e}")
    print(f"Non-PSD:  TD={trace_distance(p_mc_npsd, p_exact_npsd):.2e}, "
          f"imag leak={leak_npsd:.2e}")

    # Convergence sweep
    M_values = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    M_values = [mv for mv in M_values if mv <= 5 * M]
    td_psd, td_npsd = [], []
    for mv in M_values:
        p_p, _ = hs_sample(m_values, g_values, K2, K4_PSD, mv, seed)
        td_psd.append(trace_distance(p_p, p_exact_psd))
        p_n, _ = hs_sample(m_values, g_values, K2, K4_NON_PSD, mv, seed + 1)
        td_npsd.append(trace_distance(p_n, p_exact_npsd))

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(n_sec)
    width = 0.35

    # Panel 1: PSD
    ax = axes[0]
    ax.bar(x - width / 2, p_exact_psd, width, label="Analytic",
           color="#457b9d", alpha=0.85)
    ax.bar(x + width / 2, p_mc_psd, width, label="MC (HS)",
           color="#e63946", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}" for m in m_values], fontsize=8)
    ax.set_xlabel(r"$m$", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(rf"PSD: $k_2={K2},\; k_4={K4_PSD}$", fontsize=11)
    ax.legend(fontsize=9)

    # Panel 2: non-PSD
    ax = axes[1]
    ax.bar(x - width / 2, p_exact_npsd, width, label="Analytic",
           color="#457b9d", alpha=0.85)
    ax.bar(x + width / 2, p_mc_npsd, width, label="MC (contour)",
           color="#2a9d8f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}" for m in m_values], fontsize=8)
    ax.set_xlabel(r"$m$", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(rf"Non-PSD: $k_2={K2},\; k_4={K4_NON_PSD}$", fontsize=11)
    ax.legend(fontsize=9)

    # Panel 3: convergence
    ax = axes[2]
    ax.loglog(M_values, td_psd, "o-", color="#e63946", label="PSD", linewidth=1.5)
    ax.loglog(M_values, td_npsd, "s-", color="#2a9d8f", label="Non-PSD", linewidth=1.5)
    scale = td_psd[0] * np.sqrt(M_values[0])
    M_arr = np.array(M_values, dtype=float)
    ax.loglog(M_arr, scale / np.sqrt(M_arr), "--", color="gray", alpha=0.5,
              label=r"$\propto 1/\sqrt{M}$")
    ax.set_xlabel(r"$M$", fontsize=12)
    ax.set_ylabel("Trace distance", fontsize=12)
    ax.set_title("Convergence", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_multiqubit.pdf"), dpi=150)
    fig.savefig(os.path.join(outdir, "fig_multiqubit.png"), dpi=150)
    plt.close(fig)
    print("[Multi-qubit figure] Saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-qubit Sz HS sampling")
    parser.add_argument("--N", type=int, default=N_QUBITS, help="Number of qubits")
    parser.add_argument("--M", type=int, default=200_000, help="MC samples per config")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(__file__), "..", "results", "figures"
    )
    os.makedirs(outdir, exist_ok=True)

    N, M, seed = args.N, args.M, args.seed
    m_values, g_values = magnetisation_sectors(N)

    print(f"=== Multi-Qubit S_z Model ===")
    print(f"N={N}, k2={K2}, k4_PSD={K4_PSD}, k4_non-PSD={K4_NON_PSD}, M={M}")
    print(f"Sectors: m={list(m_values)}, g={list(g_values.astype(int))}")
    print(f"Output: {outdir}\n")

    make_figure(outdir, N, M, seed)

    # Physical interpretation
    print(f"\n=== Physical interpretation ===")
    print(f"(S_z)^2 contains {N} 1-body + {N*(N-1)//2} 2-body ZZ terms.")
    print(f"(S_z)^4 gives up to 4-body ZZZZ diagonal interactions.")
    print(f"Non-Gaussianity reshapes the sector distribution but preserves")
    print(f"the block-diagonal structure. Contour rotation handles k4 < 0.")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
