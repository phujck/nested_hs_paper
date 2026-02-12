import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

from suite_common_v2 import (
    ensure_dir,
    spin_bath_ops,
    exact_reduced_density,
    integrated_even_cumulants,
    commuting_truncated_prediction,
    diag_probs,
    normalize_density,
    to_jsonable
)

# Hardcoded bath parameters from "publish" profile for consistency
SPIN_OMEGAS = [0.60, 0.80, 1.00, 1.20, 1.40]
SPIN_C_TEMPLATE = [0.55, 0.8, 1.0, 1.2, 1.35]

def nlevel_clock_model(N, J=0.6):
    """
    Constructs the N-level clock model Hamiltonian and coupling operator.
    Hs = J * (f + f^dag)
    f = diag(1, w, w^2, ..., w^{N-1}) with w = exp(2pi i / N)
    """
    omega = np.exp(2j * np.pi / N)
    fvals = np.array([omega**k for k in range(N)], dtype=complex)
    f = np.diag(fvals)
    Hs = J * (f + f.conj().T)
    # Ensure Hs is Hermitian (it is by construction, but float/complex can start to drift)
    Hs = 0.5 * (Hs + Hs.conj().T)
    # Shift energies so ground state is 0 for nicer plotting (optional, but good for p_k logic)
    evals = np.linalg.eigvalsh(Hs)
    Hs = Hs - np.min(evals) * np.eye(N)
    return Hs, f, fvals

def run_nlevel_scan(N, beta, g_list, out_dir):
    print(f"Running N={N}, beta={beta:.2f} scan...")
    
    Hs, f, fvals = nlevel_clock_model(N)
    
    # Storage
    results = {
        "g": [],
        "p_ed": [],
        "p_k2": [],
        "p_k24": [],
        "p_k246": [],
        "alpha": {"2": [], "4": [], "6": []}
    }

    num_spins = len(SPIN_OMEGAS)
    spin_omegas = np.array(SPIN_OMEGAS)
    c_template = np.array(SPIN_C_TEMPLATE)

    for g in g_list:
        # 1. Build Bath
        couplings = g * c_template
        Hb, B_op = spin_bath_ops(num_spins, spin_omegas, couplings)
        
        # 2. ED Solution
        Hi = np.kron(f, B_op)
        rho_ed = exact_reduced_density(Hs, Hb, Hi, beta)
        p_ed = diag_probs(rho_ed)
        
        # 3. Cumulants
        # We need the cumulants for *this* bath instance (g dependent)
        alpha_map, _, stability = integrated_even_cumulants(
            Hb=Hb, B=B_op, beta=beta, orders=(2, 4, 6), h=1e-3
        )
        
        # 4. Truncations
        rho_k2, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=2)
        rho_k24, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=4)
        rho_k246, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=6)
        
        results["g"].append(g)
        results["p_ed"].append(p_ed)
        results["p_k2"].append(diag_probs(rho_k2))
        results["p_k24"].append(diag_probs(rho_k24))
        results["p_k246"].append(diag_probs(rho_k246))
        results["alpha"]["2"].append(alpha_map[2])
        results["alpha"]["4"].append(alpha_map[4])
        results["alpha"]["6"].append(alpha_map[6])
        
    return results

def plot_results(results, N, beta, out_dir):
    g_vals = np.array(results["g"])
    p_ed = np.array(results["p_ed"])
    p_k2 = np.array(results["p_k2"])
    p_k24 = np.array(results["p_k24"])
    
    # Plot 1: Populations vs g for ALL levels
    # Since N might be 4 or 5, we can use a cycle of colors
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    
    for k in range(N):
        # ED lines (Solid)
        ax.plot(g_vals, p_ed[:, k], '-', color=colors[k], label=f"Level {k} (ED)" if k==0 else None, linewidth=2.5, alpha=0.9)
        # K2 lines (Dotted)
        ax.plot(g_vals, p_k2[:, k], ':', color=colors[k], label=f"Level {k} (K2)" if k==0 else None, linewidth=2.0, alpha=0.8)
        # K2+K4 lines (Dashed) -- hopefully closer to ED
        # ax.plot(g_vals, p_k24[:, k], '--', color=colors[k], label=f"Level {k} (K2+K4)" if k==0 else None, linewidth=2.0, alpha=0.8)

    # Simplified Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=2, linestyle='-'),
                    Line2D([0], [0], color='black', lw=2, linestyle=':')]
    ax.legend(custom_lines, ['Exact (ED)', 'Gaussian (K2)'], loc='upper right')
    
    # Add colored text annotations for levels instead of a messy legend
    for k in range(N):
        y_end = p_ed[-1, k]
        ax.text(g_vals[-1]*1.02, y_end, f"|{k}>", color=colors[k], verticalalignment='center', fontweight='bold')

    ax.set_xlabel("Coupling $g$")
    ax.set_ylabel("Population $p_k$")
    ax.set_title(f"N={N} Clock Model, $\\beta={beta}$ (Low T)\nFreezing bath causes Gaussian failure")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"nhs_demo_N{N}_beta{beta}.pdf"))
    fig.savefig(os.path.join(out_dir, f"nhs_demo_N{N}_beta{beta}.png"))
    plt.close(fig)
    
    # Plot 2: Deviation / "Attachment"
    # Show the error (p_exact - p_approx) for each level
    # This visualizes "where the cumulants go"
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for k in range(N):
        err_k2 = p_ed[:, k] - p_k2[:, k]
        err_k24 = p_ed[:, k] - p_k24[:, k]
        
        # Plot K2 error as filled area
        ax2.fill_between(g_vals, err_k2, 0, color=colors[k], alpha=0.2, label=f"L{k} K2 Error" if k==0 else None)
        # Plot K2+K4 error as line
        ax2.plot(g_vals, err_k24, '--', color=colors[k], linewidth=2, label=f"L{k} K2+K4 resid" if k==0 else None)
        
    ax2.set_xlabel("Coupling $g$")
    ax2.set_ylabel("Population Error ($p_{ED} - p_{approx}$)")
    ax2.set_title(f"Cumulant impact on levels (N={N})\nShaded: K2 Error (Non-Gaussianity) | Dashed: Residual after K4")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f"nhs_demo_err_N{N}.png"))
    plt.close(fig2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4, help="Number of levels")
    parser.add_argument("--beta", type=float, default=2.0, help="Inverse temperature")
    parser.add_argument("--g_max", type=float, default=0.8, help="Max coupling")
    parser.add_argument("--points", type=int, default=15, help="Number of points in g-scan")
    parser.add_argument("--outdir", type=str, default="demo_output")
    args = parser.parse_args()
    
    ensure_dir(args.outdir)
    
    g_list = np.linspace(0.05, args.g_max, args.points)
    results = run_nlevel_scan(args.N, args.beta, g_list, args.outdir)
    plot_results(results, args.N, args.beta, args.outdir)
    print(f"Done. Results in {args.outdir}")

if __name__ == "__main__":
    main()
