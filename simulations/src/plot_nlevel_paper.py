import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suite_common_v2 import (
    ensure_dir,
    spin_bath_ops,
    exact_reduced_density,
    commuting_truncated_prediction,
    diag_probs,
    finite_difference_derivative,
    partial_trace
)
from scipy.linalg import expm

# Local local_thermal_density and local_exact_reduced_density to support complex clock couplings
def local_thermal_density(H, beta):
    # No hermitize call!
    # Shift by real part of min eval to avoid overflow
    evals = np.linalg.eigvals(H)
    shift = np.min(np.real(evals))
    return expm(-beta * (H - shift * np.eye(H.shape[0])))

def local_exact_reduced_density(Hs, Hb, Hi, beta):
    ds = Hs.shape[0]
    db = Hb.shape[0]
    htot = np.kron(Hs, np.eye(db, dtype=complex)) + np.kron(np.eye(ds, dtype=complex), Hb) + Hi
    
    # Use local non-hermitizing thermal density
    rho_tot = local_thermal_density(htot, beta)
    rho_s = partial_trace(rho_tot, [ds, db], keep=[0])
    
    # Normalize manually
    tr = np.trace(rho_s)
    return rho_s / tr

# Hardcoded bath parameters (same as "publish" profile)
SPIN_OMEGAS = [0.60, 0.80, 1.00, 1.20, 1.40]
SPIN_C_TEMPLATE = [0.55, 0.8, 1.0, 1.2, 1.35]

def nlevel_clock_model(N, J=0.2):
    # Commuting case as requested by user.
    # f is diagonal clock operator. 
    # Hs is diagonal random energies.
    omega = np.exp(2j * np.pi / N)
    fvals = np.array([omega**k for k in range(N)], dtype=complex)
    f = np.diag(fvals)
    
    # Random diagonal energies (fixed seed for reproducibility implicitly by hardcoding)
    # Using a simple spread to ensure distinct populations
    rng = np.random.default_rng(42)
    energies = np.sort(rng.uniform(0, 2.0, N))
    Hs = np.diag(energies)
    
    # Eigenvectors are just identity since Hs is diagonal
    evecs = np.eye(N)
    
    return Hs, f, evecs

def analytic_logZ(beta, spin_omegas, couplings, theta):
    # H_k = omega_k sigma_x + theta g_k sigma_z
    # E_k = sqrt(omega_k^2 + theta^2 g_k^2)
    # Z_k = 2 cosh(beta E_k)
    # log Z = sum_k log(2 cosh(beta E_k))
    E = np.sqrt(spin_omegas**2 + (theta * couplings)**2)
    return np.sum(np.log(2 * np.cosh(beta * E)))

def get_analytic_cumulants(beta, spin_omegas, couplings, orders=(2, 4, 6)):
    import math
    fun = lambda th: analytic_logZ(beta, spin_omegas, couplings, th)
    alpha = {}
    for order in orders:
        # Scalar FD is stable enough for higher orders if function is smooth
        # Stability check showed h=0.05 is stable.
        d_h = finite_difference_derivative(fun, order=order, h=0.05, radius=6)
        alpha[order] = (1.0 / beta) * d_h / math.factorial(order)
    return alpha

def get_all_pops(rho):
    return np.real(np.diag(rho))

def run_point(N, beta, g):
    Hs, f, evecs = nlevel_clock_model(N)
    num_spins = len(SPIN_OMEGAS)
    spin_omegas = np.array(SPIN_OMEGAS)
    c_template = np.array(SPIN_C_TEMPLATE)
    
    couplings = g * c_template
    Hb, B_op = spin_bath_ops(num_spins, spin_omegas, couplings)
    Hi = np.kron(f, B_op)
    
    # Exact
    # Use local non-hermitizing version to match the theoretical model used for this demo
    rho_ed = local_exact_reduced_density(Hs, Hb, Hi, beta)
    p_ed = get_all_pops(rho_ed)
    
    # Analytic cumulants
    alpha_map = get_analytic_cumulants(beta, spin_omegas, couplings, orders=(2, 4, 6))
    
    # Approximations
    rho_k2, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=2)
    rho_k24, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=4)
    rho_k246, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=6)
    
    return p_ed, get_all_pops(rho_k2), get_all_pops(rho_k24), get_all_pops(rho_k246)

def generate_figure(out_dir):
    # Safe to extend range now with analytic cumulants
    # Limit g to radius of convergence (g < omega_min approx 0.6)
    # User requested bigger range. Extending to 1.2 to show breakdown/behavior.
    g_vals = np.linspace(0.01, 1.2, 40)
    beta = 2.5
    
    N_cases = [4, 5]
    
    # Run simulation
    data = {}
    for N in N_cases:
        data[N] = {"g": [], "ed": [], "k2": [], "k24": [], "k246": []}
        print(f"Simulating N={N}...")
        for g in g_vals:
            p_ed, p_k2, p_k24, p_k246 = run_point(N, beta, g)
            data[N]["g"].append(g)
            data[N]["ed"].append(p_ed)
            data[N]["k2"].append(p_k2)
            data[N]["k24"].append(p_k24)
            data[N]["k246"].append(p_k246)
            
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Standard colors from the suite
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, N in enumerate(N_cases):
        ax = axes[i]
        g = np.array(data[N]["g"])
        
        # Data is list of arrays [ [p0, p1...], [p0, p1...] ... ]
        # Convert to array of shape (num_g, N)
        ed = np.array(data[N]["ed"])
        k2 = np.array(data[N]["k2"])
        k24 = np.array(data[N]["k24"])
        k246 = np.array(data[N]["k246"])
        
        # Plot ALL levels
        # Use color to distinguish levels? Or line style for Approx vs Exact?
        # User implies we should see ALL levels.
        # Strategy:
        # Levels 0, 1, 2... distinguished by color group (Blue, Orange...)
        # Approx distinguished by line style. 
        # But that's too messy.
        
        # Strategy 2 (Cleaner):
        # Plot Difference P_exact - P_approx? 
        # No, user wants to see "lock to each other and miss exact". implies absolute values.
        
        # Strategy 3:
        # Plot P_0 and P_1 (most populated).
        # P_0: Thick Black (Exact), Blue Dashed (K2), Red Dot (K24), Green DashDot (K246)
        # P_1: Thin Black (Exact), Thin Blue Dashed (K2)... 
        # Actually, let's just plot ONE LEVEL (P0) but be very sure it's correct?
        # User said "continued to only plot the ground state".
        # So I MUST plot more levels.
        
        # Let's plot P0 and P1.
        # Shift P1 up/down? No.
        # Use different colors for levels.
        # P0: Black/Grays. P1: Blues.
        
        # Actually, looking at previous plots, P0 ~ 1.0. Others small.
        # Let's plot P0 (high) and P1 (low) on same axes.
        
        # Custom legend handling
        if i == 0:
            # Create dummy lines for legend
            # P0 group
            ax.plot([], [], '-', color='black', label=r'Exact $P_0$')
            ax.plot([], [], '--', color='tab:blue', label=r'K2 $P_0$')
            ax.plot([], [], ':', color='tab:red', label=r'K2+4 $P_0$')
            
            # P1 group
            ax.plot([], [], '-', color='gray', label=r'Exact $P_1$')
            ax.plot([], [], '--', color='cyan', label=r'K2 $P_1$')
            
            ax.legend(loc='lower left', fontsize=8, framealpha=0.9, ncol=2)

        for level in [0, 1]: 
            c_base = 'black' if level==0 else 'gray'
            if level == 1: c_base = '#555555' 
            
            # No labels in actual plot calls to avoid duplication/warning
            ax.plot(g, ed[:, level], '-', color=c_base, lw=2.5, alpha=0.8)
            
            c_k2 = 'tab:blue' if level==0 else 'cyan'
            c_k24 = 'tab:red' if level==0 else 'magenta'
            c_k246 = 'tab:green' if level==0 else 'lime'
            
            ax.plot(g, k2[:, level], '--', color=c_k2, lw=2.0, alpha=0.9)
            ax.plot(g, k24[:, level], ':', color=c_k24, lw=2.5, alpha=0.9)
            ax.plot(g, k246[:, level], '-.', color=c_k246, lw=2.5, alpha=0.9)

        ax.set_xlabel(r"Coupling strength $g$")
        ax.set_ylabel(r"Populations $P_k$")
        ax.set_ylim(-0.05, 1.05)
        
        # Simpler titles
        title_str = r"(A) $N=4$: $K_4$ invisible, $K_6$ required" if N==4 else r"(B) $N=5$: $K_4$ active, $K_6$ refines"
        ax.set_title(title_str, fontsize=11)
        ax.grid(alpha=0.22, linewidth=0.6)
        
        
        # Add legend only to the first plot to save space, or both if they fit
        if i == 0:
            ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        if i == 1:
             ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    savename = os.path.join(out_dir, "nhs_algebraic_closure.pdf")
    plt.savefig(savename, dpi=170)
    print(f"Saved figure to {savename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="../results_v2/figures")
    args = parser.parse_args()
    generate_figure(args.outdir)
