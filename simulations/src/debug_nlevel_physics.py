import numpy as np
from suite_common_v2 import (
    spin_bath_ops,
    exact_reduced_density,
    commuting_truncated_prediction,
    diag_probs,
    source_deformed_log_partition,
    finite_difference_derivative
)
import math

SPIN_OMEGAS = [0.60, 0.80, 1.00, 1.20, 1.40]
SPIN_C_TEMPLATE = [0.55, 0.8, 1.0, 1.2, 1.35]

def nlevel_clock_model(N, J=0.2):
    omega = np.exp(2j * np.pi / N)
    fvals = np.array([omega**k for k in range(N)], dtype=complex)
    f = np.diag(fvals)
    Hs = J * (f + f.conj().T)
    Hs = 0.5 * (Hs + Hs.conj().T)
    evals = np.linalg.eigvalsh(Hs)
    Hs = Hs - np.min(evals) * np.eye(N)
    return Hs, f

def debug_point():
    N = 5
    beta = 2.5
    g = 0.4
    
    print(f"--- Debugging N={N}, beta={beta}, g={g} ---")
    Hs, f = nlevel_clock_model(N)
    print("Hs diag:", np.diag(Hs).real)
    print("f diag: ", np.diag(f))
    
    num_spins = len(SPIN_OMEGAS)
    spin_omegas = np.array(SPIN_OMEGAS)
    c_template = np.array(SPIN_C_TEMPLATE)
    couplings = g * c_template
    
    Hb, B_op = spin_bath_ops(num_spins, spin_omegas, couplings)
    Hi = np.kron(f, B_op)
    
    # ED
    rho_ed = exact_reduced_density(Hs, Hb, Hi, beta)
    p_ed = diag_probs(rho_ed)
    print("\nED Populations:")
    print(p_ed)
    
    # Cumulants
    # Go up to 10 to see convergence
    orders = (2, 4, 6, 8, 10)
    # Use larger radius for higher orders (need 2R >= Order)
    # Radius 6 allows up to order 12
    fun = lambda th: source_deformed_log_partition(Hb, B_op, beta, th)
    alpha_map = {}
    for order in orders:
        # Use radius 6
        d_h2 = finite_difference_derivative(fun, order=order, h=1e-3, radius=6)
        alpha_map[order] = float((1.0 / beta) * d_h2 / math.factorial(order))
    
    print("\nCumulants (alpha_2n):")
    for k in orders:
        print(f"  alpha_{k}: {alpha_map[k]:.6f}")
        
    # Predictions
    print("\nPredictions:")
    for order in orders:
        rho, _ = commuting_truncated_prediction(Hs, f, beta, alpha_map, max_order=order)
        p = diag_probs(rho)
        err = np.max(np.abs(p - p_ed))
        print(f"  K{order}: {p} (Max Err: {err:.2e})")

    # Manually check loop renormalization logic
    a2 = alpha_map[2]
    a6 = alpha_map[6]
    a10 = alpha_map[10]
    # N=4: f^2 is the generator. f^4=I. f^6=f^2. f^8=I. f^10=f^2.
    # Effective alpha for f^2 term:
    alpha_eff = a2 + a6 + a10
    print(f"\nManual Renormalization Check:")
    print(f"  Sum(alpha_4n+2) = {a2} + {a6} + {a10} = {alpha_eff:.6f}")
    
    # Construct manual HMF with just this effective alpha
    # H_eff = Hs - alpha_eff * f^2
    f2 = f @ f
    H_manual = Hs - (alpha_eff / beta) * f2
    from scipy.linalg import expm
    rho_manual = expm(-beta * H_manual)
    rho_manual = rho_manual / np.trace(rho_manual)
    p_manual = np.diag(rho_manual).real
    print(f"  Manual Loop Sum Prediction: {p_manual}")
    print(f"  Err vs ED: {np.max(np.abs(p_manual - p_ed)):.2e}")

if __name__ == "__main__":
    debug_point()
