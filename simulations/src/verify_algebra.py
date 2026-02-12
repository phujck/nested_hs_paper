import numpy as np
import scipy.linalg
from scipy.linalg import expm
from suite_common_v2 import spin_bath_ops, commuting_truncated_prediction, diag_probs, partial_trace

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

def verify_algebra():
    N = 4
    J = 0.2
    beta = 2.5
    
    # 1. Define Model
    omega = np.exp(2j * np.pi / N)
    fvals = np.array([omega**k for k in range(N)], dtype=complex)
    f = np.diag(fvals)
    
    rng = np.random.default_rng(42)
    energies = np.sort(rng.uniform(0, 2.0, N))
    Hs = np.diag(energies)
    
    print("--- Algebraic Structure ---")
    print(f"f diagonals: {fvals}")
    f4 = np.linalg.matrix_power(f, 4)
    print(f"f^4 diagonals: {np.diag(f4)}")
    is_identity = np.allclose(f4, np.eye(N))
    print(f"Is f^4 Identity? {is_identity}")
    
    # 2. Define Bath
    SPIN_OMEGAS = [0.60, 0.80, 1.00, 1.20, 1.40]
    SPIN_C_TEMPLATE = [0.55, 0.8, 1.0, 1.2, 1.35]
    g = 0.5 # Safe coupling
    couplings = np.array(SPIN_C_TEMPLATE) * g
    
    # 3. Compute Analytic Cumulants
    from plot_nlevel_paper import get_analytic_cumulants 
    # Need to make sure plot_nlevel_paper is importable or copy function
    # I'll rely on it being in same dir or just use suite
    
    # Let's just import the function from plot_nlevel_paper.py to ensure identical logic
    try:
        from plot_nlevel_paper import get_analytic_cumulants
    except ImportError:
        # Fallback if path issue
        import sys
        sys.path.append('.')
        from plot_nlevel_paper import get_analytic_cumulants

    # Pass step size if possible, or modify plot_nlevel_paper default
    # Looking at plot_nlevel_paper, finite_difference_derivative takes h=0.1 default.
    # I need to modify `get_analytic_cumulants` in plot_nlevel_paper.py to accept h, or change default there.
    # Let's change default in plot_nlevel_paper.py first.
    pass
    print("\n--- Cumulants ---")
    print(f"alpha_2: {alpha[2]}")
    print(f"alpha_4: {alpha[4]}")
    print(f"alpha_6: {alpha[6]}")
    
    # 4. Compute Predictions
    print("\n--- Predictions ---")
    
    # K2
    rho_k2, heff2 = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=2)
    p_k2 = diag_probs(rho_k2)
    
    # K2+4
    rho_k24, heff24 = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=4)
    p_k24 = diag_probs(rho_k24)
    
    # Check Heff difference
    diff_heff = heff24 - heff2
    print(f"Heff(K2+4) - Heff(K2) diagonals:\n{np.diag(diff_heff)}")
    expected_shift = -alpha[4] * 1.0 # f^4=I, so shift is -alpha4
    print(f"Expected Shift (-alpha4): {expected_shift}")
    
    print(f"\nPopulations K2:   {p_k2}")
    print(f"Populations K2+4: {p_k24}")
    
    dist = np.linalg.norm(p_k2 - p_k24)
    print(f"Distance K2 vs K2+4: {dist:.2e}")
    
    # Exact
    Hb, B_op = spin_bath_ops(len(SPIN_OMEGAS), SPIN_OMEGAS, couplings)
    Hi = np.kron(f, B_op)
    rho_ed = local_exact_reduced_density(Hs, Hb, Hi, beta)
    p_ed = diag_probs(rho_ed)
    
    print(f"Populations Exact:{p_ed}")
    
    # K2+4+6
    rho_k246, _ = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=6)
    p_k246 = diag_probs(rho_k246)
    print(f"Populations K246: {p_k246}")
    
    print(f"Distance K246 vs Exact: {np.linalg.norm(p_k246 - p_ed):.2e}")

    print("\n\n=== N=5 Check ===")
    N = 5
    omega = np.exp(2j * np.pi / N)
    fvals = np.array([omega**k for k in range(N)], dtype=complex)
    f = np.diag(fvals)
    # Re-roll energies for N=5
    rng = np.random.default_rng(42)
    energies = np.sort(rng.uniform(0, 2.0, N))
    Hs = np.diag(energies)
    
    rho_k2, _ = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=2)
    p_k2 = diag_probs(rho_k2)
    
    rho_k24, _ = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=4)
    p_k24 = diag_probs(rho_k24)
    
    rho_k246, _ = commuting_truncated_prediction(Hs, f, beta, alpha, max_order=6)
    p_k246 = diag_probs(rho_k246)
    
    Hi = np.kron(f, B_op)
    rho_ed = local_exact_reduced_density(Hs, Hb, Hi, beta)
    p_ed = diag_probs(rho_ed)
    
    print(f"Exact: {p_ed}")
    print(f"K2:    {p_k2}")
    print(f"K2+4:  {p_k24}")
    print(f"K246:  {p_k246}")
    
    print(f"Dist K2 vs Exact: {np.linalg.norm(p_k2 - p_ed):.2e}")
    print(f"Dist K246 vs Exact: {np.linalg.norm(p_k246 - p_ed):.2e}")

if __name__ == "__main__":
    verify_algebra()
