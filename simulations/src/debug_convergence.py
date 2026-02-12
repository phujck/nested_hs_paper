import numpy as np
import math

def analytic_logZ(beta, spin_omegas, couplings, theta):
    # E_k = sqrt(omega_k^2 + theta^2 g_k^2)
    # Z_k = 2 cosh(beta E_k)
    # log Z = sum_k log(Z_k)
    # theta can be complex
    E = np.sqrt(spin_omegas**2 + (theta * couplings)**2 + 0j)
    return np.sum(np.log(2 * np.cosh(beta * E)))

def finite_difference_derivative(func, x0, order, h=0.1):
    # Central differences
    # generic weights? 
    # Let's just use naive finite diff for debugging
    if order == 0: return func(x0)
    if order == 2:
        return (func(x0+h) - 2*func(x0) + func(x0-h)) / h**2
    if order == 4:
        # -1/6 f(x-2h) + 2 f(x-h) - 13/3 f(x) + ...?
        # Use simpler recursion or standard coefficients
        # 1 -4 6 -4 1
        return (func(x0+2*h) - 4*func(x0+h) + 6*func(x0) - 4*func(x0-h) + func(x0-2*h)) / h**4
    if order == 6:
        # 1 -6 15 -20 15 -6 1
        return (func(x0+3*h) - 6*func(x0+2*h) + 15*func(x0+h) - 20*func(x0) 
                + 15*func(x0-h) - 6*func(x0-2*h) + func(x0-3*h)) / h**6
    return 0

def check_convergence():
    beta = 2.5
    SPIN_OMEGAS = np.array([0.60, 0.80, 1.00, 1.20, 1.40])
    SPIN_C_TEMPLATE = np.array([0.55, 0.8, 1.0, 1.2, 1.35])
    
    g_vals = [0.5, 1.0, 1.5]
    
    print(f"{'g':<5} {'Exact(1)':<15} {'K2':<15} {'K2+4':<15} {'K2+4+6':<15}")
    
    for g in g_vals:
        couplings = g * SPIN_C_TEMPLATE
        
        # Exact Phi(1)
        phi_1 = analytic_logZ(beta, SPIN_OMEGAS, couplings, 1.0)
        phi_0 = analytic_logZ(beta, SPIN_OMEGAS, couplings, 0.0) # To subtract Z(0) offset if needed? (Cumulants usually give logZ(lambda) - logZ(0))
        # Cumulants are derivatives at 0.
        # logZ(1) = logZ(0) + sum alpha_k/k!
        
        # Compute derivatives at 0
        fun = lambda th: analytic_logZ(beta, SPIN_OMEGAS, couplings, th)
        
        # Explicitly calculate alpha_k without 1/beta factor, as we are summing LogZ directly
        # alpha_k in paper was (1/beta) * d^k logZ / d lambda^k
        # But here checking logZ sum: sum (beta * alpha_k) * 1^k / k!
        # Equivalently sum (d^k Phi / d theta^k) / k!
        
        d0 = fun(0)
        d2 = finite_difference_derivative(fun, 0, 2, h=0.1)
        d4 = finite_difference_derivative(fun, 0, 4, h=0.1)
        d6 = finite_difference_derivative(fun, 0, 6, h=0.1)
        
        approx_2 = d0 + d2/math.factorial(2)
        approx_4 = approx_2 + d4/math.factorial(4)
        approx_6 = approx_4 + d6/math.factorial(6)
        
        print(f"{g:<5.1f} {phi_1.real:<15.4f} {approx_2.real:<15.4f} {approx_4.real:<15.4f} {approx_6.real:<15.4f}")

        # Check lambda = 1j (Imaginary coupling - critical for N=4)
        print(f"--- g={g} lambda=i ---")
        try:
            phi_i = analytic_logZ(beta, SPIN_OMEGAS, couplings, 1.0j)
            
            # Taylor expansion for phi(i): sum alpha_k * (i)^k / k!
            # d0 is phi(0).
            # d2 is alpha_2 * beta * 2!
            # approx = d0 + d2*(i)**2/2! + d4*(i)**4/4! ...
            # i^2 = -1, i^4 = 1, i^6 = -1
            
            approx_i_2 = d0 - d2/math.factorial(2)
            approx_i_4 = approx_i_2 + d4/math.factorial(4)
            approx_i_6 = approx_i_4 - d6/math.factorial(6)
            
            print(f"Exact(i): {phi_i:.4f}")
            print(f"K2(i):    {approx_i_2:.4f}")
            print(f"K2+4(i):  {approx_i_4:.4f}")
            print(f"K2+4+6(i):{approx_i_6:.4f}")
        except Exception as e:
            print(f"Error at lambda=i: {e}")

if __name__ == "__main__":
    check_convergence()
