import numpy as np
import math
from suite_common_v2 import (
    spin_bath_ops,
    source_deformed_log_partition,
    finite_difference_derivative
)

SPIN_OMEGAS = [0.60, 0.80, 1.00, 1.20, 1.40]
SPIN_C_TEMPLATE = [0.55, 0.8, 1.0, 1.2, 1.35]

def test_stability():
    beta = 2.5
    g = 1.5
    
    # Setup bath
    num_spins = len(SPIN_OMEGAS)
    spin_omegas = np.array(SPIN_OMEGAS)
    c_template = np.array(SPIN_C_TEMPLATE)
    couplings = g * c_template
    Hb, B_op = spin_bath_ops(num_spins, spin_omegas, couplings)
    
    fun = lambda th: source_deformed_log_partition(Hb, B_op, beta, th)
    
    h_vals = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]
    
    print(f"Testing derivative stability for g={g}, beta={beta}")
    print(f"{'h':<10} {'alpha_2':<15} {'alpha_4':<15} {'alpha_6':<15}")
    
    for h in h_vals:
        # Order 2
        d2 = finite_difference_derivative(fun, order=2, h=h, radius=4)
        a2 = (1.0 / beta) * d2 / math.factorial(2)
        
        # Order 4
        d4 = finite_difference_derivative(fun, order=4, h=h, radius=4)
        a4 = (1.0 / beta) * d4 / math.factorial(4)
        
        # Order 6
        d6 = finite_difference_derivative(fun, order=6, h=h, radius=6)
        a6 = (1.0 / beta) * d6 / math.factorial(6)
        
        print(f"{h:<10.0e} {a2:<15.6f} {a4:<15.6f} {a6:<15.6f}")

if __name__ == "__main__":
    test_stability()
