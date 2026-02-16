
import sys
import os

# Add the source directory to the path so we can import the modules
sys.path.append(r'c:\Users\gerar\VScodeProjects\nested_hs_paper\simulations\src')

import numpy as np
from suite_common_v2 import spin_bath_ops, integrated_even_cumulants, finite_difference_derivative, source_deformed_log_partition
import math

def check_odd_cumulant():
    # Define a simple spin bath exactly as in the code
    num_spins = 2
    omega_list = [1.0, 1.2]
    coupling_list = [0.5, 0.7]
    beta = 1.0
    
    hb, b_op = spin_bath_ops(num_spins, omega_list, coupling_list)
    
    # We want to check the 3rd order cumulant
    # The function integrated_even_cumulants is named 'even', but the logic inside is generic for any order
    # if we pass odd orders to it.
    
    orders = (2, 3, 4)
    alpha, deriv, stability = integrated_even_cumulants(hb, b_op, beta, orders=orders)
    
    print(f"Alpha 2: {alpha[2]}")
    print(f"Alpha 3: {alpha[3]}")
    print(f"Alpha 4: {alpha[4]}")
    
    if abs(alpha[3]) < 1e-10:
        print("Alpha 3 is effectively ZERO.")
    else:
        print("Alpha 3 is NON-ZERO.")

    # Let's also verify the symmetry argument directly
    # Check if Z_B(theta) is symmetric
    theta = 0.1
    z_plus = source_deformed_log_partition(hb, b_op, beta, theta)
    z_minus = source_deformed_log_partition(hb, b_op, beta, -theta)
    
    print(f"Z_B(theta): {z_plus}")
    print(f"Z_B(-theta): {z_minus}")
    print(f"Diff: {abs(z_plus - z_minus)}")

if __name__ == "__main__":
    check_odd_cumulant()
