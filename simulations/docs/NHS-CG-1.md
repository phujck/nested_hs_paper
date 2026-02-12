# NHS-CG-1

## Hamiltonian
`H_tot = H_S + H_B + H_I` with
- `H_S = diag(0,0.7,1.6)`, `f=diag(0,1,2)`,
- `H_B = sum_k omega_k(a_k^dag a_k + 1/2)`,
- `H_I = f \otimes sum_k c_k x_k`, `x_k=(a_k+a_k^dag)/sqrt(2 omega_k)`.

## Exact reference
`rho_S^ED = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]`.

## Prediction
`rho_S^pred propto exp[-beta(H_S - lambda f^2)]`.

## What is plotted
- ED and prediction overlays for `p0`.
- Residual `D(rho_S^ED, rho_S^pred)` vs cutoff.
