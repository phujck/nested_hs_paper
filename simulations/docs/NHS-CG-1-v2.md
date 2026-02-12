# NHS-CG-1-v2

## Model

Total Hamiltonian:
`H_tot = H_S + H_B + H_I`

- `H_S = diag(0, 0.9, 1.8)`
- `f = diag(0, 1, 2)` (commuting with `H_S`)
- `H_B = sum_k omega_k (a_k^dag a_k + 1/2)`
- `H_I = f ⊗ sum_k c_k x_k`, `x_k=(a_k+a_k^dag)/sqrt(2 omega_k)`

Couplings are generated from an Ohmic discretisation:
`J(omega)=2 eta g^2 omega exp(-omega/omega_c)`.

## Exact Reference and Prediction

- Exact reference:
  `rho_S^ED = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]`.
- Commuting Gaussian prediction:
  `rho_S^an ∝ exp[-beta(H_S - lambda_disc f^2)]`,
  `lambda_disc = sum_k c_k^2/(2 omega_k^2)`.
- Stochastic estimators:
  scalar HS and path HS (same discretised kernel).

## Plotted Quantities

- Coupled-level population: `p2 = <2|rho_S|2>`.
- Residual distance: `D = (1/2)||rho_a-rho_b||_1`.
- Effective shift estimator:
  `lambda_est = (E2-E0 + beta^{-1} ln(p2/p0)) / (f2^2-f0^2)`.

## Objections Answered

- “Analytic agreement is fitted”: disproved by direct ED overlays at identical parameters.
- “HS Monte Carlo is only qualitative”: scalar/path HS compared directly to analytic and ED.
- “Cutoff artifacts dominate”: explicit convergence panel versus bosonic cutoff.

## Acceptance Criteria

- `td_ed_analytic` decreases with cutoff for most `(beta,g)` points.
- HS residuals sit within Monte Carlo uncertainty and scale approximately as `1/sqrt(M)`.
- Density diagnostics satisfy normalisation and Hermiticity tolerances.
